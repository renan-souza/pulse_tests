import ast
import inspect
import struct
import time
import resource

import numpy as np
import numba as nb

nb.config.BOUNDSCHECK = 0

from numba import njit, float32, int64, uint8
from pulse_fast_ext import Pulse


SCALE = float32(10.0)
BIAS = float32(0.5)

PULSE_MAGIC = 0x31534C50
PULSE_VERSION = 1


@njit(cache=True, fastmath=True, inline="always")
def pulse_log(acc_view, loss_view, i, capacity, acc, loss):
    idx = i % capacity
    acc_view[idx] = uint8(acc * SCALE + BIAS)
    loss_view[idx] = uint8(loss * SCALE + BIAS)
    return i + 1


class _RewriteLoggerLog(ast.NodeTransformer):
    def __init__(self, logger_name, i_name, cap_name, acc_view_name, loss_view_name):
        super().__init__()
        self.logger_name = logger_name
        self.i_name = i_name
        self.cap_name = cap_name
        self.acc_view_name = acc_view_name
        self.loss_view_name = loss_view_name

    def visit_Expr(self, node):
        node = self.generic_visit(node)

        call = node.value
        if not isinstance(call, ast.Call):
            return node

        func = call.func
        if not isinstance(func, ast.Attribute):
            return node

        if func.attr != "log":
            return node

        if not isinstance(func.value, ast.Name):
            return node

        if func.value.id != self.logger_name:
            return node

        if len(call.args) != 2:
            return node

        new_call = ast.Call(
            func=ast.Name(id="pulse_log", ctx=ast.Load()),
            args=[
                ast.Name(id=self.acc_view_name, ctx=ast.Load()),
                ast.Name(id=self.loss_view_name, ctx=ast.Load()),
                ast.Name(id=self.i_name, ctx=ast.Load()),
                ast.Name(id=self.cap_name, ctx=ast.Load()),
                call.args[0],
                call.args[1],
            ],
            keywords=[],
        )

        return ast.Assign(
            targets=[ast.Name(id=self.i_name, ctx=ast.Store())],
            value=new_call,
        )


def _make_compilable_user_fn(user_fn):
    import textwrap

    src = inspect.getsource(user_fn)

    # Normalize indentation safely
    src = textwrap.dedent(src)

    # Ensure function body is properly indented
    lines = src.splitlines()
    if not lines:
        raise ValueError("Empty source for user_fn")

    header = lines[0]
    body = lines[1:]

    if not header.lstrip().startswith("def "):
        raise ValueError("Expected a function definition")

    if body:
        body = ["    " + line if line.strip() else line for line in body]
        src = "\n".join([header] + body)
    else:
        src = header + "\n    pass"

    mod = ast.parse(src)

    fn_defs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    if not fn_defs:
        raise ValueError("Could not find a function definition in user_fn source")
    fn = fn_defs[0]

    if not fn.args.args or fn.args.args[0].arg != "logger":
        raise ValueError("user_fn must have signature user_fn(logger, ...) with first arg named 'logger'")

    logger_name = "logger"
    i_name = "_pulse_i"
    cap_name = "_pulse_capacity"
    acc_view_name = "_pulse_acc_view"
    loss_view_name = "_pulse_loss_view"

    new_args = [
        ast.arg(arg=acc_view_name),
        ast.arg(arg=loss_view_name),
        ast.arg(arg="i0"),
        ast.arg(arg=cap_name),
    ] + fn.args.args[1:]

    fn.args.args = new_args
    fn.returns = None

    init_i = ast.Assign(
        targets=[ast.Name(id=i_name, ctx=ast.Store())],
        value=ast.Call(func=ast.Name(id="int64", ctx=ast.Load()), args=[ast.Name(id="i0", ctx=ast.Load())], keywords=[]),
    )

    rewriter = _RewriteLoggerLog(
        logger_name=logger_name,
        i_name=i_name,
        cap_name=cap_name,
        acc_view_name=acc_view_name,
        loss_view_name=loss_view_name,
    )
    fn.body = [rewriter.visit(b) for b in fn.body]
    fn.body = [b for b in fn.body if b is not None]

    fn.body.insert(0, init_i)

    has_return = any(isinstance(n, ast.Return) for n in fn.body)
    if not has_return:
        fn.body.append(ast.Return(value=ast.Name(id=i_name, ctx=ast.Load())))

    ast.fix_missing_locations(mod)

    ns = {}
    g = {
        "np": np,
        "nb": nb,
        "njit": njit,
        "float32": float32,
        "int64": int64,
        "uint8": uint8,
        "pulse_log": pulse_log,
        "SCALE": SCALE,
        "BIAS": BIAS,
    }

    code = compile(mod, filename="<pulse_user_fn>", mode="exec")
    exec(code, g, ns)

    new_fn = ns.get(fn.name, None)
    if new_fn is None:
        raise RuntimeError("Failed to synthesize transformed user function")

    return new_fn


def build_compiled_runner(user_fn):
    transformed = _make_compilable_user_fn(user_fn)
    return njit(cache=True, fastmath=True)(transformed)


def read_pulse_file(filepath):
    import pandas as pd

    with open(filepath, "rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError("Missing/short header")

        magic_u32, version_u32, nrecords_u64 = struct.unpack("<IIQ", header)
        raw = f.read()
        data = np.frombuffer(raw, dtype=np.uint8)

    n = int(nrecords_u64)
    expected_bytes = 2 * n
    if data.size < expected_bytes:
        raise ValueError(f"File truncated: expected {expected_bytes} bytes, got {data.size}")

    data = data[:expected_bytes]
    acc_u8 = data[:n]
    loss_u8 = data[n : 2 * n]

    acc = (acc_u8.astype(np.float32) - float(BIAS)) / float(SCALE)
    loss = (loss_u8.astype(np.float32) - float(BIAS)) / float(SCALE)

    return pd.DataFrame({"acc": acc, "loss": loss}), magic_u32, version_u32, n


def train_model(logger, max_iterations, alpha, beta):
    current_loss = float32(1.0)
    current_acc = float32(0.1)
    for k in range(max_iterations):
        current_loss *= alpha
        current_acc += beta
        logger.log(current_acc, current_loss)


class PulseManager:
    def __init__(self, capacity, user_fn, *args, filepath="pulse_log.bin"):
        self.capacity = int(capacity)
        self.filepath = filepath
        self.args = args

        self.backend = None
        self.acc_view = None
        self.loss_view = None

        self.i = np.int64(0)
        self._iterations_run = 0

        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")

        self.compiled_user_fn = build_compiled_runner(user_fn)

    def _warmup(self):
        max_iterations, alpha, beta = self.args
        iters = np.int64(1)
        self.i = self.compiled_user_fn(
            self.acc_view,
            self.loss_view,
            self.i,
            np.int64(self.capacity),
            iters,
            alpha,
            beta,
        )
        self.i = np.int64(0)

    def pulse(self):
        self._iterations_run = int(self.args[0])
        max_iterations, alpha, beta = self.args
        self.i = self.compiled_user_fn(
            self.acc_view,
            self.loss_view,
            self.i,
            np.int64(self.capacity),
            max_iterations,
            alpha,
            beta,
        )

    def logger_size(self):
        n = self._iterations_run
        if n <= 0:
            return 0
        return self.capacity if n >= self.capacity else n

    def __enter__(self):
        self.backend = Pulse(self.capacity)
        self.acc_view, self.loss_view = self.backend.arrays()
        self.i = np.int64(0)
        self._warmup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend.set_cursor(self.logger_size())
        self.backend.flush(self.filepath)


def run_simulation():
    STORAGE_CAPACITY = 256
    TRAIN_ITERATIONS = np.int64(300_000_000)
    ALPHA = np.float32(0.9999999999)
    BETA = np.float32(0.00000000001)

    train_args = (TRAIN_ITERATIONS, ALPHA, BETA)

    with PulseManager(STORAGE_CAPACITY, train_model, *train_args) as pm:
        t0 = time.time()
        pm.pulse()
        t1 = time.time()
        print(f"Loop completed in: {t1 - t0:.4f}s")
        print(f"Logger size (valid records): {pm.logger_size()}")

    df, magic, version, nrecords = read_pulse_file("pulse_log.bin")
    print(f"File header: magic=0x{magic:08x} version={version} nrecords={nrecords}")

    print(f"Read back {len(df)} records.")
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Peak Memory: {mem_mb:.2f} MB")

    expected = min(int(TRAIN_ITERATIONS), STORAGE_CAPACITY)
    print(f"Size of df={len(df)} expected={expected}")
    assert len(df) == expected
    print(df.head())
    print(df.tail())
    print("Verification Successful.")


if __name__ == "__main__":
    run_simulation()
