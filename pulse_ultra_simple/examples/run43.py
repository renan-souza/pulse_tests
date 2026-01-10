# examples/run43.py
import os
import struct
import sys
import time
from time import perf_counter

import ctypes
import numpy as np
import numba as nb
import resource
from numba import njit, types
from numba.extending import intrinsic
from numba.core.typing.templates import signature
from llvmlite import ir
from llvmlite import binding as llvm

nb.config.BOUNDSCHECK = 0

ENABLE_PROFILING = True

MAGIC = 0x31534C50
VERSION = 7
HEADER_FMT = "<IIQdQdB7s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

LOG_EVERY = 1
ITERS = np.int64(100_000_000)

FRONT_CAP = 256
NSHARDS = 4


def _import_repo_root_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import pulse_rt_ext as _rt
    return _rt


rt = _import_repo_root_module()


@intrinsic
def rdtsc(typingctx):
    sig = signature(types.uint64)

    def codegen(context, builder, sig_obj, args):
        f_type = ir.FunctionType(ir.IntType(64), [])
        f = builder.module.declare_intrinsic("llvm.readcyclecounter", [], f_type)
        return builder.call(f, [], name="tsc")

    return sig, codegen


@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def rdtsc_func():
    return rdtsc()


lib = ctypes.CDLL(rt.__file__)
llvm.add_symbol("pulse_copy256", ctypes.cast(lib.pulse_copy256, ctypes.c_void_p).value)
llvm.add_symbol("pulse_write_shards", ctypes.cast(lib.pulse_write_shards, ctypes.c_void_p).value)


@intrinsic
def pulse_copy256_intrinsic(
    typingctx,
    back_acc_ty,
    back_loss_ty,
    back_seq_ty,
    back_ticks_ty,
    base_ty,
    front_acc_ty,
    front_loss_ty,
    front_seq_ty,
    front_ticks_ty,
    has_time_ty,
):
    sig = signature(
        types.int32,
        types.uintp,
        types.uintp,
        types.uintp,
        types.uintp,
        types.uint64,
        types.uintp,
        types.uintp,
        types.uintp,
        types.uintp,
        types.int32,
    )

    def codegen(context, builder, sig_obj, args):
        fn_ty = ir.FunctionType(
            ir.IntType(32),
            [
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(32),
            ],
        )
        mod = builder.module
        fn = mod.globals.get("pulse_copy256")
        if fn is None:
            fn = ir.Function(mod, fn_ty, name="pulse_copy256")
            fn.linkage = "external"
        return builder.call(fn, list(args))

    return sig, codegen


@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def _pulse_copy256(
    back_acc_ptr_u,
    back_loss_ptr_u,
    back_seq_ptr_u,
    back_ticks_ptr_u,
    base_u64,
    front_acc_ptr_u,
    front_loss_ptr_u,
    front_seq_ptr_u,
    front_ticks_ptr_u,
    has_time_i32,
):
    return pulse_copy256_intrinsic(
        back_acc_ptr_u,
        back_loss_ptr_u,
        back_seq_ptr_u,
        back_ticks_ptr_u,
        base_u64,
        front_acc_ptr_u,
        front_loss_ptr_u,
        front_seq_ptr_u,
        front_ticks_ptr_u,
        has_time_i32,
    )


@intrinsic
def pulse_write_shards_intrinsic(
    typingctx,
    filepath_ptr_ty,
    filepath_len_ty,
    n_ty,
    has_time_ty,
    base_wall_epoch_ty,
    base_tsc_ty,
    tps_ty,
    acc_ptr_ty,
    loss_ptr_ty,
    seq_ptr_ty,
    ticks_ptr_ty,
    nshards_ty,
):
    sig = signature(
        types.int32,
        types.uintp,
        types.int64,
        types.int64,
        types.int32,
        types.float64,
        types.uint64,
        types.float64,
        types.uintp,
        types.uintp,
        types.uintp,
        types.uintp,
        types.int32,
    )

    def codegen(context, builder, sig_obj, args):
        fn_ty = ir.FunctionType(
            ir.IntType(32),
            [
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(32),
                ir.DoubleType(),
                ir.IntType(64),
                ir.DoubleType(),
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(64),
                ir.IntType(32),
            ],
        )
        mod = builder.module
        fn = mod.globals.get("pulse_write_shards")
        if fn is None:
            fn = ir.Function(mod, fn_ty, name="pulse_write_shards")
            fn.linkage = "external"
        return builder.call(fn, list(args))

    return sig, codegen


@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def _pulse_write_shards(
    filepath_ptr_u,
    filepath_len_i64,
    n_i64,
    has_time_i32,
    base_wall_epoch_f64,
    base_tsc_u64,
    tps_f64,
    acc_ptr_u,
    loss_ptr_u,
    seq_ptr_u,
    ticks_ptr_u,
    nshards_i32,
):
    return pulse_write_shards_intrinsic(
        filepath_ptr_u,
        filepath_len_i64,
        n_i64,
        has_time_i32,
        base_wall_epoch_f64,
        base_tsc_u64,
        tps_f64,
        acc_ptr_u,
        loss_ptr_u,
        seq_ptr_u,
        ticks_ptr_u,
        nshards_i32,
    )


@njit(inline="always", fastmath=True, boundscheck=False)
def pulse(i, acc, loss, args):
    if (i & args[10][0]) != 0:
        return

    fcur = args[8][0]

    args[0][fcur] = acc
    args[1][fcur] = loss
    args[2][fcur] = i
    if args[10][1] != 0:
        args[3][fcur] = rdtsc()

    fcur += np.int64(1)
    args[8][0] = fcur

    if fcur != np.int64(FRONT_CAP):
        return

    base_u64 = np.uint64(args[9][0])

    _pulse_copy256(
        args[11][4],
        args[11][5],
        args[11][6],
        args[11][7],
        base_u64,
        args[11][0],
        args[11][1],
        args[11][2],
        args[11][3],
        np.int32(1 if args[10][1] != 0 else 0),
    )

    args[9][0] = np.int64(base_u64 + np.uint64(FRONT_CAP))
    args[8][0] = np.int64(0)


def train_model(iterations, x, alpha, beta, *args):
    curr_acc = np.float32(0.1)
    curr_loss = np.float32(1.0)

    rng = np.int64(0x9E3779B97F4A7C15)
    h = np.int64(0xCBF29CE484222325)

    n = np.int64(x.shape[0])
    mask = n - np.int64(1)

    c_half = np.float32(0.5)
    c_1e3 = np.float32(1e-3)
    c_1e6 = np.float32(1e6)
    c_1e6_u = np.float32(1e-6)

    mul = np.int64(6364136223846793005)
    inc = np.int64(1442695040888963407)
    modmask = np.int64(0xFFFFFFFFFFFFFFFF)
    fnv = np.int64(0x100000001B3)

    iters = np.int64(iterations)
    for i in range(iters):
        rng = (rng * mul + inc) & modmask

        j = (np.int64(i) + rng) & mask
        u = np.float32(x[j])

        grad = (u - c_half) + (curr_acc * c_1e3)
        curr_acc = curr_acc + alpha * grad
        curr_loss = (curr_loss * beta) + grad * grad + u * c_1e6_u

        q = np.int64(curr_acc * c_1e6) ^ (np.int64(curr_loss * c_1e6) << np.int64(1))
        h = (h ^ q) * fnv

        pulse(i, curr_acc, curr_loss, args)

    return h


def _read_pulse_file(filepath):
    with open(filepath, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        if len(header_bytes) != HEADER_SIZE:
            raise ValueError("File too small for header.")
        magic, version, n, base_wall_epoch, base_tsc, tps, has_time, _ = struct.unpack(HEADER_FMT, header_bytes)
        if int(magic) != MAGIC:
            raise ValueError("Invalid file magic.")
        if int(version) != VERSION:
            raise ValueError(f"Unexpected version: {version} != {VERSION}")
        return {
            "header": {
                "version": int(version),
                "n": int(n),
                "base_wall_epoch": float(base_wall_epoch),
                "base_tsc": int(base_tsc),
                "tps": float(tps),
                "has_time": int(has_time),
            }
        }


class PulseManager:
    @staticmethod
    def run(user_fn, iterations, *args, filepath="/tmp/pulse_log.bin"):
        iters = int(iterations)

        front_acc = np.empty(FRONT_CAP, dtype=np.float32)
        front_loss = np.empty(FRONT_CAP, dtype=np.float32)
        front_seq = np.empty(FRONT_CAP, dtype=np.int64)
        front_ticks = np.empty(FRONT_CAP, dtype=np.uint64)

        back_acc = np.empty(iters, dtype=np.float32)
        back_loss = np.empty(iters, dtype=np.float32)
        back_seq = np.empty(iters, dtype=np.int64)
        back_ticks = np.empty(iters, dtype=np.uint64)

        front_cursor = np.zeros(1, dtype=np.int64)
        back_cursor = np.zeros(1, dtype=np.int64)

        sparse_mask = np.int64(0)
        prof_flag = np.int64(1 if ENABLE_PROFILING else 0)
        params = np.array([sparse_mask, prof_flag], dtype=np.int64)

        ptrs = np.array(
            [
                np.uintp(front_acc.ctypes.data),
                np.uintp(front_loss.ctypes.data),
                np.uintp(front_seq.ctypes.data),
                np.uintp(front_ticks.ctypes.data),
                np.uintp(back_acc.ctypes.data),
                np.uintp(back_loss.ctypes.data),
                np.uintp(back_seq.ctypes.data),
                np.uintp(back_ticks.ctypes.data),
            ],
            dtype=np.uintp,
        )

        packed_args = (
            front_acc,
            front_loss,
            front_seq,
            front_ticks,
            back_acc,
            back_loss,
            back_seq,
            back_ticks,
            front_cursor,
            back_cursor,
            params,
            ptrs,
        )

        if isinstance(user_fn, nb.core.registry.CPUDispatcher):
            fast_user_fn = user_fn
        else:
            fast_user_fn = njit(user_fn, cache=True, fastmath=True, boundscheck=False)

        fast_user_fn(np.int64(1), *args[1:], *packed_args)
        fast_user_fn(np.int64(1), *args[1:], *packed_args)

        front_cursor[0] = np.int64(0)
        back_cursor[0] = np.int64(0)

        base_wall_epoch = time.time()
        c0 = rdtsc_func()
        t0 = perf_counter()

        h = fast_user_fn(args[0], *args[1:], *packed_args)

        t1 = perf_counter()
        c1 = rdtsc_func()

        elapsed = float(t1 - t0)
        total_cycles = float(np.uint64(c1 - c0))
        tps = float(total_cycles / elapsed) if elapsed > 0.0 else 0.0

        n_written = int(back_cursor[0])
        fcur = int(front_cursor[0])
        if fcur != 0:
            if n_written + fcur > iters:
                raise RuntimeError("backing arrays overflow (warmup/state bug)")
            j0 = n_written
            j1 = n_written + fcur
            back_acc[j0:j1] = front_acc[0:fcur]
            back_loss[j0:j1] = front_loss[0:fcur]
            back_seq[j0:j1] = front_seq[0:fcur]
            if ENABLE_PROFILING:
                back_ticks[j0:j1] = front_ticks[0:fcur]
            n_written = j1

        path_bytes = filepath.encode("utf-8")
        path_buf = np.frombuffer(path_bytes, dtype=np.uint8)

        rc = _pulse_write_shards(
            np.uintp(path_buf.ctypes.data),
            np.int64(path_buf.size),
            np.int64(n_written),
            np.int32(1 if ENABLE_PROFILING else 0),
            np.float64(base_wall_epoch),
            np.uint64(c0),
            np.float64(tps),
            np.uintp(back_acc.ctypes.data),
            np.uintp(back_loss.ctypes.data),
            np.uintp(back_seq.ctypes.data),
            np.uintp(back_ticks.ctypes.data),
            np.int32(NSHARDS),
        )
        if int(rc) != 0:
            raise RuntimeError("pulse_write_shards failed")

        return {
            "elapsed": elapsed,
            "ticks_per_second": float(tps),
            "h": int(h),
            "n_written": int(n_written),
            "meta": {"front_capacity": int(FRONT_CAP), "log_every": int(LOG_EVERY), "shards": int(NSHARDS)},
        }


def main():
    iters = np.int64(ITERS)
    alpha = np.float32(0.0001)
    beta = np.float32(0.9999)
    x = np.random.random(1 << 20).astype(np.float32)

    filepath = "/tmp/pulse_log.bin"
    out = PulseManager.run(train_model, iters, iters, x, alpha, beta, filepath=filepath)

    print(
        f"Profiling={'ON' if ENABLE_PROFILING else 'OFF'} "
        f"log_every={LOG_EVERY} front_capacity={out['meta']['front_capacity']} shards={out['meta']['shards']}"
    )
    print(f"Elapsed: {out['elapsed']:.8f}s")
    print(f"n_written: {out['n_written']}")

    shard0 = filepath + ".shard000"
    try:
        hdr = _read_pulse_file(shard0)
        print("shard0 header:", hdr["header"])
    except Exception as e:
        print("shard0 header read failed:", str(e))

    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    mem_mb = mem_usage / divisor
    print(f"Peak Memory: {mem_mb:.2f} MB")


if __name__ == "__main__":
    main()
