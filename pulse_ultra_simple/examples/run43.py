import time
import os
import numpy as np
import numba as nb
from numba import njit, types
from numba.extending import intrinsic
import pulse_rt_ext as rt

ITERS = 100_000_000

@intrinsic
def rdtsc(typingctx):
    from llvmlite import ir
    sig = nb.core.typing.templates.signature(types.uint64)
    def codegen(context, builder, sig_obj, args):
        f = builder.module.declare_intrinsic("llvm.readcyclecounter", [], ir.FunctionType(ir.IntType(64), []))
        return builder.call(f, [], name="tsc")
    return sig, codegen

@intrinsic
def to_ptr(typingctx, addr_ty, dtype_ty):
    from llvmlite import ir
    ptr_ty = types.CPointer(dtype_ty.dtype)
    sig = ptr_ty(types.uint64, dtype_ty)
    def codegen(context, builder, sig_obj, args):
        return builder.inttoptr(args[0], context.get_value_type(ptr_ty))
    return sig, codegen

@njit(inline="always", fastmath=True)
def pulse(i, acc, loss, args):
    # API: pulse(i, acc, loss, args)
    # args: (acc_v, loss_v, seq_v, tick_v)
    # This matches the run41.py indexing pattern exactly.
    args[0][i] = acc
    args[1][i] = loss
    args[2][i] = i
    args[3][i] = rdtsc()

# --- USER FUNCTION (Unmodified from run41.py, added njit) ---
@njit(fastmath=True)
def train_model(iterations, x, alpha, beta, *args):
    curr_acc, curr_loss = np.float32(0.1), np.float32(1.0)
    rng, h = np.int64(0x9E3779B97F4A7C15), np.int64(0xCBF29CE484222325)
    n = np.int64(x.shape[0])
    mask, c_half, c_1e3, c_1e6, c_1e6_u = n - 1, np.float32(0.5), np.float32(1e-3), np.float32(1e6), np.float32(1e-6)
    mul, inc, modmask, fnv = np.int64(6364136223846793005), np.int64(1442695040888963407), np.int64(
        0xFFFFFFFFFFFFFFFF), np.int64(0x100000001B3)

    for i in range(iterations):
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

@njit(fastmath=True)
def _dispatch_run(iters, x, alpha, beta, ptr_acc, ptr_loss, ptr_seq, ptr_tick):
    # Construct views inside JIT to avoid NotImplementedError
    acc_v = nb.carray(to_ptr(ptr_acc, nb.float32), (iters,))
    loss_v = nb.carray(to_ptr(ptr_loss, nb.float32), (iters,))
    seq_v = nb.carray(to_ptr(ptr_seq, nb.int64), (iters,))
    tick_v = nb.carray(to_ptr(ptr_tick, nb.uint64), (iters,))

    pulse_args = (acc_v, loss_v, seq_v, tick_v)
    return train_model(iters, x, alpha, beta, *pulse_args)

class PulseManager:
    @staticmethod
    def run(user_fn, iters, x, alpha, beta):
        addrs = rt.init(int(iters))

        # Warmup
        _dispatch_run(np.int64(1000), x, alpha, beta,
                      np.uint64(addrs[0]), np.uint64(addrs[1]),
                      np.uint64(addrs[2]), np.uint64(addrs[3]))

        print("Timed Region Starting...")
        t0 = time.perf_counter()
        h = _dispatch_run(iters, x, alpha, beta,
                          np.uint64(addrs[0]), np.uint64(addrs[1]),
                          np.uint64(addrs[2]), np.uint64(addrs[3]))
        elapsed = time.perf_counter() - t0
        print(f"Elapsed: {elapsed:.4f}s")

        print("Saving Binary Data...")
        rt.save("pulse_output.bin")
        return h

def verify_output(filepath, expected_n):
    print(f"\n--- Verifying Integrity ---")
    expected_size = expected_n * 24
    if os.path.getsize(filepath) != expected_size:
        raise AssertionError(f"Size mismatch! Found {os.path.getsize(filepath)}, expected {expected_size}")

    with open(filepath, "rb") as f:
        # Check sequence array (offset: acc(4)+loss(4) = 8 bytes per record)
        # Skip 100M float32 (acc) and 100M float32 (loss)
        f.seek(int(expected_n * 8 + (expected_n - 1) * 8))
        last_val = np.frombuffer(f.read(8), dtype=np.int64)[0]
        print(f"Last record seq check: {last_val}")
        if last_val != expected_n - 1:
            raise AssertionError("Sequence corruption!")
    print("Verification Successful.")

if __name__ == "__main__":
    x = np.random.random(1 << 20).astype(np.float32)
    alpha, beta = np.float32(0.0001), np.float32(0.9999)
    PulseManager.run(train_model, np.int64(ITERS), x, alpha, beta)
    verify_output("pulse_output.bin", ITERS)