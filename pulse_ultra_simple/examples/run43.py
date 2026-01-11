import time
import os
import numpy as np
import numba as nb
from numba import njit, types
from numba.extending import intrinsic
import pulse_rt_ext as rt

# Initial allocation budget (can be very large, e.g., 200M)
BUDGET = 100_000_000


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
    # args: (ptr_acc, ptr_loss, ptr_seq, ptr_tick)
    # We use direct pointer math to allow for non-fixed iteration counts.

    off_f = i * 4
    off_8 = i * 8

    # Each store hits L1 first; Write-Combining buffers then stream to DRAM.
    to_ptr(args[0] + off_f, nb.float32)[0] = acc
    to_ptr(args[1] + off_f, nb.float32)[0] = loss
    to_ptr(args[2] + off_8, nb.int64)[0] = i
    to_ptr(args[3] + off_8, nb.uint64)[0] = rdtsc()


# --- USER FUNCTION (Flexible termination, Unmodified from run41.py) ---
@njit(fastmath=True)
def train_model(iterations, x, alpha, beta, *args):
    curr_acc, curr_loss = np.float32(0.1), np.float32(1.0)
    rng, h = np.int64(0x9E3779B97F4A7C15), np.int64(0xCBF29CE484222325)
    n = np.int64(x.shape[0])
    mask, c_half, c_1e3, c_1e6, c_1e6_u = n - 1, np.float32(0.5), np.float32(1e-3), np.float32(1e6), np.float32(1e-6)
    mul, inc, modmask, fnv = np.int64(6364136223846793005), np.int64(1442695040888963407), np.int64(
        0xFFFFFFFFFFFFFFFF), np.int64(0x100000001B3)

    # loop can now use runtime termination if needed
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

        # Example of runtime termination logic support:
        # if h == 0: break

    return h


class PulseManager:
    @staticmethod
    def run(user_fn, iters, x, alpha, beta):
        # We pre-allocate based on a budget, but the loop is open-ended.
        addrs = rt.init(int(BUDGET))

        # Pass raw addresses as uint64
        packed_args = (np.uint64(addrs[0]), np.uint64(addrs[1]),
                       np.uint64(addrs[2]), np.uint64(addrs[3]))

        # No dispatcher needed anymore, we call the user_fn directly as it is JITed
        print("Warmup...")
        user_fn(np.int64(1000), x, alpha, beta, *packed_args)

        print("Timed Region Starting...")
        t0 = time.perf_counter()
        h = user_fn(iters, x, alpha, beta, *packed_args)
        elapsed = time.perf_counter() - t0
        print(f"Elapsed: {elapsed:.4f}s")

        rt.save("pulse_output.bin")
        return h


if __name__ == "__main__":
    x = np.random.random(1 << 20).astype(np.float32)
    alpha, beta = np.float32(0.0001), np.float32(0.9999)
    # We call with ITERS, but the logic now aids any count up to BUDGET
    PulseManager.run(train_model, np.int64(BUDGET), x, alpha, beta)