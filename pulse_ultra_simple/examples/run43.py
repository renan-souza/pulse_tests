import time
import numpy as np
import numba as nb
from numba import njit, types
from numba.extending import intrinsic
import pulse_rt_ext as rt

NUM_RINGS = 4096


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


@njit(inline="always")
def pulse(i, acc, loss, args):
    # args: [rings_base_ptr]
    ring_idx = i >> 8
    slot = i & 255

    # ring size = 256 * 24 (data) + 4 (state)
    ring_stride = 6144 + 4
    ring_ptr = args[0] + (ring_idx % NUM_RINGS) * ring_stride
    base = ring_ptr + (slot * 24)

    to_ptr(base, nb.float32)[0] = acc
    to_ptr(base + 4, nb.float32)[0] = loss
    to_ptr(base + 8, nb.int64)[0] = i
    to_ptr(base + 16, nb.uint64)[0] = rdtsc()

    if slot == 255:
        # Seal ring (state = 2). Consumers on other cores will bleed this to DRAM.
        to_ptr(ring_ptr + 6144, nb.int32)[0] = 2


@njit(fastmath=True)
def train_model(iterations, x, alpha, beta, *args):
    curr_acc, curr_loss = np.float32(0.1), np.float32(1.0)
    rng, h = np.int64(0x9E3779B97F4A7C15), np.int64(0xCBF29CE484222325)
    n = np.int64(x.shape[0])
    mask = n - 1

    for i in range(iterations):
        rng = (rng * 6364136223846793005 + 1442695040888963407)
        u = x[rng & mask]
        grad = (u - 0.5) + (curr_acc * 1e-3)
        curr_acc += alpha * grad
        curr_loss = (curr_loss * 0.9999) + grad * grad + u * 1e-6
        h = (h ^ (np.int64(curr_acc * 1e6) * 0x100000001B3))

        pulse(i, curr_acc, curr_loss, args)

    return h


class PulseManager:
    @staticmethod
    def run(user_fn, iters, x, alpha, beta):
        rings_ptr = rt.init()
        packed_args = (np.uint64(rings_ptr),)

        print("Warmup...")
        user_fn(np.int64(1024), x, alpha, beta, *packed_args)

        print("Timed Region Starting...")
        t0 = time.perf_counter()
        h = user_fn(iters, x, alpha, beta, *packed_args)
        elapsed = time.perf_counter() - t0
        print(f"Elapsed: {elapsed:.4f}s")

        rt.save("pulse_final.bin")
        return h


if __name__ == "__main__":
    x = np.random.random(1 << 20).astype(np.float32)
    PulseManager.run(train_model, np.int64(100_000_000), x, np.float32(0.0001), np.float32(0.9999))