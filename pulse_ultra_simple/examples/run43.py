import time
import numpy as np
import numba as nb
from numba import njit, types
from numba.extending import intrinsic
import pulse_rt_ext as rt

ITERS = 100_000_000
CAP = 256


@intrinsic
def call_copy256(typingctx, f_ptr, off, a_ptr, l_ptr, s_ptr):
    from llvmlite import ir
    # void pulse_copy256(uint64_t off, float* a, float* l, int64_t* s)
    f_ty = ir.FunctionType(ir.VoidType(), [ir.IntType(64), ir.IntType(64), ir.IntType(64), ir.IntType(64)])
    sig = types.void(types.uint64, types.uint64, types.uint64, types.uint64, types.uint64)

    def codegen(context, builder, sig_obj, args):
        fn_ptr = builder.inttoptr(args[0], f_ty.as_pointer())
        # args[1]=off, args[2]=a_ptr, args[3]=l_ptr, args[4]=s_ptr
        builder.call(fn_ptr, [args[1], args[2], args[3], args[4]])
        return context.get_dummy_value()

    return sig, codegen


@njit(inline="always")
def pulse(i, acc, loss, args):
    # args: [l_acc, l_loss, l_seq, cur_arr, copy_fn_ptr, total_off_arr]
    cur = args[3][0]
    args[0][cur] = acc
    args[1][cur] = loss
    args[2][cur] = i

    cur += 1
    args[3][0] = cur

    if cur == CAP:
        off = args[5][0]
        # Pass the 256-entry local buffers for handoff
        call_copy256(
            args[4],
            off,
            args[0].ctypes.data,
            args[1].ctypes.data,
            args[2].ctypes.data
        )
        args[5][0] = off + CAP
        args[3][0] = 0


def train_model(iterations, x, alpha, beta, *args):
    curr_acc, curr_loss = np.float32(0.1), np.float32(1.0)
    rng = np.int64(0x9E3779B97F4A7C15)
    n_mask = np.int64(x.shape[0] - 1)

    for i in range(iterations):
        rng = (rng * 6364136223846793005 + 1442695040888963407)
        u = x[rng & n_mask]
        grad = (u - np.float32(0.5)) + (curr_acc * np.float32(1e-3))
        curr_acc += alpha * grad
        curr_loss = (curr_loss * beta) + grad * grad + u * np.float32(1e-6)

        pulse(i, curr_acc, curr_loss, args)
    return curr_acc


class PulseManager:
    @staticmethod
    def run(user_fn, iters, x, alpha, beta):
        copy_fn_addr = rt.init(int(iters))

        l_acc = np.zeros(CAP, dtype=np.float32)
        l_loss = np.zeros(CAP, dtype=np.float32)
        l_seq = np.zeros(CAP, dtype=np.int64)
        cur = np.zeros(1, dtype=np.int64)
        off = np.zeros(1, dtype=np.int64)

        packed_args = (l_acc, l_loss, l_seq, cur, copy_fn_addr, off)
        fast_fn = njit(user_fn, fastmath=True, boundscheck=False)

        print("Warmup...")
        fast_fn(np.int64(1000), x, alpha, beta, *packed_args)
        cur[0], off[0] = 0, 0

        print("Timed Region...")
        t0 = time.perf_counter()
        h = fast_fn(iters, x, alpha, beta, *packed_args)
        elapsed = time.perf_counter() - t0
        print(f"Elapsed: {elapsed:.4f}s")

        print("Saving Full 100M Dataset...")
        rt.save("pulse_output.bin")
        return h


if __name__ == "__main__":
    x = np.random.random(1 << 20).astype(np.float32)
    PulseManager.run(train_model, np.int64(ITERS), x, np.float32(0.0001), np.float32(0.9999))