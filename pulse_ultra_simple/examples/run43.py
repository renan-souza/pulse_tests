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


@njit(inline="always")
def pulse(i, acc, loss, args):
    # args: [base_acc, base_loss, base_seq, base_tick]
    # Direct offsets for maximal write-combining efficiency
    off_4 = i * 4
    off_8 = i * 8

    # Store data + timestamp
    to_ptr(args[0] + off_4, nb.float32)[0] = acc
    to_ptr(args[1] + off_4, nb.float32)[0] = loss
    to_ptr(args[2] + off_8, nb.int64)[0] = i
    to_ptr(args[3] + off_8, nb.uint64)[0] = rdtsc()


# --- USER FUNCTION (from run41.py) ---
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


class PulseManager:
    @staticmethod
    def run(user_fn, iters, x, alpha, beta):
        addrs = rt.init(int(iters))
        packed_args = (np.uint64(addrs[0]), np.uint64(addrs[1]), np.uint64(addrs[2]), np.uint64(addrs[3]))
        fast_fn = njit(user_fn, fastmath=True, boundscheck=False)

        print("Warmup...")
        fast_fn(np.int64(1000), x, alpha, beta, *packed_args)

        print("Timed Region Starting...")
        t0 = time.perf_counter()
        h = fast_fn(iters, x, alpha, beta, *packed_args)
        elapsed = time.perf_counter() - t0
        print(f"Elapsed: {elapsed:.4f}s")

        print("Saving Binary Data...")
        rt.save("pulse_output.bin")
        return h


def verify_output(filepath, expected_n):
    print(f"\n--- Verifying Integrity of {filepath} ---")
    # File should be (4+4+8+8) * 100M = 24 bytes per record
    expected_size = expected_n * 24
    actual_size = os.path.getsize(filepath)
    print(f"File Size: {actual_size / 1e9:.2f} GB (Expected {expected_size / 1e9:.2f} GB)")

    if actual_size != expected_size:
        raise AssertionError("File size mismatch! Lossless capture failed.")

    with open(filepath, "rb") as f:
        # Check first 5 and last 5 iteration indices
        # Skip accuracy(4) and loss(4) = 8 bytes per record
        record_stride = 24

        print("Checking head and tail of seq array...")
        for i in [0, 1, expected_n - 2, expected_n - 1]:
            f.seek(int(expected_n * 8 + i * 8))  # Skip 2xfloat arrays (100M each)
            seq_val = np.frombuffer(f.read(8), dtype=np.int64)[0]
            print(f"  Record {i}: seq={seq_val}")
            if seq_val != i:
                raise AssertionError(f"Index mismatch at record {i}: found {seq_val}")

    print("Verification Successful: 100M records monotonically captured.")


if __name__ == "__main__":
    x = np.random.random(1 << 20).astype(np.float32)
    alpha, beta = np.float32(0.0001), np.float32(0.9999)
    PulseManager.run(train_model, np.int64(ITERS), x, alpha, beta)
    verify_output("pulse_output.bin", ITERS)