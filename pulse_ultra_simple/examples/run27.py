import time
import numpy as np
import numba as nb
import resource
from numba import njit, uint8, float32, int64, objmode
from pulse_fast_ext import Pulse

# Global configuration for performance
nb.config.BOUNDSCHECK = 0


# --------------------------------------------------------------------------------
# FUNCTION FACTORY (Equivalent to C #DEFINE)
# --------------------------------------------------------------------------------
def create_log_fn(interval_mask, scale, bias):
    """
    Creates a specialized logging function where constants are baked into
    the machine code as immediates.
    """
    # Cast to specific types to aid the compiler
    s = float32(scale)
    b = float32(bias)
    m = int64(interval_mask)

    @njit(cache=True, fastmath=True, boundscheck=False, inline="always")
    def log_impl(i, acc, loss, args):
        # Bitwise 'AND' check (e.g., i & 7) is faster than modulo i % 8
        if i & m != 0:
            return
        mask = args[0]
        views = args[1]
        j = i & mask
        # s and b are hard-coded in the assembly, no memory lookup needed
        views[0][j] = uint8(acc * s + b)
        views[1][j] = uint8(loss * s + b)

    return log_impl


# "Define" our constants here
# Use (2^n - 1) for the interval to support bitwise logic (e.g., 7 = every 8th)
log = create_log_fn(interval_mask=7, scale=10.0, bias=0.5)


# The user-facing code stays clean and standard
def train_model(iterations, alpha, beta, *args):
    curr_loss = 1.0
    curr_acc = 0.1

    for i in range(iterations):
        curr_loss *= alpha
        curr_acc += beta

        log(i, curr_acc, curr_loss, args)

    return True


class PulseManager:
    @staticmethod
    def run(user_fn, capacity, iterations, alpha, beta, filepath="pulse_log.bin"):
        # 1. JIT the user function on-the-fly
        mask = capacity - 1

        fast_user_fn = njit(user_fn, cache=True, fastmath=True, boundscheck=False)

        backend = Pulse(capacity)
        views = backend.arrays()

        # 2. Warm-up
        fast_user_fn(int64(1), alpha, beta, mask, views)

        # 3. Execution with centralized timing
        t0 = time.perf_counter()
        fast_user_fn(iterations, alpha, beta, mask, views)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        backend.flush(filepath, capacity)
        return elapsed

def main():
    # Configuration
    CAPACITY = 256
    MAX_ITERS = int64(1_000_000_000)
    ALPHA = 0.999999
    BETA = 0.0000001
    THRESHOLD = 1.28

    print(f"Starting simulation: {MAX_ITERS:,} iterations...")

    elapsed = PulseManager.run(train_model, CAPACITY, MAX_ITERS, ALPHA, BETA)

    print(f"Loop completed in: {elapsed:.4f} s.")

    if elapsed >= THRESHOLD:
        raise Exception(f"Performance Regression: {elapsed:.4f}s exceeds {THRESHOLD}s limit.")

    # Verification and Analytics
    from read_into_pandas import read_pulse_file
    df = read_pulse_file("pulse_log.bin")

    print(f"Read back {len(df)} records.")
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Peak Memory: {mem_mb:.2f} MB")

    assert len(df) == CAPACITY
    print(df.head(2))
    print(df.tail(2))
    print("Verification Successful.")


if __name__ == "__main__":
    main()
