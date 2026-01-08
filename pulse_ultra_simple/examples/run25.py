import time
import numpy as np
import numba as nb
import resource
from numba import njit, uint8, float32, int64, objmode
from pulse_fast_ext import Pulse

# Global configuration for performance
nb.config.BOUNDSCHECK = 0


@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def log_tight(views, i, mask, scale, bias, acc, loss):
    """
    Inlined logger using bitwise masking and register-localized constants.
    Passing scalars prevents tuple-packing overhead in the hot loop.
    """
    j = i & mask

    # Numba unrolls this direct indexing at compile-time for UniTuples
    views[0][j] = uint8(acc * scale + bias)
    views[1][j] = uint8(loss * scale + bias)


# The user-facing code stays clean and standard
def train_model(capacity, views, iterations, alpha, beta):
    # No decorators, no Numba imports here
    scale = 10.0
    bias = 0.5

    curr_loss = 1.0
    curr_acc = 0.1

    mask = capacity - 1

    # This still needs to be efficient, so we'll JIT it internally
    for i in range(iterations):
        curr_loss *= alpha
        curr_acc += beta

        # Internally, PulseManager will ensure this is fast
        log_tight(views, i, mask, scale, bias, curr_acc, curr_loss)

    return True  # Timing handled by manager now


class PulseManager:
    @staticmethod
    def run(user_fn, capacity, iterations, alpha, beta, filepath="pulse_log.bin"):
        # 1. JIT the user function on-the-fly
        # We use 'nopython=True' to lead the compiler to the best performance
        fast_user_fn = njit(user_fn, cache=True, fastmath=True, boundscheck=False)

        backend = Pulse(capacity)
        views = backend.arrays()

        # 2. Warm-up
        fast_user_fn(capacity, views, int64(1), alpha, beta)

        # 3. Execution with centralized timing
        t0 = time.perf_counter()
        fast_user_fn(capacity, views, iterations, alpha, beta)
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
    THRESHOLD = 1.35

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
