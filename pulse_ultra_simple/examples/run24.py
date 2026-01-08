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


@njit(cache=True, fastmath=True, boundscheck=False)
def train_model(capacity, views, iterations, alpha, beta):
    """
    High-performance simulation loop. Uses float32 to catalyze
    SIMD vectorization on modern CPU architectures.
    """
    # 1. Localize constants to catalyze register-only access
    scale = float32(10.0)
    bias = float32(0.5)

    # 2. Initialize state with float32 for maximum throughput
    curr_loss = float32(1.0)
    curr_acc = float32(0.1)
    a = float32(alpha)
    b = float32(beta)

    mask = int64(capacity - 1)
    iters = int64(iterations)

    with objmode(t0='float64'):
        t0 = time.time()

    # 3. The Tight Hot Loop
    for i in range(iters):
        curr_loss *= a
        curr_acc += b

        # Lead the compiler to keep variables in registers via scalar passing
        log_tight(views, i, mask, scale, bias, curr_acc, curr_loss)

    with objmode(t1='float64'):
        t1 = time.time()

    return t1 - t0


class PulseManager:
    @staticmethod
    def run(user_fn, capacity, iterations, alpha, beta, filepath="pulse_log.bin"):
        """
        Orchestrates simulation lifecycle: allocation, warm-up, execution, and flush.
        """
        # Safety check: bitwise mask requires power-of-two capacity
        if (capacity & (capacity - 1)) != 0 or capacity <= 0:
            raise ValueError("Capacity must be a power of two for bitwise optimization.")

        backend = Pulse(capacity)
        views = backend.arrays()

        # Warm-up: Triggers JIT and fills instruction cache before timing
        user_fn(capacity, views, int64(1), alpha, beta)

        # Real execution
        elapsed = user_fn(capacity, views, iterations, alpha, beta)

        # Cleanup and Persistence
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