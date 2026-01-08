import time
import numpy as np
import numba as nb
import resource
from numba import objmode
nb.config.BOUNDSCHECK = 0

from numba import njit, uint8, float64, int64, uint16, uint32
from pulse_fast_ext import Pulse


@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def log(views, i, mask, scale, bias, values):
    j = i & mask
    views[0][j] = uint8(values[0] * scale + bias)
    views[1][j] = uint8(values[1] * scale + bias)

@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def train_model(capacity, views, iterations, alpha, beta):
    curr_loss = float64(1.0)
    curr_acc = float64(0.1)
    iters = int64(iterations)
    mask = int64(capacity - 1)
    scale = float64(10.0)
    bias = float64(0.5)

    with objmode(t0='float64'):
        t0 = time.time()

    for i in range(iters):
        curr_loss *= alpha
        curr_acc += beta

        values = (curr_acc, curr_loss)
        log(views, i, mask, scale, bias, values)

    with objmode(t1='float64'):
        t1 = time.time()

    return t1-t0

class PulseManager:
    @staticmethod
    def run(user_fn, capacity, *args, filepath="pulse_log.bin"):
        # 1. Safety check for power-of-two (required for bitwise mask)
        if (capacity & (capacity - 1)) != 0 or capacity <= 0:
            raise ValueError("Capacity must be a power of two for optimized bitwise masking.")

        # 2. Initialize Backend
        backend = Pulse(capacity)
        views = backend.arrays()

        # 3. Encapsulated Warm-up (Compiles the function before the real run)
        # Using 1 iteration to trigger JIT and cache loading
        user_fn(capacity, views, int64(1), *args[1:])

        # 4. The Real Execution
        elapsed = user_fn(capacity, views, *args)

        # 5. Cleanup and Persist
        backend.flush(filepath, capacity)

        return elapsed


def main():
    CAPACITY = 256
    MAX_ITERS = int64(1_000_000_000)
    ALPHA = float64(0.999999)
    BETA = float64(0.0000001)

    elapsed = PulseManager.run(train_model, CAPACITY, MAX_ITERS, ALPHA, BETA)

    print("Loop completed in ", elapsed, " s.")
    if elapsed >= 1.35:
        raise Exception("LONG TIME!")

    # 4. Verification
    print("Reading back data into pandas DataFrame...")
    from read_into_pandas import read_pulse_file
    df = read_pulse_file("pulse_log.bin")

    print(f"Read back {len(df)} records.")
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Peak Memory: {mem_mb:.2f} MB")

    assert len(df) == CAPACITY
    print(df.head(1))
    print(df.tail(1))
    print("Verification Successful.")


if __name__ == "__main__":
    main()
