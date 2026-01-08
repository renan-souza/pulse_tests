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
    for k in range(len(views)):
        views[k][j] = uint8(values[k] * scale + bias)

@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def train_model(capacity, views, iterations, alpha, beta):
    curr_loss = float64(1.0)
    curr_acc = float64(0.1)
    iters = int64(iterations)
    cap = int64(capacity)
    mask = int64(capacity - 1)

    SCALE = float64(10.0)
    BIAS = float64(0.5)

    with objmode(t0='float64'):
        t0 = time.time()

    for i in range(iters):
        curr_loss *= alpha
        curr_acc += beta

        values = (curr_acc, curr_loss)
        log(views, i, mask, SCALE, BIAS, values)

    with objmode(t1='float64'):
        t1 = time.time()

    return t1-t0

class PulseManager:
    def __init__(self, capacity, filepath="pulse_log.bin"):
        self.capacity = int(capacity)
        self.filepath = filepath

        self.backend = None
        self.views = None

    def __enter__(self):
        self.backend = Pulse(self.capacity)

        views = self.backend.arrays()
        if not isinstance(views, tuple):
            raise TypeError("Pulse.arrays() must return a tuple of NumPy arrays")
        if len(views) == 0:
            raise ValueError("Pulse.arrays() returned no arrays")

        self.views = views

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend.flush(self.filepath, self.capacity)


def main():
    CAPACITY = 256
    MAX_ITERS = int64(1_000_000_000)
    ALPHA = float64(0.999999)
    BETA = float64(0.0000001)

    with PulseManager(CAPACITY) as pm:
        train_model(CAPACITY, pm.views, int64(1), ALPHA, BETA)
        elapsed = train_model(CAPACITY, pm.views, MAX_ITERS, ALPHA, BETA)

    print("Loop completed in ", elapsed, " s.")
    if elapsed >= 1.57:
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
