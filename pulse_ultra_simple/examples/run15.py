import time
import numpy as np
import numba as nb
import resource

nb.config.BOUNDSCHECK = 0

from numba import njit, uint8, float64, int64
from pulse_fast_ext import Pulse


SCALE = float64(10.0)
BIAS = float64(0.5)


@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def log(views, i, cap, values):
    j = i % cap
    for k in range(len(views)):
        views[k][j] = uint8(values[k] * SCALE + BIAS)


@njit(cache=True, fastmath=True, boundscheck=False)
def train_model(acc_view, loss_view, capacity, iterations, alpha, beta):
    curr_loss = float64(1.0)
    curr_acc = float64(0.1)

    iters = int64(iterations)
    cap = int64(capacity)

    views = (acc_view, loss_view)

    for i in range(iters):
        curr_loss *= alpha
        curr_acc += beta

        values = (curr_acc, curr_loss)
        log(views, i, cap, values)


class PulseManager:
    def __init__(self, capacity, max_iters, user_fn, *user_fn_args, filepath="pulse_log.bin"):
        self.capacity = int(capacity)
        self.max_iters = int64(max_iters)
        self.filepath = filepath

        self.user_fn = user_fn
        self.user_fn_args = user_fn_args

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

        warm_views = tuple(v[:1] for v in self.views)
        self.user_fn(*warm_views, int64(1), int64(1), *self.user_fn_args)

        return self

    def run(self):
        self.user_fn(*self.views, int64(self.capacity), int64(self.max_iters), *self.user_fn_args)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend.flush(self.filepath, self.capacity)


def main():
    CAPACITY = 256
    MAX_ITERS = 1_000_000_000

    ALPHA = float64(0.999999)
    BETA = float64(0.0000001)

    with PulseManager(CAPACITY, MAX_ITERS, train_model, ALPHA, BETA) as pm:
        t0 = time.time()
        pm.run()
        t1 = time.time()

    print(f"Loop completed in: {t1 - t0:.4f}s")

    # 4. Verification
    print("Reading back data into pandas DataFrame...")
    from read_into_pandas import read_pulse_file
    df = read_pulse_file("pulse_log.bin")

    print(f"Read back {len(df)} records.")
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Peak Memory: {mem_mb:.2f} MB")

    assert len(df) == CAPACITY
    print(df.head())
    print(df.tail())
    print("Verification Successful.")


if __name__ == "__main__":
    main()
