import time
import numpy as np
import numba as nb

nb.config.BOUNDSCHECK = 0

from numba import njit, uint8, float64, int64
from pulse_fast_ext import Pulse


SCALE = float64(10.0)
BIAS = float64(0.5)


@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def log(acc_view, loss_view, i, cap, acc, loss):
    j = i % cap
    acc_view[j] = uint8(acc * SCALE + BIAS)
    loss_view[j] = uint8(loss * SCALE + BIAS)


@njit(cache=True, fastmath=True, boundscheck=False)
def train_model(acc_view, loss_view, capacity, iterations, alpha, beta):
    curr_loss = float64(1.0)
    curr_acc = float64(0.1)

    iters = int64(iterations)
    cap = int64(capacity)

    for i in range(iters):
        curr_loss *= alpha
        curr_acc += beta
        log(acc_view, loss_view, i, cap, curr_acc, curr_loss)


class PulseManager:
    def __init__(self, capacity, train_fn, alpha, beta, filepath="pulse_log.bin"):
        self.capacity = int(capacity)
        self.filepath = filepath

        self.train_fn = train_fn
        self.alpha = alpha
        self.beta = beta

        self.backend = None
        self.acc_view = None
        self.loss_view = None

    def __enter__(self):
        self.backend = Pulse(self.capacity)
        self.acc_view, self.loss_view = self.backend.arrays()

        # Warmup (exact signature, no extras)
        self.train_fn(
            self.acc_view[:1],
            self.loss_view[:1],
            int64(1),
            int64(1),
            self.alpha,
            self.beta,
        )

        return self

    def run(self, iterations):
        self.train_fn(
            self.acc_view,
            self.loss_view,
            int64(self.capacity),
            int64(iterations),
            self.alpha,
            self.beta,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend.flush(self.filepath, self.capacity)



def main():
    CAPACITY = 256
    MAX_ITERS = 50_000_000
    ALPHA = float64(0.999999)
    BETA = float64(0.0000001)

    with PulseManager(CAPACITY, train_model, ALPHA, BETA) as pm:
        t0 = time.time()
        pm.run(MAX_ITERS)
        t1 = time.time()

    print(f"Ring loop completed in: {t1 - t0:.4f}s")

if __name__ == "__main__":
    main()
