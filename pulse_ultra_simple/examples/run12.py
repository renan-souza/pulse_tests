import time
import numpy as np
import numba as nb

nb.config.BOUNDSCHECK = 0

from numba import njit, uint8, float32, int64
from pulse_fast_ext import Pulse


SCALE = float32(10.0)
BIAS = float32(0.5)


@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def _log_u8(acc_view, loss_view, idx, acc, loss):
    acc_view[idx] = uint8(acc * SCALE + BIAS)
    loss_view[idx] = uint8(loss * SCALE + BIAS)


@njit(cache=True, fastmath=True, boundscheck=False)
def train_model_jit(acc_view, loss_view, capacity, max_iterations, alpha, beta):
    current_loss = float32(1.0)
    current_acc = float32(0.1)

    iters = int64(max_iterations)
    cap = int64(capacity)

    for i in range(iters):
        current_loss *= alpha
        current_acc += beta

        idx = i % cap
        _log_u8(acc_view, loss_view, idx, current_acc, current_loss)


class PulseManager:
    def __init__(self, capacity, filepath="pulse_log.bin"):
        self.capacity = int(capacity)
        self.filepath = filepath
        self.backend = None

    def __enter__(self):
        self.backend = Pulse(self.capacity)
        return self.backend.arrays()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend.flush(self.filepath, self.capacity)


def run_simulation():
    CAPACITY = 256
    MAX_ITERS = 100_000_000
    ALPHA = np.float32(0.999999)
    BETA = np.float32(0.0000001)

    with PulseManager(CAPACITY) as (acc_view, loss_view):
        train_model_jit(acc_view[:1], loss_view[:1], np.int64(1), np.int64(1), ALPHA, BETA)

        t0 = time.time()
        train_model_jit(acc_view, loss_view, np.int64(CAPACITY), np.int64(MAX_ITERS), ALPHA, BETA)
        t1 = time.time()

    print(f"Loop completed in: {t1 - t0:.4f}s")


if __name__ == "__main__":
    run_simulation()
