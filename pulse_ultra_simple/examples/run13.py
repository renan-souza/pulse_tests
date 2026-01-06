import time
import numpy as np
import numba as nb

nb.config.BOUNDSCHECK = 0

from numba import njit, uint8, float64, int64
from pulse_fast_ext import Pulse


SCALE = float64(10.0)
BIAS = float64(0.5)


@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def _log_u8_linear(acc_view, loss_view, i, acc, loss):
    acc_view[i] = uint8(acc * SCALE + BIAS)
    loss_view[i] = uint8(loss * SCALE + BIAS)


@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def _log_u8_ring(acc_view, loss_view, i, cap, acc, loss):
    j = i % cap
    acc_view[j] = uint8(acc * SCALE + BIAS)
    loss_view[j] = uint8(loss * SCALE + BIAS)


@njit(cache=True, fastmath=True, boundscheck=False)
def train_model_linear(acc_view, loss_view, iterations, alpha, beta):
    curr_loss = float64(1.0)
    curr_acc = float64(0.1)

    iters = int64(iterations)

    for i in range(iters):
        curr_loss *= alpha
        curr_acc += beta
        _log_u8_linear(acc_view, loss_view, i, curr_acc, curr_loss)


@njit(cache=True, fastmath=True, boundscheck=False)
def train_model_ring(acc_view, loss_view, capacity, iterations, alpha, beta):
    curr_loss = float64(1.0)
    curr_acc = float64(0.1)

    iters = int64(iterations)
    cap = int64(capacity)

    for i in range(iters):
        curr_loss *= alpha
        curr_acc += beta
        _log_u8_ring(acc_view, loss_view, i, cap, curr_acc, curr_loss)


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


def run_linear_benchmark():
    MAX_EPOCHS = 50_000_000
    ALPHA = float64(0.999999)
    BETA = float64(0.0000001)

    with PulseManager(MAX_EPOCHS) as (acc_view, loss_view):
        train_model_linear(acc_view[:1], loss_view[:1], int64(1), ALPHA, BETA)

        t0 = time.time()
        train_model_linear(acc_view, loss_view, int64(MAX_EPOCHS), ALPHA, BETA)
        t1 = time.time()

    print(f"Linear loop completed in: {t1 - t0:.4f}s")


def run_ring_benchmark():
    CAPACITY = 256
    MAX_ITERS = 50_000_000
    ALPHA = float64(0.999999)
    BETA = float64(0.0000001)

    with PulseManager(CAPACITY) as (acc_view, loss_view):
        train_model_ring(acc_view[:1], loss_view[:1], int64(1), int64(1), ALPHA, BETA)

        t0 = time.time()
        train_model_ring(acc_view, loss_view, int64(CAPACITY), int64(MAX_ITERS), ALPHA, BETA)
        t1 = time.time()

    print(f"Ring loop completed in: {t1 - t0:.4f}s")


if __name__ == "__main__":
    #run_linear_benchmark()
    run_ring_benchmark()
