import numpy as np
import time
from numba import njit, uint8, float64
from pulse_fast_ext import Pulse




# 1. THE BARE-METAL LOOP
# Using explicit array arguments instead of a class method
# This is the ONLY way to get 1:1 parity with Pure C.
@njit(nopython=True, fastmath=True, boundscheck=False)
def train_loop_bare_metal(acc_view, loss_view, iterations, alpha, beta):
    # Localize for register-level speed
    curr_loss = 1.0
    curr_acc = 0.1

    # Scale and offset constants
    scale = 10.0
    offset = 0.5

    for i in range(iterations):
        curr_loss *= alpha
        curr_acc += beta

        # Direct memory writes allow the compiler to use SIMD (AVX/SSE)
        acc_view[i] = uint8(curr_acc * scale + offset)
        loss_view[i] = uint8(curr_loss * scale + offset)


# 2. THE MANAGER (ORCHESTRATOR)
class PulseManager:
    def __init__(self, capacity, filepath="pulse_log.bin"):
        self.capacity = capacity
        self.filepath = filepath
        self.backend = Pulse(capacity)

    def __enter__(self):
        # Extract the raw NumPy views from the C extension
        return self.backend.arrays()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend.flush(self.filepath, self.capacity)


def run_simulation():
    MAX_EPOCHS = 50_000_000
    ALPHA, BETA = 0.999999, 0.0000001

    with PulseManager(MAX_EPOCHS) as (acc_view, loss_view):

        train_loop_bare_metal(acc_view[:1], loss_view[:1], 1, ALPHA, BETA)

        t0 = time.time()
        # The Hot Path
        train_loop_bare_metal(acc_view, loss_view, MAX_EPOCHS, ALPHA, BETA)
        t1 = time.time()

    print(f"Loop completed in: {t1 - t0:.4f}s")


if __name__ == "__main__":
    run_simulation()