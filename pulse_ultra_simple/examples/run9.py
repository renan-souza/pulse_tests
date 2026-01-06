import numpy as np
import time
from numba import njit, uint8, carray
from pulse_fast_ext import Pulse


# 1. PURE DOMAIN LOGIC (v7 Usability)
def train_model(logger, iterations, alpha, beta):
    curr_loss, curr_acc = 1.0, 0.1
    for _ in range(iterations):
        curr_loss *= alpha
        curr_acc += beta
        # The user sees the clean API
        logger.log(curr_acc, curr_loss)


# 2. THE HIGH-PERFORMANCE MANAGER (v8.1 Performance)
class PulseManager:
    def __init__(self, capacity, pulsed_function, *args, filepath="pulse_log.bin"):
        self.capacity = capacity
        self.filepath = filepath
        self.user_func = pulsed_function
        self.args = args
        self.backend = Pulse(capacity)
        self.fast_train = None

    def __enter__(self):
        # EXTRACT RAW POINTERS: Bypass NumPy metadata entirely
        # We need the memory addresses as integers
        acc_ptr = self.backend.get_acc_ptr()
        loss_ptr = self.backend.get_loss_ptr()

        # PERFORMANCE: Capture pointers in a closure
        # carray creates a writable view directly from a memory address
        @njit(fastmath=True, boundscheck=False, error_model='numpy')
        def specialized_train(n, alpha, beta):
            # Map raw pointers to writable Numba arrays
            acc_view = carray(acc_ptr, (n,), dtype=uint8)
            loss_view = carray(loss_ptr, (n,), dtype=uint8)

            c_loss, c_acc = 1.0, 0.1
            for i in range(n):
                c_loss *= alpha
                c_acc += beta
                # Bare-metal writes: SIMD-vectorizable by LLVM
                acc_view[i] = uint8(c_acc * 10.0 + 0.5)
                loss_view[i] = uint8(c_loss * 10.0 + 0.5)

        self.fast_train = specialized_train
        return self

    def pulse(self):
        self.fast_train(self.capacity, *self.args)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend.flush(self.filepath, self.capacity)


# --- EXECUTION ---
def run_simulation():
    MAX_EPOCHS = 50_000_000
    ALPHA, BETA = 0.999999, 0.0000001

    with PulseManager(MAX_EPOCHS, train_model, ALPHA, BETA) as pm:
        t0 = time.time()
        pm.pulse()
        t1 = time.time()

    print(f"Loop completed in: {t1 - t0:.4f}s")


if __name__ == "__main__":
    run_simulation()