import numpy as np
from numba import njit
from time import time

@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def pulse(i, acc, loss, args):
    ix = i & args[0]
    args[1][ix] = acc
    args[2][ix] = loss

def train_model(iterations, alpha, beta, *args):
    curr_acc = 0.1
    curr_loss = 1.0

    for i in range(iterations):
        curr_acc += alpha
        curr_loss *= beta
        pulse(i, curr_acc, curr_loss, args)

    return True

class PulseManager:
    @staticmethod
    def run(user_fn, capacity, iterations, alpha, beta, filepath="pulse_log.bin"):

        mask = np.int64(capacity - 1)
        arr0 = np.empty(capacity, dtype=np.float32)
        arr1 = np.empty(capacity, dtype=np.float32)

        fast_user_fn = njit(user_fn, cache=True, fastmath=True, boundscheck=False)

        # Warm up call:
        fast_user_fn(1, alpha, beta, mask, arr0, arr1)

        t0 = time()
        fast_user_fn(iterations, alpha, beta, mask, arr0, arr1)
        t1 = time()
        elapsed = t1 - t0
        return elapsed


def main():
    capacity = 256
    iters = 1_000_000_000
    alpha = 0.0001
    beta = 0.9999
    elapsed = PulseManager.run(train_model, capacity, iters, alpha, beta)
    print(f"{elapsed:.8f}s.")

if __name__ == "__main__":
    main()
