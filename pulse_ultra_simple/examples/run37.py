import numpy as np
from numba import njit
from time import perf_counter


@njit(inline="always", fastmath=True, boundscheck=False)
def pulse(i, acc, loss, args):
    if (i & args[1]) != 0: # Sparse logging
        return
    ix = i & args[0]
    args[2][ix] = acc
    args[3][ix] = loss


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
        mask = capacity - 1
        sparse_mask = (2**20) - 1

        arr0 = np.empty(capacity, dtype=np.float32)
        arr1 = np.empty(capacity, dtype=np.float32)

        fast_user_fn = njit(user_fn, cache=True, fastmath=True, boundscheck=False)

        # Warm up: MUST match the timed-call signature exactly
        fast_user_fn(1, alpha, beta, mask, sparse_mask, arr0, arr1)

        t0 = perf_counter()
        fast_user_fn(iterations, alpha, beta, mask, sparse_mask, arr0, arr1)
        t1 = perf_counter()

        return t1 - t0


def main():
    capacity = 256
    iters = 1_000_000_000
    alpha = 0.0001
    beta = 0.9999
    elapsed = PulseManager.run(train_model, capacity, iters, alpha, beta)
    print(f"{elapsed:.8f}s.")


if __name__ == "__main__":
    main()
