import numpy as np
from numba import njit, int64, float64
from time import perf_counter


@njit(inline="always", fastmath=True, boundscheck=False)
def pulse(step, acc, loss, ring_mask, sparse_mask, acc_arr, loss_arr):
    if (step & sparse_mask) != 0:
        return
    ix = step & ring_mask
    acc_arr[ix] = acc
    loss_arr[ix] = loss


@njit(cache=True, fastmath=True, boundscheck=False)
def train_model(iterations, alpha, beta, ring_mask, sparse_mask, acc_arr, loss_arr):
    curr_acc = float64(0.1)
    curr_loss = float64(1.0)

    iters = int64(iterations)
    for i in range(iters):
        curr_acc += alpha
        curr_loss *= beta
        pulse(i, curr_acc, curr_loss, ring_mask, sparse_mask, acc_arr, loss_arr)

    return curr_acc


def main():
    capacity = 256
    iters = 1_000_000_000
    alpha = float64(0.0001)
    beta = float64(0.9999)

    ring_mask = int64(capacity - 1)

    # log every 2^k steps: sparse_mask = (2^k - 1)
    # log every step: sparse_mask = 0
    sparse_mask = int64((2**20) - 1)

    acc_arr = np.empty(capacity, dtype=np.float32)
    loss_arr = np.empty(capacity, dtype=np.float32)

    train_model(1, alpha, beta, ring_mask, sparse_mask, acc_arr, loss_arr)

    t0 = perf_counter()
    out = train_model(iters, alpha, beta, ring_mask, sparse_mask, acc_arr, loss_arr)
    t1 = perf_counter()
    print(f"elapsed={t1 - t0:.8f}s out={out}")


if __name__ == "__main__":
    main()
