import numpy as np
from numba import njit
from time import time

@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def pulse(i, alpha, beta, mask, v0, v1):
    ix = i & mask
    v0[ix] = alpha
    v1[ix] = beta

@njit(cache=True, fastmath=True, boundscheck=False)
def train_model(iterations, alpha, beta, mask, v0, v1):
    curr_loss = 1.0
    curr_acc = 0.1

    for i in range(iterations):
        curr_loss *= alpha
        curr_acc += beta
        pulse(i, curr_acc, curr_loss, mask, v0, v1)

    return True

def main():
    capacity = 1 << 20
    mask = np.int64(capacity - 1)

    v0 = np.empty(capacity, dtype=np.float32)
    v1 = np.empty(capacity, dtype=np.float32)

    iters = 10_000_000
    alpha = 0.9999
    beta = 0.0001

    t0 = time()
    train_model(iters, alpha, beta, mask, v0, v1)
    t1 = time()
    print(t1-t0)

if __name__ == "__main__":
    main()
