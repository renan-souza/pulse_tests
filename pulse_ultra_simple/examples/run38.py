import numpy as np
from numba import njit
from numba import int64, float64
from time import perf_counter
from math import sin, sqrt

@njit(inline="always", fastmath=True, boundscheck=False)
def pulse(i, acc, loss, args):
    if (i & args[1]) != 0: # Sparse logging
        return
    ix = i & args[0]
    args[2][ix] = acc
    args[3][ix] = loss

def train_model(iterations, x, alpha, beta, *args):
    curr_acc = float64(0.1)
    curr_loss = float64(1.0)

    # loop-carried state
    rng = int64(0x9E3779B97F4A7C15)
    h = int64(0xCBF29CE484222325)

    n = int64(x.shape[0])
    mask = n - 1  # assumes n is power-of-two for speed; if not, use % n

    iters = int64(iterations)
    for i in range(iters):
        # cheap PRNG (loop-carried)
        rng = (rng * int64(6364136223846793005) + int64(1442695040888963407)) & int64(0xFFFFFFFFFFFFFFFF)

        # data-dependent runtime load (prevents closed-form collapse)
        j = (i + rng) & mask
        u = float64(x[j])  # x is runtime data

        # "training-like" update with nonlinearity + data dependence
        grad = (u - 0.5) + (curr_acc * 1e-3)
        curr_acc = curr_acc + alpha * grad
        curr_loss = (curr_loss * beta) + grad * grad + u * 1e-6

        # integer hash depends on evolving floats (forces loop execution)
        q = int64(curr_acc * 1e6) ^ (int64(curr_loss * 1e6) << int64(1))
        h = (h ^ q) * int64(0x100000001B3)  # FNV-ish

        pulse(i, curr_acc, curr_loss, args)

    return h



class PulseManager:
    @staticmethod
    def run(user_fn, capacity, *args, filepath="pulse_log.bin"):
        mask = int64(capacity - 1)
        sparse_mask = int64((2**10) - 1)

        arr0 = np.empty(capacity, dtype=np.float32)
        arr1 = np.empty(capacity, dtype=np.float32)

        fast_user_fn = njit(user_fn, cache=True, fastmath=True, boundscheck=False)

        # Warm up: MUST match the timed-call signature exactly
        fast_user_fn(1,       *args[1:], mask, sparse_mask, arr0, arr1)
        t0 = perf_counter()
        fast_user_fn(args[0], *args[1:], mask, sparse_mask, arr0, arr1)
        t1 = perf_counter()

        return t1 - t0


def main():
    capacity = 256
    iters = 100_000_000
    alpha = 0.0001
    beta = 0.9999
    x = np.random.random(1 << 20).astype(np.float64)  # power-of-two length
    elapsed = PulseManager.run(train_model, capacity, iters, x, alpha, beta)
    print(f"{elapsed:.8f}s.")


if __name__ == "__main__":
    main()
