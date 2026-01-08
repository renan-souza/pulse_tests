import numpy as np
from numba import njit
from time import perf_counter


@njit(inline="always", fastmath=True, boundscheck=False)
def pulse(i, acc, loss, args):
    if (i & args[1]) != 0:
        return
    ix = i & args[0]
    args[2][ix] = acc
    args[3][ix] = loss


def train_model(iterations, x, alpha, beta, *args):
    curr_acc = np.float32(0.1)
    curr_loss = np.float32(1.0)

    rng = np.int64(0x9E3779B97F4A7C15)
    h = np.int64(0xCBF29CE484222325)

    n = np.int64(x.shape[0])
    mask = n - np.int64(1)

    c_half = np.float32(0.5)
    c_1e3 = np.float32(1e-3)
    c_1e6 = np.float32(1e6)
    c_1e6_u = np.float32(1e-6)

    mul = np.int64(6364136223846793005)
    inc = np.int64(1442695040888963407)
    modmask = np.int64(0xFFFFFFFFFFFFFFFF)
    fnv = np.int64(0x100000001B3)

    for i in range(iterations):
        rng = (rng * mul + inc) & modmask

        j = (np.int64(i) + rng) & mask
        u = np.float32(x[j])

        grad = (u - c_half) + (curr_acc * c_1e3)
        curr_acc = curr_acc + alpha * grad
        curr_loss = (curr_loss * beta) + grad * grad + u * c_1e6_u

        q = np.int64(curr_acc * c_1e6) ^ (np.int64(curr_loss * c_1e6) << np.int64(1))
        h = (h ^ q) * fnv

        pulse(i, curr_acc, curr_loss, args)

    return h


class PulseManager:
    @staticmethod
    def run(user_fn, capacity, *args, filepath="pulse_log.bin"):
        mask = np.int64(capacity - 1)
        sparse_mask = np.int64((2**10) - 1)

        arr0 = np.empty(capacity, dtype=np.float32)
        arr1 = np.empty(capacity, dtype=np.float32)

        fast_user_fn = njit(user_fn, cache=True, fastmath=True, boundscheck=False)

        fast_user_fn(1, *args[1:], mask, sparse_mask, arr0, arr1)

        t0 = perf_counter()
        fast_user_fn(args[0], *args[1:], mask, sparse_mask, arr0, arr1)
        t1 = perf_counter()

        return t1 - t0


def main():
    capacity = 256
    iters = np.int64(100_000_000)
    alpha = np.float32(0.0001)
    beta = np.float32(0.9999)
    x = np.random.random(1 << 20).astype(np.float32)

    elapsed = PulseManager.run(train_model, capacity, iters, x, alpha, beta)
    print(f"{elapsed:.8f}s.")


if __name__ == "__main__":
    main()
