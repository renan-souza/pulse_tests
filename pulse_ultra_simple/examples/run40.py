import numpy as np
import numba as nb
from numba import njit
from time import perf_counter
from numba.extending import intrinsic
from numba import types
from numba.core.typing.templates import signature
from llvmlite import ir

nb.config.BOUNDSCHECK = 0

ENABLE_PROFILING = True
LOG_EVERY = 1024


@intrinsic
def rdtsc(typingctx):
    sig = signature(types.uint64)

    def codegen(context, builder, sig_obj, args):
        f_type = ir.FunctionType(ir.IntType(64), [])
        f = builder.module.declare_intrinsic("llvm.readcyclecounter", [], f_type)
        return builder.call(f, [], name="tsc")

    return sig, codegen


@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def rdtsc_func():
    return rdtsc()


@njit(cache=True, fastmath=True, boundscheck=False, inline="always")
def _log2_shift_from_pow2_minus1(mask_pow2_minus1):
    v = np.int64(mask_pow2_minus1 + np.int64(1))
    s = np.int64(0)
    while v > 1:
        v >>= np.int64(1)
        s += np.int64(1)
    return s


@njit(inline="always", fastmath=True, boundscheck=False)
def pulse(i, acc, loss, args):
    if (i & args[1]) != 0:
        return

    ring_mask = args[0]
    shift = args[2]
    ix = (i >> shift) & ring_mask

    args[3][ix] = acc
    args[4][ix] = loss
    args[5][ix] = i

    if args[7] != 0:
        args[6][ix] = rdtsc()


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

    iters = np.int64(iterations)
    for i in range(iters):
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


def _estimate_iter_time(seq_arr, tick_arr, tps):
    valid = seq_arr >= 0
    if np.count_nonzero(valid) < 2 or tps <= 0.0:
        return -1.0, -1.0, -1.0, 0

    s = seq_arr[valid].astype(np.int64, copy=False)
    t = tick_arr[valid].astype(np.uint64, copy=False)

    order = np.argsort(s)
    s = s[order]
    t = t[order]

    ds = s[1:] - s[:-1]
    ok = ds > 0
    if np.count_nonzero(ok) == 0:
        return -1.0, -1.0, -1.0, 0

    dt = (t[1:][ok] - t[:-1][ok]).astype(np.float64, copy=False)
    di = ds[ok].astype(np.float64, copy=False)

    per_iter_sec = (dt / di) / np.float64(tps)

    return (
        float(np.mean(per_iter_sec)),
        float(np.median(per_iter_sec)),
        float(np.percentile(per_iter_sec, 95)),
        int(per_iter_sec.size),
    )


class PulseManager:
    @staticmethod
    def run(user_fn, capacity, *args, filepath="pulse_log.bin"):
        ring_mask = np.int64(capacity - 1)
        log_every = np.int64(LOG_EVERY)
        sparse_mask = np.int64(log_every - 1)
        shift = _log2_shift_from_pow2_minus1(sparse_mask)

        acc_arr = np.empty(capacity, dtype=np.float32)
        loss_arr = np.empty(capacity, dtype=np.float32)
        seq_arr = np.full(capacity, -1, dtype=np.int64)

        if ENABLE_PROFILING:
            tick_arr = np.empty(capacity, dtype=np.uint64)
            prof_flag = np.int64(1)
        else:
            tick_arr = np.empty(1, dtype=np.uint64)
            prof_flag = np.int64(0)

        packed = (ring_mask, sparse_mask, shift, acc_arr, loss_arr, seq_arr, tick_arr, prof_flag)

        fast_user_fn = njit(user_fn, cache=True, fastmath=True, boundscheck=False)

        fast_user_fn(np.int64(1), *args[1:], *packed)
        fast_user_fn(np.int64(1), *args[1:], *packed)

        c0 = rdtsc_func()
        t0_wall = perf_counter()
        h = fast_user_fn(args[0], *args[1:], *packed)
        t1_wall = perf_counter()
        c1 = rdtsc_func()

        elapsed = t1_wall - t0_wall
        total_cycles = float(np.uint64(c1 - c0))
        tps = total_cycles / elapsed if elapsed > 0.0 else 0.0

        stats = None
        if ENABLE_PROFILING:
            mean_it, med_it, p95_it, pairs = _estimate_iter_time(seq_arr, tick_arr, tps)
            stats = {
                "pairs": pairs,
                "mean_iter_ns": mean_it * 1e9 if mean_it >= 0.0 else -1.0,
                "median_iter_ns": med_it * 1e9 if med_it >= 0.0 else -1.0,
                "p95_iter_ns": p95_it * 1e9 if p95_it >= 0.0 else -1.0,
            }

        return {
            "elapsed": float(elapsed),
            "t0_wall": float(t0_wall),
            "c0_cycles": int(c0),
            "ticks_per_second": float(tps),
            "h": int(h),
            "views": (acc_arr, loss_arr, seq_arr, tick_arr),
            "stats": stats,
        }


def main():
    capacity = 256
    iters = np.int64(100_000_000)
    alpha = np.float32(0.0001)
    beta = np.float32(0.9999)
    x = np.random.random(1 << 20).astype(np.float32)

    out = PulseManager.run(train_model, capacity, iters, x, alpha, beta)
    print(f"Profiling={'ON' if ENABLE_PROFILING else 'OFF'} log_every={LOG_EVERY} capacity={capacity}")
    print(f"Elapsed: {out['elapsed']:.8f}s")
    print(f"t0_wall: {out['t0_wall']:.6f}")
    print(f"c0_cycles: {out['c0_cycles']}")
    print(f"ticks_per_second: {out['ticks_per_second']:.3f}")
    print(f"h: {out['h']}")

    if out["stats"] is not None:
        s = out["stats"]
        print(
            f"pairs={s['pairs']} mean_iter={s['mean_iter_ns']:.3f}ns "
            f"median_iter={s['median_iter_ns']:.3f}ns p95_iter={s['p95_iter_ns']:.3f}ns"
        )


if __name__ == "__main__":
    main()
