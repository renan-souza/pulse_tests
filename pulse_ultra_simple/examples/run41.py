import os
import struct
import sys
import time
from time import perf_counter

import numpy as np
import numba as nb
import resource
from numba import njit
from numba.extending import intrinsic
from numba import types
from numba.core.typing.templates import signature
from llvmlite import ir

nb.config.BOUNDSCHECK = 0

ENABLE_PROFILING = True
LOG_EVERY = 1024

MAGIC = 0x31534C50
VERSION = 2

# Header layout:
# Magic(4) + Ver(4) + N(8) + BaseWallEpoch(8) + BaseTSC(8) + TPS(8) + HasTime(1) + Pad(7)
# - BaseWallEpoch is time.time() seconds since epoch (so timestamps are meaningful).
# - BaseTSC is the tick counter at the start of the timed region.
# - TPS is ticks_per_second estimated from (c1-c0) / (t1-t0) for this run.
HEADER_FMT = "<IIQdQdB7s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


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
    # Computes shift = log2(mask_pow2_minus1 + 1), assuming (mask_pow2_minus1 + 1) is a power of two.
    # Example: LOG_EVERY=1024 -> sparse_mask=1023 -> (sparse_mask+1)=1024 -> shift=10.
    #
    # Why we need shift:
    # - We log only when (i & sparse_mask) == 0, so logging iterations are multiples of LOG_EVERY.
    # - sample_id = i >> shift is a cheap divide by LOG_EVERY.
    # - ring slot = sample_id & ring_mask keeps the log buffers small (capacity entries).
    v = np.int64(mask_pow2_minus1 + np.int64(1))
    shift = np.int64(0)
    while v > 1:
        v >>= np.int64(1)
        shift += np.int64(1)
    return shift


@njit(inline="always", fastmath=True, boundscheck=False)
def pulse(i, acc, loss, args):
    # args tuple layout. WARNING: Do not unpack args.
    # User arrays first, then internal arrays, then internal masks, then internal control params.
    #
    #   args[0] = user_array0 (float32[capacity])  # accuracy buffer (user data)
    #   args[1] = user_array1 (float32[capacity])  # loss buffer (user data)
    #   args[2] = seq_arr (int64[capacity])        # iteration numbers for each logged sample
    #   args[3] = tick_arr (uint64[capacity] or dummy)  # cycle ticks for each logged sample
    #   args[4] = ring_mask  (int64)               # capacity-1
    #   args[5] = sparse_mask (int64)              # log gating mask, log only when (i & sparse_mask) == 0
    #   args[6] = shift (int64)                    # log2(LOG_EVERY), used to map i -> sample_id (i >> shift)
    #   args[7] = prof_flag (int64)                # 1 enables rdtsc capture, 0 disables

    # Gate logging (log every LOG_EVERY iters): skip if not a logging iteration
    if (i & args[5]) != 0:
        return

    # Map i -> ring slot:
    # - sample_id = i >> shift (cheap divide by LOG_EVERY)
    # - ix = sample_id & ring_mask (wrap into [0, capacity-1])
    ix = (i >> args[6]) & args[4]

    # User arrays (user data)
    args[0][ix] = acc
    args[1][ix] = loss

    # Internal arrays
    args[2][ix] = i

    # Internal control: capture rdtsc ONLY when profiling is enabled
    # This is the only place rdtsc() runs inside the hot loop.
    if args[7] != 0:
        args[3][ix] = rdtsc()


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
    valid_mask = seq_arr >= 0
    if np.count_nonzero(valid_mask) < 2 or tps <= 0.0:
        return -1.0, -1.0, -1.0, 0, None

    seq_valid = seq_arr[valid_mask].astype(np.int64, copy=False)
    ticks_valid = tick_arr[valid_mask].astype(np.uint64, copy=False)

    sort_idx = np.argsort(seq_valid)
    seq_sorted = seq_valid[sort_idx]
    ticks_sorted = ticks_valid[sort_idx]

    seq_deltas = seq_sorted[1:] - seq_sorted[:-1]
    ok = seq_deltas > 0
    if np.count_nonzero(ok) == 0:
        return -1.0, -1.0, -1.0, 0, None

    tick_deltas = (ticks_sorted[1:][ok] - ticks_sorted[:-1][ok]).astype(np.float64, copy=False)
    iter_deltas = seq_deltas[ok].astype(np.float64, copy=False)

    per_iter_sec = (tick_deltas / iter_deltas) / np.float64(tps)

    return (
        float(np.mean(per_iter_sec)),
        float(np.median(per_iter_sec)),
        float(np.percentile(per_iter_sec, 95)),
        int(per_iter_sec.size),
        per_iter_sec,
    )


def _detect_outliers_mad(x, z=6.0):
    # Robust outlier detection using MAD (median absolute deviation).
    # Intended for identifying rare OS jitter / interrupts showing up in sampled tick intervals.
    if x is None or x.size == 0:
        return 0, None, None, None

    med = float(np.median(x))
    abs_dev = np.abs(x - med)
    mad = float(np.median(abs_dev))

    if mad == 0.0:
        # Extremely tight / quantized distribution; MAD-based z-score becomes unusable.
        # Fallback: flag values above p99 to avoid false explosions.
        p99 = float(np.percentile(x, 99.0))
        mask = x > p99
        return int(np.count_nonzero(mask)), mask, med, mad

    robust_z = abs_dev / (1.4826 * mad)
    mask = robust_z > z
    return int(np.count_nonzero(mask)), mask, med, mad


def _write_pulse_file(
    filepath,
    user_array0,
    user_array1,
    seq_arr,
    tick_arr,
    base_wall_epoch,
    base_tsc,
    tps,
    has_time,
):
    n = int(user_array0.shape[0])
    has_time_b = 1 if has_time else 0
    pad = b"\x00" * 7

    header = struct.pack(
        HEADER_FMT,
        int(MAGIC),
        int(VERSION),
        int(n),
        float(base_wall_epoch),
        int(np.uint64(base_tsc)),
        float(tps),
        int(has_time_b),
        pad,
    )

    d = os.path.dirname(filepath)
    if d:
        os.makedirs(d, exist_ok=True)

    # NOTE: This is intentionally simple and fast:
    # - header (one write)
    # - contiguous arrays (a few large writes)
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(user_array0.tobytes(order="C"))
        f.write(user_array1.tobytes(order="C"))
        f.write(seq_arr.tobytes(order="C"))
        if has_time_b:
            f.write(tick_arr.tobytes(order="C"))


def _read_pulse_file(filepath):
    with open(filepath, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        if len(header_bytes) != HEADER_SIZE:
            raise ValueError("File too small for header.")

        magic, version, n, base_wall_epoch, base_tsc, tps, has_time, _ = struct.unpack(HEADER_FMT, header_bytes)
        if int(magic) != MAGIC:
            raise ValueError("Invalid file magic.")
        if int(version) != VERSION:
            raise ValueError(f"Unexpected version: {version} != {VERSION}")

        n = int(n)

        # Read the rest, then slice via frombuffer (fast, avoids per-field file reads).
        payload = f.read()
        off = 0

        def take(dtype, count):
            nonlocal off
            nbytes = np.dtype(dtype).itemsize * count
            chunk = payload[off : off + nbytes]
            if len(chunk) != nbytes:
                raise ValueError("Truncated file payload.")
            off += nbytes
            return np.frombuffer(chunk, dtype=dtype, count=count)

        user0 = take(np.float32, n)
        user1 = take(np.float32, n)
        seq = take(np.int64, n)
        ticks = None
        if int(has_time) != 0:
            ticks = take(np.uint64, n)

        return {
            "header": {
                "version": int(version),
                "n": int(n),
                "base_wall_epoch": float(base_wall_epoch),
                "base_tsc": int(base_tsc),
                "tps": float(tps),
                "has_time": int(has_time),
            },
            "user_array0": user0,
            "user_array1": user1,
            "seq": seq,
            "ticks": ticks,
        }


def _verify_run_requirements(
    *,
    capacity,
    ring_mask,
    sparse_mask,
    shift,
    seq_arr,
    tick_arr,
    tps,
    base_wall_epoch,
    base_tsc,
    iters_total,
):
    print("\n--- Verification: structure / sampling / timing ---")

    # Ring indexing requires power-of-two capacity so ring_mask = capacity - 1 is valid.
    capacity_is_pow2 = (capacity & (capacity - 1)) == 0
    print(f"capacity power-of-two: {capacity_is_pow2}")
    if not capacity_is_pow2:
        raise AssertionError("capacity must be power-of-two (required by ring indexing).")

    print(f"ring_mask == capacity-1: {int(ring_mask) == int(capacity - 1)}")
    if int(ring_mask) != int(capacity - 1):
        raise AssertionError("ring_mask mismatch.")

    expected_sparse_mask = int(LOG_EVERY - 1)
    print(f"sparse_mask == LOG_EVERY-1: {int(sparse_mask) == expected_sparse_mask}")
    if int(sparse_mask) != expected_sparse_mask:
        raise AssertionError("sparse_mask mismatch. LOG_EVERY must be power-of-two for this mask to be correct.")

    expected_shift = int(np.log2(LOG_EVERY))
    print(f"shift == log2(LOG_EVERY): {int(shift) == expected_shift}")
    if int(shift) != expected_shift:
        raise AssertionError("shift mismatch. LOG_EVERY must be power-of-two.")

    valid_mask = seq_arr >= 0
    n_valid = int(np.count_nonzero(valid_mask))
    print(f"valid logged slots: {n_valid}/{capacity}")
    if n_valid < 2:
        print("Not enough logged samples to verify timing (need >= 2).")
        return

    seq_valid = seq_arr[valid_mask].astype(np.int64, copy=False)
    sort_idx = np.argsort(seq_valid)
    seq_sorted = seq_valid[sort_idx]

    seq_deltas = seq_sorted[1:] - seq_sorted[:-1]
    seq_strictly_increasing = bool(np.all(seq_deltas > 0))
    print(f"seq strictly increasing after sort: {seq_strictly_increasing}")
    if not seq_strictly_increasing:
        raise AssertionError("seq contains duplicates or went backwards (unexpected).")

    # Gate sanity: logged i values should be multiples of LOG_EVERY.
    seq_multiple_ok = (seq_sorted % np.int64(LOG_EVERY)) == 0
    frac_multiple_ok = float(np.mean(seq_multiple_ok))
    print(f"seq multiples of LOG_EVERY: {frac_multiple_ok * 100.0:.2f}%")
    if frac_multiple_ok < 0.99:
        raise AssertionError("Many logged seq values are not multiples of LOG_EVERY. Sampling gate may be wrong.")

    # Delta structure sanity: most deltas should equal LOG_EVERY (unless you only see a tail with overwrite gaps).
    frac_delta_expected = float(np.mean(seq_deltas == np.int64(LOG_EVERY)))
    print(f"seq delta == LOG_EVERY: {frac_delta_expected * 100.0:.2f}%")
    if frac_delta_expected < 0.80:
        print("WARNING: Many seq deltas are not LOG_EVERY. This can indicate overwrite gaps or sampling mismatch.")

    # Ring-window range check:
    expected_span = (capacity - 1) * LOG_EVERY
    span_ok = int(seq_sorted[-1] - seq_sorted[0]) == int(expected_span)

    end_target = int(iters_total) - int(LOG_EVERY)
    end_ok = abs(int(seq_sorted[-1]) - end_target) <= int(LOG_EVERY)

    print(f"ring span == (capacity-1)*LOG_EVERY: {span_ok}")
    print(f"last sample near end (within LOG_EVERY): {end_ok}")
    if not span_ok:
        raise AssertionError("Ring does not cover a contiguous capacity-sized window of samples.")
    if not end_ok:
        raise AssertionError("Last sample not near end; mapping/gating may be wrong.")

    if tick_arr is not None and tps > 0.0:
        ticks_valid = tick_arr[valid_mask].astype(np.uint64, copy=False)
        ticks_sorted = ticks_valid[sort_idx]

        frac_after = float(np.mean(ticks_sorted >= np.uint64(base_tsc)))
        print(f"ticks >= base_tsc: {frac_after * 100.0:.2f}%")
        if frac_after < 0.99:
            print("WARNING: many ticks are < base_tsc; timing alignment may be wrong.")

        tick_deltas_i64 = (ticks_sorted[1:] - ticks_sorted[:-1]).astype(np.int64, copy=False)
        min_tick_delta = int(tick_deltas_i64.min())
        ticks_non_decreasing = bool(np.all(tick_deltas_i64 >= 0))
        print(f"ticks non-decreasing after sort: {ticks_non_decreasing} (min dt_ticks={min_tick_delta})")
        if not ticks_non_decreasing:
            print("WARNING: tick counter went backwards for some samples. Timing estimates may be unreliable.")

        seq_deltas_f64 = seq_deltas.astype(np.float64, copy=False)
        tick_deltas_f64 = (ticks_sorted[1:] - ticks_sorted[:-1]).astype(np.float64, copy=False)
        per_iter_sec = (tick_deltas_f64 / seq_deltas_f64) / float(tps)
        per_iter_ns = per_iter_sec * 1e9

        mean_ns = float(np.mean(per_iter_ns))
        med_ns = float(np.median(per_iter_ns))
        p95_ns = float(np.percentile(per_iter_ns, 95))
        p99_ns = float(np.percentile(per_iter_ns, 99))
        min_ns = float(np.min(per_iter_ns))
        max_ns = float(np.max(per_iter_ns))

        print("\nTiming estimate from ticks (interval-averaged, not per-iteration exact):")
        print(f"mean={mean_ns:.3f} ns  median={med_ns:.3f} ns  p95={p95_ns:.3f} ns  p99={p99_ns:.3f} ns")
        print(f"min={min_ns:.3f} ns  max={max_ns:.3f} ns")
        print(
            "Notes:\n"
            "- This is an estimate because ticks are a counter, not a wall clock.\n"
            "- ticks_per_second is calibrated over the whole timed region, so it averages DVFS/turbo and jitter.\n"
            "- Each sample spans LOG_EVERY iterations, so each point is an average over that window.\n"
            "- OS interrupts can inflate some intervals. We detect those as outliers.\n"
        )

        n_out, out_mask, out_med, out_mad = _detect_outliers_mad(per_iter_ns, z=6.0)
        if out_mask is None:
            print("outliers: n/a")
        else:
            print(f"outliers (MAD>6.0): {n_out}/{per_iter_ns.size} (median={out_med:.3f} ns, MAD={out_mad:.6f})")
            if n_out > 0:
                worst = float(np.max(per_iter_ns[out_mask]))
                print(f"worst outlier per-iter estimate: {worst:.3f} ns")
    else:
        print("Profiling ticks not available (either profiling OFF or tps <= 0).")

    plausible_epoch = base_wall_epoch > 1_500_000_000.0
    print(f"\nbase_wall_epoch looks like epoch seconds: {plausible_epoch}")
    if not plausible_epoch:
        print("WARNING: base_wall_epoch is not epoch-based. Timestamps derived from it will not be real dates.")


class PulseManager:
    @staticmethod
    def run(user_fn, capacity, *args, filepath="pulse_log.bin"):
        ring_mask = np.int64(capacity - 1)
        log_every = np.int64(LOG_EVERY)
        sparse_mask = np.int64(log_every - 1)
        shift = _log2_shift_from_pow2_minus1(sparse_mask)

        user_array0 = np.empty(capacity, dtype=np.float32)
        user_array1 = np.empty(capacity, dtype=np.float32)
        seq_arr = np.full(capacity, -1, dtype=np.int64)

        if ENABLE_PROFILING:
            tick_arr = np.empty(capacity, dtype=np.uint64)
            prof_flag = np.int64(1)
        else:
            tick_arr = np.empty(1, dtype=np.uint64)
            prof_flag = np.int64(0)

        packed_args = (user_array0, user_array1, seq_arr, tick_arr, ring_mask, sparse_mask, shift, prof_flag)

        fast_user_fn = njit(user_fn, cache=True, fastmath=True, boundscheck=False)

        # Two warmups:
        # - first call forces compilation for the exact signature used in the timed run
        # - second reduces first-call artifacts (lazy init, page faults, cache fills, DVFS ramp)
        fast_user_fn(np.int64(1), *args[1:], *packed_args)
        fast_user_fn(np.int64(1), *args[1:], *packed_args)

        base_wall_epoch = time.time()

        c0 = rdtsc_func()
        t0 = perf_counter()
        h = fast_user_fn(args[0], *args[1:], *packed_args)
        t1 = perf_counter()
        c1 = rdtsc_func()

        elapsed = float(t1 - t0)
        total_cycles = float(np.uint64(c1 - c0))
        tps = float(total_cycles / elapsed) if elapsed > 0.0 else 0.0

        stats = None
        if ENABLE_PROFILING:
            mean_it, med_it, p95_it, pairs, _ = _estimate_iter_time(seq_arr, tick_arr, tps)
            stats = {
                "pairs": pairs,
                "mean_iter_ns": mean_it * 1e9 if mean_it >= 0.0 else -1.0,
                "median_iter_ns": med_it * 1e9 if med_it >= 0.0 else -1.0,
                "p95_iter_ns": p95_it * 1e9 if p95_it >= 0.0 else -1.0,
            }

        _write_pulse_file(
            filepath=filepath,
            user_array0=user_array0,
            user_array1=user_array1,
            seq_arr=seq_arr,
            tick_arr=tick_arr if ENABLE_PROFILING else np.empty(0, dtype=np.uint64),
            base_wall_epoch=base_wall_epoch,
            base_tsc=np.uint64(c0),
            tps=tps,
            has_time=bool(ENABLE_PROFILING),
        )

        return {
            "elapsed": elapsed,
            "base_wall_epoch": float(base_wall_epoch),
            "c0_cycles": int(np.uint64(c0)),
            "ticks_per_second": float(tps),
            "h": int(h),
            "views": (user_array0, user_array1, seq_arr, tick_arr),
            "stats": stats,
            "meta": {
                "capacity": int(capacity),
                "ring_mask": int(ring_mask),
                "sparse_mask": int(sparse_mask),
                "shift": int(shift),
            },
        }


def main():
    capacity = 256
    iters = np.int64(100_000_000)
    alpha = np.float32(0.0001)
    beta = np.float32(0.9999)
    x = np.random.random(1 << 20).astype(np.float32)

    out = PulseManager.run(train_model, capacity, iters, x, alpha, beta, filepath="pulse_log.bin")

    print(f"Profiling={'ON' if ENABLE_PROFILING else 'OFF'} log_every={LOG_EVERY} capacity={capacity}")
    print(f"Elapsed: {out['elapsed']:.8f}s")
    print(f"base_wall_epoch: {out['base_wall_epoch']:.6f} (time.time(), epoch seconds)")
    print(f"c0_cycles: {out['c0_cycles']}")
    print(f"ticks_per_second: {out['ticks_per_second']:.3f}")
    print(f"h: {out['h']}")

    if out["stats"] is not None:
        stats = out["stats"]
        print(
            f"pairs={stats['pairs']} mean_iter={stats['mean_iter_ns']:.3f}ns "
            f"median_iter={stats['median_iter_ns']:.3f}ns p95_iter={stats['p95_iter_ns']:.3f}ns"
        )

    print("\nVerifying Data Integrity (read back the file we just wrote)...")
    blob = _read_pulse_file("pulse_log.bin")
    header = blob["header"]

    assert header["n"] == capacity, f"Length mismatch: {header['n']} != {capacity}"
    assert header["has_time"] == (1 if ENABLE_PROFILING else 0), "has_time mismatch"
    assert abs(header["tps"] - out["ticks_per_second"]) / max(out["ticks_per_second"], 1e-12) < 0.05, "tps mismatch too large"

    user0 = blob["user_array0"]
    user1 = blob["user_array1"]
    seq = blob["seq"]
    ticks = blob["ticks"]

    print("head:")
    for k in range(2):
        tk = int(ticks[k]) if ticks is not None else None
        print(f"  k={k:3d} user0={float(user0[k]): .6f} user1={float(user1[k]): .6f} seq={int(seq[k])} ticks={tk}")

    print("tail:")
    for k in range(capacity - 2, capacity):
        tk = int(ticks[k]) if ticks is not None else None
        print(f"  k={k:3d} user0={float(user0[k]): .6f} user1={float(user1[k]): .6f} seq={int(seq[k])} ticks={tk}")

    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    mem_mb = mem_usage / divisor
    print(f"Peak Memory: {mem_mb:.2f} MB")

    user_array0, user_array1, seq_arr, tick_arr = out["views"]
    meta = out["meta"]
    _verify_run_requirements(
        capacity=capacity,
        ring_mask=np.int64(meta["ring_mask"]),
        sparse_mask=np.int64(meta["sparse_mask"]),
        shift=np.int64(meta["shift"]),
        seq_arr=seq_arr,
        tick_arr=tick_arr if ENABLE_PROFILING else None,
        tps=out["ticks_per_second"],
        base_wall_epoch=out["base_wall_epoch"],
        base_tsc=out["c0_cycles"],
        iters_total=int(iters),
    )

    print("\nVerification Successful.")


if __name__ == "__main__":
    main()
