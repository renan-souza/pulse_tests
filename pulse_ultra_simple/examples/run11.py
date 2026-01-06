import time
import resource
import struct

import numpy as np
import numba as nb

nb.config.BOUNDSCHECK = 0

from numba import njit, float32, int64, uint8
from pulse_fast_ext import Pulse


SCALE = float32(10.0)
BIAS = float32(0.5)


def build_compiled_runner(user_fn):
    @njit(cache=True, fastmath=True)
    def _runner(acc_view, loss_view, i0, max_iterations, alpha, beta, capacity):
        current_loss = float32(1.0)
        current_acc = float32(0.1)

        i = int64(i0)
        iters = int64(max_iterations)
        cap = int64(capacity)

        for _ in range(iters):
            current_loss *= alpha
            current_acc += beta

            idx = i % cap

            acc_view[idx] = uint8(current_acc * SCALE + BIAS)
            loss_view[idx] = uint8(current_loss * SCALE + BIAS)

            i += 1

        return i

    return _runner


def read_pulse_file(filepath):
    import pandas as pd

    with open(filepath, "rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError("Missing/short header")

        magic_u32, version_u32, nrecords_u64 = struct.unpack("<IIQ", header)
        raw = f.read()
        data = np.frombuffer(raw, dtype=np.uint8)

    n = int(nrecords_u64)
    expected_bytes = 2 * n
    if data.size < expected_bytes:
        raise ValueError(f"File truncated: expected {expected_bytes} bytes, got {data.size}")

    data = data[:expected_bytes]
    acc_u8 = data[:n]
    loss_u8 = data[n : 2 * n]

    acc = (acc_u8.astype(np.float32) - float(BIAS)) / float(SCALE)
    loss = (loss_u8.astype(np.float32) - float(BIAS)) / float(SCALE)

    return pd.DataFrame({"acc": acc, "loss": loss}), magic_u32, version_u32, n


def train_model(logger, max_iterations, alpha, beta):
    current_loss = float32(1.0)
    current_acc = float32(0.1)
    for _ in range(max_iterations):
        current_loss *= alpha
        current_acc += beta
        logger.log(current_acc, current_loss)


class PulseManager:
    def __init__(self, capacity, user_fn, *args, filepath="pulse_log.bin"):
        self.capacity = int(capacity)
        self.filepath = filepath
        self.args = args

        self.backend = None
        self.acc_view = None
        self.loss_view = None

        self.i = np.int64(0)
        self._iterations_run = 0

        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")

        self.runner = build_compiled_runner(user_fn)

    def _warmup(self):
        max_iterations, alpha, beta = self.args
        iters = np.int64(1)
        self.i = self.runner(self.acc_view, self.loss_view, self.i, iters, alpha, beta, self.capacity)
        self.i = np.int64(0)

    def pulse(self):
        max_iterations, alpha, beta = self.args
        self._iterations_run = int(max_iterations)
        self.i = self.runner(self.acc_view, self.loss_view, self.i, max_iterations, alpha, beta, self.capacity)

    def logger_size(self):
        n = self._iterations_run
        if n <= 0:
            return 0
        return self.capacity if n >= self.capacity else n

    def __enter__(self):
        self.backend = Pulse(self.capacity)
        self.acc_view, self.loss_view = self.backend.arrays()
        self.i = np.int64(0)
        self._warmup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend.set_cursor(self.logger_size())
        self.backend.flush(self.filepath)


def run_simulation():
    STORAGE_CAPACITY = 256
    TRAIN_ITERATIONS = np.int64(300_000_000)
    ALPHA = np.float32(0.9999999999)
    BETA = np.float32(0.00000000001)

    train_args = (TRAIN_ITERATIONS, ALPHA, BETA)

    with PulseManager(STORAGE_CAPACITY, train_model, *train_args) as pm:
        t0 = time.time()
        pm.pulse()
        t1 = time.time()
        print(f"Loop completed in: {t1 - t0:.4f}s")
        print(f"Logger size (valid records): {pm.logger_size()}")

    df, magic, version, nrecords = read_pulse_file("pulse_log.bin")
    print(f"File header: magic=0x{magic:08x} version={version} nrecords={nrecords}")

    print(f"Read back {len(df)} records.")
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Peak Memory: {mem_mb:.2f} MB")

    expected = min(int(TRAIN_ITERATIONS), STORAGE_CAPACITY)
    print(f"Size of df={len(df)} expected={expected}")
    assert len(df) == expected
    print(df.head())
    print(df.tail())
    print("Verification Successful.")


if __name__ == "__main__":
    run_simulation()
