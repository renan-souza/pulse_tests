import numpy as np
import time
import resource
from numba import njit, uint8, int64
from numba.experimental import jitclass
from pulse_fast_ext import Pulse

# 1. THE FAST BACKEND
# 'i' is uint8 and wraps at 256 per your requirement
spec = [('acc_view', uint8[:]), ('loss_view', uint8[:]), ('i', uint8)]


@jitclass(spec)
class PulseBuffer:
    def __init__(self, acc_view, loss_view):
        self.acc_view, self.loss_view, self.i = acc_view, loss_view, uint8(0)

    def log(self, acc, loss):
        idx = self.i
        self.acc_view[idx] = uint8(acc * 10.0 + 0.5)
        self.loss_view[idx] = uint8(loss * 10.0 + 0.5)
        # Ensure uint8 wrap-around logic (0-255 cycling)
        self.i = uint8((int(idx) + 1) % 256)


# 2. PURE PYTHON TRAINING FUNCTION
# The order of arguments here must match the order passed to PulseManager
def train_model(logger, max_iterations, alpha, beta):
    current_loss, current_acc = 1.0, 0.1
    for _ in range(max_iterations):
        current_loss *= alpha
        current_acc += beta
        # C++ backed high-speed log
        logger.log(current_acc, current_loss)


# 3. THE OPTIMIZED MANAGER
class PulseManager:
    def __init__(self, capacity, pulsed_function, *args, filepath="pulse_log.bin"):
        self.capacity = capacity
        self.filepath = filepath
        self.pulsed_function = njit(pulsed_function, cache=True, fastmath=True, nogil=True)

        # We store the positional arguments (iterations, alpha, beta, etc.)
        self.args = args
        self.logger = None
        self.backend = None

    def pulse(self):
        # We pass the logger first, then unpack the user's positional args
        # This keeps the training function signature clean.
        self.pulsed_function(self.logger, *self.args)

    def _warmup(self):
        self.pulsed_function(self.logger, 1, *self.args[1:])  # compile with same types
        self.backend.set_cursor(0) # TODO?!
        self.logger.i = uint8(0)

    def __enter__(self):
        # Initialize C++ backend and extract NumPy views
        self.backend = Pulse(self.capacity)
        acc_arr, loss_arr = self.backend.arrays()
        self.logger = PulseBuffer(acc_arr, loss_arr)

        self._warmup()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush the pre-allocated C++ buffers to disk
        self.backend.flush(self.filepath, self.capacity)


# --- EXECUTION & VERIFICATION ---
def run_simulation():
    # 1. Define storage and training parameters separately
    STORAGE_CAPACITY = 50_000_000
    TRAIN_ITERATIONS = 50_000_000
    ALPHA = 0.999999
    BETA = 0.0000001

    # 2. Package positional arguments for the training function
    # Note: These correspond to (iterations, alpha, beta)
    train_args = [TRAIN_ITERATIONS, ALPHA, BETA]

    # 3. Run simulation
    with PulseManager(STORAGE_CAPACITY, train_model, *train_args) as pm:
        t0 = time.time()
        pm.pulse()
        t1 = time.time()

    print(f"Loop completed in: {t1 - t0:.4f}s")

    # 4. Verification
    print("Reading back data into pandas DataFrame...")
    from read_into_pandas import read_pulse_file
    df = read_pulse_file("pulse_log.bin")

    print(f"Read back {len(df)} records.")
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Peak Memory: {mem_mb:.2f} MB")

    assert len(df) == STORAGE_CAPACITY
    print("Verification Successful.")


if __name__ == "__main__":
    run_simulation()