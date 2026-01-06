import numpy as np
import time
import resource
from numba import njit, uint8, float32, int64
from numba.experimental import jitclass
from pulse_fast_ext import Pulse

# Define the structure for Numba to compile the class into C-memory
spec = [
    ('acc_view', uint8[:]),   # 1D array of uint8
    ('loss_view', uint8[:]),  # 1D array of uint8
    ('i', int64),             # Current index/cursor
]

@jitclass(spec)
class PulseContext:
    def __init__(self, acc_view, loss_view):
        self.acc_view = acc_view
        self.loss_view = loss_view
        self.i = 0

    def log(self, acc, loss):
        # Hidden quantization and indexing
        idx = self.i
        self.acc_view[idx] = uint8(acc * 10.0 + 0.5)
        self.loss_view[idx] = uint8(loss * 10.0 + 0.5)
        self.i = idx + 1

@njit
def main_loop(ctx, iterations):
    current_loss = 1.0
    current_acc = 0.1
    
    for _ in range(iterations):
        current_loss *= 0.999999
        current_acc += 0.0000001
        
        # CLEANEST API: No views or indices passed manually
        ctx.log(current_acc, current_loss)

def run_simulation():
    MAX_EPOCHS = 50_000_000
    logger = Pulse(MAX_EPOCHS)
    
    # Get the raw memory views from C extension
    acc_arr, loss_arr = logger.arrays()
    
    # Initialize the compiled context
    ctx = PulseContext(acc_arr, loss_arr)

    t0 = time.time()
    main_loop(ctx, MAX_EPOCHS)
    t1 = time.time()

    # Reporting
    print(f"Loop completed in: {t1 - t0:.4f}s")
    
    fpath = "pulse_log.bin"
    # Flush using the C-extension logic
    logger.flush(fpath, MAX_EPOCHS)
    print("Data flushed to binary.")
    
    from read_into_pandas import read_pulse_file
    df = read_pulse_file(fpath)
    print(f"Read back {len(df)} records.")
    
    assert len(df) == MAX_EPOCHS
    print("Verification Successful.")

if __name__ == "__main__":
    run_simulation()