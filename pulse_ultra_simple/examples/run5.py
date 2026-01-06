import numpy as np
import time
from numba import njit
from pulse_fast_ext import Pulse

# This function is inlined by Numba, so there is ZERO call overhead
@njit(inline='always')
def log(acc_view, loss_view, i, acc, loss):
    acc_view[i] = np.uint8(acc * 10.0 + 0.5)
    loss_view[i] = np.uint8(loss * 10.0 + 0.5)

@njit
def main_loop(acc_view, loss_view, iterations):
    current_loss = 1.0
    current_acc = 0.1
    
    for i in range(iterations):
        current_loss *= 0.999999
        current_acc += 0.0000001
        
        # SINGLE CALL: Complexity is hidden, performance is preserved
        log(acc_view, loss_view, i, current_acc, current_loss)

def run_simulation():
    MAX_EPOCHS = 50_000_000
    logger = Pulse(MAX_EPOCHS)
    
    # Extract the raw memory views once
    acc_view, loss_view = logger.arrays()

    t0 = time.time()
    # Pass views into the compiled loop
    main_loop(acc_view, loss_view, MAX_EPOCHS)
    t1 = time.time()

    print(f"Loop completed in: {t1 - t0:.4f}s")
    
    
    fpath = "pulse_log.bin"
    # Flush using the C-extension logic
    logger.flush(fpath, MAX_EPOCHS)
    print("Data flushed to binary.")
    
    # print(f"Records: {logger.size()}")  
    
    print("Reading back data into pandas DataFrame...")
    from read_into_pandas import read_pulse_file
    df = read_pulse_file(fpath)
    print(f"Read back {len(df)} records from {fpath}")
    print(df.head())
    print(df.tail())    
    assert len(df) == MAX_EPOCHS
       
    
    

    

if __name__ == "__main__":
    run_simulation()