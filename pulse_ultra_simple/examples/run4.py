from pulse_fast_ext import Pulse
import time
import resource

def run_simulation():
    MAX_EPOCHS = 50_000_000
    logger = Pulse(MAX_EPOCHS)
    append = logger.append

    current_loss = 1.0
    current_acc = 0.1

    t0, t0_cpu = time.time(), time.process_time()
    for epoch in range(MAX_EPOCHS):
        current_loss *= 0.999999
        current_acc += 0.0000001
        # User sees high-level floats, C++ handles the quantization
        append(current_acc, current_loss)
    t1, t1_cpu = time.time(), time.process_time()
    

    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Memory Usage: {mem_mb:.2f} MB")     
    print(f"Loop completed in: {t1 - t0:.4f}s (CPU: {t1_cpu - t0_cpu:.4f}s)")
    print(f"Records: {logger.size()}")
    
    fpath = "pulse_log.bin"
    logger.flush(fpath)
    print("Data flushed to pulse_log.bin")
    
    print("Reading back data into pandas DataFrame...")
    from read_into_pandas import read_pulse_file
    df = read_pulse_file(fpath)
    print(f"Read back {len(df)} records from {fpath}")
    # print(df.head())
    # print(df.tail())    
    assert len(df) == MAX_EPOCHS
       
    
    

if __name__ == "__main__":
    run_simulation()
