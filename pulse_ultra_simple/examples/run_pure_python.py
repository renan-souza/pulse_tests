import numpy as np
import time
import resource

def run_simulation():
    MAX_EPOCHS = 50_000_000
    
    # Pre-allocate separate arrays for maximum speed
    # Assigning to individual arrays is faster than a structured array inside a loop
    accuracies = np.empty(MAX_EPOCHS, dtype=np.float32)
    losses = np.empty(MAX_EPOCHS, dtype=np.float32)
    epochs = np.empty(MAX_EPOCHS, dtype=np.int32)
    # Using 'S' for fixed-length byte strings
    
    # Simulation constants
    current_loss = 1.0
    current_acc = 0.1

    t0, t0_cpu = time.time(), time.process_time()
    
    # The Mandatory Loop
    for epoch in range(MAX_EPOCHS):
        # Simulation Logic: 
        # Each iteration depends on the previous one
        current_loss *= 0.999999
        current_acc += 0.0000001
        
        # FASTEST ASSIGNMENT: Direct index access to separate arrays
        epochs[epoch] = epoch
        accuracies[epoch] = current_acc
        losses[epoch] = current_loss

    t1, t1_cpu = time.time(), time.process_time()
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 
    print(f"Memory Usage: {mem_mb:.2f} MB")     
    print(f"Loop completed in: {t1 - t0:.4f}s (CPU: {t1_cpu - t0_cpu:.4f}s)")

if __name__ == "__main__":
    run_simulation()