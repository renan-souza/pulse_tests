import numpy as np
import time
import resource

def run_simulation():
    N = 50_000_000

    epoch_u8 = np.empty(N, dtype=np.uint8)
    acc_u8 = np.empty(N, dtype=np.uint8)
    loss_u8 = np.empty(N, dtype=np.uint8)

    current_loss = 1.0
    current_acc = 0.1

    scale = 10.0
    bias = 0.5
    mask = 255

    t0, t0_cpu = time.time(), time.process_time()

    for epoch in range(N):
        current_loss *= 0.999999
        current_acc += 0.0000001

        epoch_u8[epoch] = epoch & mask
        acc_u8[epoch] = current_acc * scale + bias
        loss_u8[epoch] = current_loss * scale + bias

    t1, t1_cpu = time.time(), time.process_time()

    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Memory Usage: {mem_mb:.2f} MB")
    print(f"Loop completed in: {t1 - t0:.4f}s (CPU: {t1_cpu - t0_cpu:.4f}s)")

if __name__ == "__main__":
    run_simulation()
