import numpy as np
import time, resource

from pulse_fast_ext import Pulse as CPulse

class Pulse:
    def __init__(self, capacity, batch_size=1000):
        self.backend = CPulse(capacity)
        self.batch_size = batch_size
        self._e = np.empty(batch_size, dtype=np.int32)
        self._a = np.empty(batch_size, dtype=np.float32)
        self._l = np.empty(batch_size, dtype=np.float32)
        self._idx = 0

    def append(self, epoch, acc, loss):
        i = self._idx
        self._e[i], self._a[i], self._l[i] = epoch, acc, loss
        self._idx = i + 1
        
        if self._idx == self.batch_size:
            self.backend.push_batch(self._e, self._a, self._l)
            self._idx = 0

    def size(self):
        return self.backend.size() + self._idx


def run_simulation():
    MAX_EPOCHS = 50_000_000
    logger = Pulse(MAX_EPOCHS)
    current_loss, current_acc = 1.0, 0.1

    t0, t0_cpu = time.time(), time.process_time()
    for epoch in range(MAX_EPOCHS):
        current_loss *= 0.999999
        current_acc += 0.0000001
        # Complexity is hidden here
        logger.append(epoch, current_acc, current_loss)
    
    t1, t1_cpu = time.time(), time.process_time()
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"Memory Usage: {mem_mb:.2f} MB")     
    print(f"Loop completed in: {t1 - t0:.4f}s (CPU: {t1_cpu - t0_cpu:.4f}s)")
    


if __name__ == "__main__":
    run_simulation()
