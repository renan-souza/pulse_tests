import numpy as np
import time

class PulseLogger:
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs
        # Internal pre-allocation (Structure of Arrays for speed)
        self._acc = np.empty(max_epochs, dtype=np.float32)
        self._loss = np.empty(max_epochs, dtype=np.float32)
        self._epochs = np.empty(max_epochs, dtype=np.int32)
        self._cursor = 0

    def append(self, epoch, acc, loss):
        # We use the cursor for direct index assignment
        idx = self._cursor
        self._acc[idx] = acc
        self._loss[idx] = loss
        self._epochs[idx] = epoch
        self._cursor = idx + 1

    def get_data(self):
        # Return a view of the filled data
        return self._acc[:self._cursor], self._loss[:self._cursor]

def run_simulation():
    MAX_EPOCHS = 50_000_000
    logger = PulseLogger(MAX_EPOCHS)
    
    current_loss = 1.0
    current_acc = 0.1
    

    t0 = time.time()
    
    # Clean user loop
    for epoch in range(MAX_EPOCHS):
        current_loss *= 0.999999
        current_acc += 0.0000001
        
        # Single line, hidden complexity
        logger.append(epoch, current_acc, current_loss)

    print(f"Loop time: {time.time() - t0:.4f}s")

if __name__ == "__main__":
    run_simulation()
