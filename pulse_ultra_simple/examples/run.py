from pulse import Pulse
from time import time, process_time
import resource

def without_pulse():
    MAX_EPOCHS = 10_000_000
    p = []
    loss = 5.0
    t0 = time()
    t0_cpu = process_time()
    for i in range(MAX_EPOCHS):
        epoch = i
        loss *= 0.999 + 0.1 * 0.0005
        p.append((epoch, loss))
    t1 = time()
    t1_cpu = process_time()
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("without pulse:", t1 - t0, "seconds")
    print("without pulse cpu:", t1_cpu - t0_cpu, "seconds")
    print("without pulse mem:", mem_mb, "MB")
    
    #p.flush("pulse_simple.bin")
    

def with_pulse():
    p = Pulse()
    MAX_EPOCHS = 50_000_000
    p.reserve(MAX_EPOCHS)

    loss = 5.0
    t0 = time()
    t0_cpu = process_time()
    for i in range(MAX_EPOCHS):
        epoch = i
        #loss *= 0.999 + 0.1 * 0.0005
        loss = int(2)
        p.append(1, 2)
    t1 = time()
    t1_cpu = process_time()
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("with pulse:", t1 - t0, "seconds")
    print("with pulse cpu:", t1_cpu - t0_cpu, "seconds")
    print("with pulse mem:", mem_mb, "MB")

    #p.flush("pulse_simple.bin")
    print("records:", p.size())
    #print("wrote pulse_simple.bin")

if __name__ == "__main__":
    with_pulse()
    #without_pulse()
