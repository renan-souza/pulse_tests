// pulse_ring_fair.c
//
// Matches the FINAL Python version exactly in the ways that matter for a fair comparison:
//
// - Ring behavior: j = i % CAPACITY on every iteration
// - Single "log" call per iteration that writes BOTH fields
// - Same quantization: uint8(x * 10.0 + 0.5)
// - Same initial values: curr_loss=1.0, curr_acc=0.1
// - Same update order per iter: loss*=alpha; acc+=beta; then log(acc, loss)
// - Same on-disk format as pulse_fast_ext flush:
//     PulseFileHeader (packed, 16 bytes): magic(u32), version(u32), nrecords(u64)
//     then acc block:  nrecords bytes (uint8)
//     then loss block: nrecords bytes (uint8)
//
// IMPORTANT:
// - In your Python final code, CAPACITY=256 and you flush exactly CAPACITY records.
// - So here we also flush exactly CAPACITY records, not TRAIN_ITERATIONS.
// - TRAIN_ITERATIONS can be >> CAPACITY; we overwrite the ring many times.
//

#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define PULSE_MAGIC 0x31534C50u
#define PULSE_VERSION 1u

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t nrecords;
} PulseFileHeader;
#pragma pack(pop)

static inline uint8_t quantize_u8(double x) {
    return (uint8_t)(x * 10.0 + 0.5);
}

static int write_all(int fd, const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w > 0) {
            p += (size_t)w;
            n -= (size_t)w;
            continue;
        }
        if (w < 0 && errno == EINTR) continue;
        return -1;
    }
    return 0;
}

int main(int argc, char** argv) {
    // Defaults chosen to mirror your Python "final version"
    int64_t capacity = 256;
    int64_t train_iters = 1000000000LL;
    double alpha = 0.999999;
    double beta = 0.0000001;

    if (argc >= 2) capacity = atoll(argv[1]);
    if (argc >= 3) train_iters = atoll(argv[2]);
    if (argc >= 4) alpha = atof(argv[3]);
    if (argc >= 5) beta = atof(argv[4]);

    if (capacity <= 0) {
        fprintf(stderr, "capacity must be > 0\n");
        return 1;
    }
    if (train_iters < 0) {
        fprintf(stderr, "train_iters must be >= 0\n");
        return 1;
    }

    // Allocate two separate uint8 arrays, matching the Python extension buffers
    uint8_t* acc = (uint8_t*)malloc((size_t)capacity);
    uint8_t* loss = (uint8_t*)malloc((size_t)capacity);
    if (!acc || !loss) {
        fprintf(stderr, "malloc failed\n");
        free(acc);
        free(loss);
        return 1;
    }

    double curr_loss = 1.0;
    double curr_acc = 0.1;

    const int64_t cap = capacity;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Training loop: matches Python ordering exactly
    for (int64_t i = 0; i < train_iters; i++) {
        curr_loss *= alpha;
        curr_acc += beta;

        int64_t j = i % cap; // ring behavior (same as Python)
        acc[j] = quantize_u8(curr_acc);
        loss[j] = quantize_u8(curr_loss);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

    // Flush: write EXACTLY `capacity` records, because Python flushes cursor=capacity
    const char* path = "pulse_log.bin";
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) {
        perror("open");
        free(acc);
        free(loss);
        return 1;
    }

    PulseFileHeader h;
    h.magic = PULSE_MAGIC;
    h.version = PULSE_VERSION;
    h.nrecords = (uint64_t)capacity;

    int rc = 0;
    if (write_all(fd, &h, sizeof(h)) != 0) rc = -1;
    if (rc == 0 && capacity > 0) {
        if (write_all(fd, acc, (size_t)capacity) != 0) rc = -1;
        if (write_all(fd, loss, (size_t)capacity) != 0) rc = -1;
    }
    close(fd);

    free(acc);
    free(loss);

    if (rc != 0) {
        perror("write");
        return 1;
    }

    printf("Loop completed in: %.6fs\n", sec);
    printf("Wrote pulse_log.bin with nrecords=%" PRId64 " (acc block then loss block)\n", capacity);
    return 0;
}
