#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/task_info.h>
#include <sys/resource.h>
#elif defined(__linux__)
#include <sys/resource.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PULSE_INLINE __attribute__((always_inline)) inline
#else
#define PULSE_INLINE inline
#endif

static PULSE_INLINE double now_wall_s(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static PULSE_INLINE double now_cpu_s(void) {
    struct timespec ts;
#if defined(CLOCK_PROCESS_CPUTIME_ID)
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double max_rss_mb(void) {
#if defined(__linux__)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1.0;
    return (double)ru.ru_maxrss / 1024.0;
#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return -1.0;
    return (double)info.resident_size / (1024.0 * 1024.0);
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1.0;
    return (double)ru.ru_maxrss / 1024.0;
#endif
}

static PULSE_INLINE uint8_t quant10(double x) {
    return (uint8_t)(x * 10.0 + 0.5);
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    const uint64_t N = 50000000ULL;

    uint8_t* epoch_u8 = (uint8_t*)malloc((size_t)N);
    uint8_t* acc_u8   = (uint8_t*)malloc((size_t)N);
    uint8_t* loss_u8  = (uint8_t*)malloc((size_t)N);

    if (!epoch_u8 || !acc_u8 || !loss_u8) {
        fprintf(stderr, "malloc failed\n");
        free(epoch_u8);
        free(acc_u8);
        free(loss_u8);
        return 1;
    }

    double current_loss = 1.0;
    double current_acc = 0.1;

    const double scale = 10.0;
    const double bias = 0.5;
    const uint64_t mask = 255ULL;

    double t0 = now_wall_s();
    double c0 = now_cpu_s();

    for (uint64_t epoch = 0; epoch < N; ++epoch) {
        current_loss *= 0.999999;
        current_acc += 0.0000001;

        epoch_u8[epoch] = (uint8_t)(epoch & mask);
        acc_u8[epoch]   = (uint8_t)(current_acc * scale + bias);
        loss_u8[epoch]  = (uint8_t)(current_loss * scale + bias);
    }

    double t1 = now_wall_s();
    double c1 = now_cpu_s();

    double rss = max_rss_mb();
    printf("Memory Usage: %.2f MB\n", rss);
    printf("Loop completed in: %.4fs (CPU: %.4fs)\n", (t1 - t0), (c1 - c0));

    volatile uint64_t sink = 0;
    sink += epoch_u8[0] + epoch_u8[N - 1] + acc_u8[0] + acc_u8[N - 1] + loss_u8[0] + loss_u8[N - 1];
    if (sink == 0xFFFFFFFFFFFFFFFFULL) printf("sink=%" PRIu64 "\n", sink);

    free(epoch_u8);
    free(acc_u8);
    free(loss_u8);
    return 0;
}
