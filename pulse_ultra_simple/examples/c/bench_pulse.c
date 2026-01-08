// bench_pulse.c
// Build:
//   gcc -O3 -march=native -ffast-math -funroll-loops -std=c11 bench_pulse.c -o bench_pulse
//
// Run:
//   ./bench_pulse [iters] [capacity] [k] [x_n]

#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(__GNUC__) || defined(__clang__)
#define INLINE __attribute__((always_inline)) inline
#else
#define INLINE inline
#endif

static INLINE uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static INLINE float frand01(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= (z >> 31);
    uint32_t r = (uint32_t)(z >> 40);
    return (float)r * (1.0f / 16777216.0f);
}

static INLINE void pulse(
    int64_t i,
    float acc,
    float loss,
    int64_t ring_mask,
    int64_t sparse_mask,
    float *acc_arr,
    float *loss_arr
) {
    if ((i & sparse_mask) != 0) return;
    int64_t ix = i & ring_mask;
    acc_arr[ix] = acc;
    loss_arr[ix] = loss;
}

static int64_t train_with_pulse(
    int64_t iterations,
    const float *x,
    int64_t x_n,
    float alpha,
    float beta,
    int64_t ring_mask,
    int64_t sparse_mask,
    float *acc_arr,
    float *loss_arr
) {
    float curr_acc = 0.1f;
    float curr_loss = 1.0f;

    uint64_t rng = 0x9E3779B97F4A7C15ull;
    int64_t h = (int64_t)0xCBF29CE484222325ull;
    const int64_t fnv = (int64_t)0x100000001B3ull;

    for (int64_t i = 0; i < iterations; i++) {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;

        int64_t j = (i + (int64_t)rng) % x_n;
        float u = x[j];

        float grad = (u - 0.5f) + curr_acc * 1e-3f;
        curr_acc += alpha * grad;
        curr_loss = curr_loss * beta + grad * grad + u * 1e-6f;

        int64_t q =
            (int64_t)(curr_acc * 1e6f) ^
            ((int64_t)(curr_loss * 1e6f) << 1);

        h = (h ^ q) * fnv;

        pulse(i, curr_acc, curr_loss, ring_mask, sparse_mask, acc_arr, loss_arr);
    }
    return h;
}

static int64_t train_without_pulse(
    int64_t iterations,
    const float *x,
    int64_t x_n,
    float alpha,
    float beta
) {
    float curr_acc = 0.1f;
    float curr_loss = 1.0f;

    uint64_t rng = 0x9E3779B97F4A7C15ull;
    int64_t h = (int64_t)0xCBF29CE484222325ull;
    const int64_t fnv = (int64_t)0x100000001B3ull;

    for (int64_t i = 0; i < iterations; i++) {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;

        int64_t j = (i + (int64_t)rng) % x_n;
        float u = x[j];

        float grad = (u - 0.5f) + curr_acc * 1e-3f;
        curr_acc += alpha * grad;
        curr_loss = curr_loss * beta + grad * grad + u * 1e-6f;

        int64_t q =
            (int64_t)(curr_acc * 1e6f) ^
            ((int64_t)(curr_loss * 1e6f) << 1);

        h = (h ^ q) * fnv;
    }
    return h;
}

int main(int argc, char **argv) {
    int64_t iters    = (argc > 1) ? atoll(argv[1]) : 100000000ll;
    int64_t capacity = (argc > 2) ? atoll(argv[2]) : 256ll;
    int64_t k        = (argc > 3) ? atoll(argv[3]) : 10ll;
    int64_t x_n      = (argc > 4) ? atoll(argv[4]) : 1000000ll;

    int64_t ring_mask   = capacity - 1;
    int64_t sparse_mask = (k == 0) ? 0 : ((1ll << k) - 1);

    float alpha = 0.0001f;
    float beta  = 0.9999f;

    float *x = aligned_alloc(64, (size_t)x_n * sizeof(float));
    float *acc_arr = aligned_alloc(64, (size_t)capacity * sizeof(float));
    float *loss_arr = aligned_alloc(64, (size_t)capacity * sizeof(float));

    uint64_t seed = 1234567;
    for (int64_t i = 0; i < x_n; i++) x[i] = frand01(&seed);

    train_with_pulse(1, x, x_n, alpha, beta, ring_mask, sparse_mask, acc_arr, loss_arr);
    train_without_pulse(1, x, x_n, alpha, beta);

    uint64_t t0, t1;
    int64_t out;

    t0 = now_ns();
    out = train_without_pulse(iters, x, x_n, alpha, beta);
    t1 = now_ns();
    printf("baseline (no pulse):        %.8fs  out=%lld\n",
           (t1 - t0) * 1e-9, (long long)out);

    t0 = now_ns();
    out = train_with_pulse(iters, x, x_n, alpha, beta,
                           ring_mask, sparse_mask, acc_arr, loss_arr);
    t1 = now_ns();
    printf("with pulse (sparse 2^%lld): %.8fs  out=%lld\n",
           (long long)k, (t1 - t0) * 1e-9, (long long)out);

    free(x);
    free(acc_arr);
    free(loss_arr);
    return 0;
}
