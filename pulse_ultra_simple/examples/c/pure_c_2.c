#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define STORAGE_CAPACITY 50000000
#define TRAIN_ITERATIONS 50000000
#define ALPHA 0.999999f
#define BETA 0.0000001f

#pragma pack(push, 1)
typedef struct {
    uint8_t accuracy;
    uint8_t loss;
} LogRecord;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t nrecords;
} PulseHeader;
#pragma pack(pop)

// Simulates the PulseBuffer.log logic
void log_record(LogRecord* data, size_t* i, float acc, float loss) {
    // Note: Python code used idx = i then i = (idx + 1) % 256.
    // To match exactly, we use a local cast for the cycling index.
    uint8_t idx = (uint8_t)(*i % 256);

    data[idx].accuracy = (uint8_t)(acc * 10.0f + 0.5f);
    data[idx].loss = (uint8_t)(loss * 10.0f + 0.5f);

    (*i)++;
}

int main() {
    // 1. Setup (PulseManager.__enter__)
    LogRecord* records = malloc(STORAGE_CAPACITY * sizeof(LogRecord));
    if (!records) return 1;

    float current_loss = 1.0f;
    float current_acc = 0.1f;
    size_t cursor = 0;

    clock_t start = clock();

    // 2. Training Loop (train_model)
    for (size_t _ = 0; _ < TRAIN_ITERATIONS; _++) {
        current_loss *= ALPHA;
        current_acc += BETA;
        log_record(records, &cursor, current_acc, current_loss);
    }

    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // 3. Flush (PulseManager.__exit__)
    FILE* f = fopen("pulse_log.bin", "wb");
    PulseHeader header = {0x31534C50, 1, STORAGE_CAPACITY};

    fwrite(&header, sizeof(PulseHeader), 1, f);
    // Writing records sequentially to match your read_pulse_file logic
    // (Note: If your Python C-ext writes two separate blocks,
    // you would iterate and write accuracy then loss here).
    fwrite(records, sizeof(LogRecord), STORAGE_CAPACITY, f);

    fclose(f);
    free(records);

    printf("Loop completed in: %.4fs\n", cpu_time);
    return 0;
}