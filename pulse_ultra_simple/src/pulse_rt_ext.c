// src/pulse_rt_ext.c
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAGIC 0x31534C50u
#define VERSION 7u

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t n;
    double base_wall_epoch;
    uint64_t base_tsc;
    double tps;
    uint8_t has_time;
    uint8_t pad[7];
} PulseFileHeader;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    uint64_t start_index;
    uint64_t nrecords;
    uint32_t flags;
    uint32_t pad;
} PulseShardBlockHeader;
#pragma pack(pop)

static inline int write_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
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

static int mkdirs_for_path(const char *path) {
    const char *slash = strrchr(path, '/');
    if (!slash) return 0;
    size_t len = (size_t)(slash - path);
    if (len == 0) return 0;

    char *tmp = (char *)malloc(len + 1);
    if (!tmp) return -1;
    memcpy(tmp, path, len);
    tmp[len] = '\0';

    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
    free(tmp);
    return 0;
}

int pulse_copy256(
    uintptr_t back_acc_ptr,
    uintptr_t back_loss_ptr,
    uintptr_t back_seq_ptr,
    uintptr_t back_ticks_ptr,
    uint64_t base_index,
    uintptr_t front_acc_ptr,
    uintptr_t front_loss_ptr,
    uintptr_t front_seq_ptr,
    uintptr_t front_ticks_ptr,
    int32_t has_time
) {
    if (!back_acc_ptr || !back_loss_ptr || !back_seq_ptr || !front_acc_ptr || !front_loss_ptr || !front_seq_ptr) return -1;

    float *back_acc = (float *)back_acc_ptr;
    float *back_loss = (float *)back_loss_ptr;
    int64_t *back_seq = (int64_t *)back_seq_ptr;

    const float *front_acc = (const float *)front_acc_ptr;
    const float *front_loss = (const float *)front_loss_ptr;
    const int64_t *front_seq = (const int64_t *)front_seq_ptr;

    size_t off = (size_t)base_index;

    memcpy(back_acc + off, front_acc, 256u * sizeof(float));
    memcpy(back_loss + off, front_loss, 256u * sizeof(float));
    memcpy(back_seq + off, front_seq, 256u * sizeof(int64_t));

    if (has_time) {
        if (!back_ticks_ptr || !front_ticks_ptr) return -1;
        uint64_t *back_ticks = (uint64_t *)back_ticks_ptr;
        const uint64_t *front_ticks = (const uint64_t *)front_ticks_ptr;
        memcpy(back_ticks + off, front_ticks, 256u * sizeof(uint64_t));
    }
    return 0;
}

typedef struct {
    const char *basepath;
    int shard_id;
    int nshards;
    int has_time;

    double base_wall_epoch;
    uint64_t base_tsc;
    double tps;

    uint64_t n_total;

    const float *acc;
    const float *loss;
    const int64_t *seq;
    const uint64_t *ticks;

    int rc;
} ShardJob;

static void *writer_thread_main(void *arg) {
    ShardJob *job = (ShardJob *)arg;

    uint64_t n = job->n_total;
    int s = job->shard_id;
    int S = job->nshards;

    uint64_t chunk = (n + (uint64_t)S - 1u) / (uint64_t)S;
    uint64_t start = (uint64_t)s * chunk;
    uint64_t end = start + chunk;
    if (end > n) end = n;

    char path[4096];
    snprintf(path, sizeof(path), "%s.shard%03d", job->basepath, s);
    if (mkdirs_for_path(path) != 0) {
        job->rc = -1;
        return NULL;
    }

    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) {
        job->rc = -1;
        return NULL;
    }

    PulseFileHeader h;
    memset(&h, 0, sizeof(h));
    h.magic = MAGIC;
    h.version = VERSION;
    h.n = (uint64_t)n;
    h.base_wall_epoch = job->base_wall_epoch;
    h.base_tsc = job->base_tsc;
    h.tps = job->tps;
    h.has_time = (uint8_t)(job->has_time ? 1 : 0);

    if (write_all(fd, &h, sizeof(h)) != 0) {
        close(fd);
        job->rc = -1;
        return NULL;
    }

    uint64_t nrec = (end > start) ? (end - start) : 0;
    PulseShardBlockHeader bh;
    memset(&bh, 0, sizeof(bh));
    bh.start_index = start;
    bh.nrecords = nrec;
    bh.flags = (uint32_t)(job->has_time ? 1u : 0u);

    if (write_all(fd, &bh, sizeof(bh)) != 0) {
        close(fd);
        job->rc = -1;
        return NULL;
    }

    if (nrec > 0) {
        const float *acc = job->acc + start;
        const float *loss = job->loss + start;
        const int64_t *seq = job->seq + start;

        if (write_all(fd, acc, (size_t)nrec * sizeof(float)) != 0) { close(fd); job->rc = -1; return NULL; }
        if (write_all(fd, loss, (size_t)nrec * sizeof(float)) != 0) { close(fd); job->rc = -1; return NULL; }
        if (write_all(fd, seq, (size_t)nrec * sizeof(int64_t)) != 0) { close(fd); job->rc = -1; return NULL; }

        if (job->has_time) {
            const uint64_t *ticks = job->ticks + start;
            if (write_all(fd, ticks, (size_t)nrec * sizeof(uint64_t)) != 0) { close(fd); job->rc = -1; return NULL; }
        }
    }

    close(fd);
    job->rc = 0;
    return NULL;
}

int pulse_write_shards(
    uintptr_t filepath_ptr,
    int64_t filepath_len,
    int64_t n,
    int32_t has_time,
    double base_wall_epoch,
    uint64_t base_tsc,
    double tps,
    uintptr_t acc_ptr,
    uintptr_t loss_ptr,
    uintptr_t seq_ptr,
    uintptr_t ticks_ptr,
    int32_t nshards
) {
    if (!filepath_ptr || filepath_len <= 0 || n < 0 || nshards <= 0) return -1;

    char *basepath = (char *)malloc((size_t)filepath_len + 1u);
    if (!basepath) return -1;
    memcpy(basepath, (const void *)filepath_ptr, (size_t)filepath_len);
    basepath[filepath_len] = '\0';

    pthread_t *threads = (pthread_t *)calloc((size_t)nshards, sizeof(pthread_t));
    ShardJob *jobs = (ShardJob *)calloc((size_t)nshards, sizeof(ShardJob));
    if (!threads || !jobs) {
        free(basepath);
        free(threads);
        free(jobs);
        return -1;
    }

    for (int s = 0; s < nshards; s++) {
        jobs[s].basepath = basepath;
        jobs[s].shard_id = s;
        jobs[s].nshards = nshards;
        jobs[s].has_time = (has_time != 0);

        jobs[s].base_wall_epoch = base_wall_epoch;
        jobs[s].base_tsc = base_tsc;
        jobs[s].tps = tps;

        jobs[s].n_total = (uint64_t)n;

        jobs[s].acc = (const float *)acc_ptr;
        jobs[s].loss = (const float *)loss_ptr;
        jobs[s].seq = (const int64_t *)seq_ptr;
        jobs[s].ticks = (const uint64_t *)ticks_ptr;

        jobs[s].rc = -1;

        int rc = pthread_create(&threads[s], NULL, writer_thread_main, &jobs[s]);
        if (rc != 0) {
            free(basepath);
            free(threads);
            free(jobs);
            return -1;
        }
    }

    int ok = 0;
    for (int s = 0; s < nshards; s++) pthread_join(threads[s], NULL);
    for (int s = 0; s < nshards; s++) if (jobs[s].rc == 0) ok++;

    free(basepath);
    free(threads);
    free(jobs);

    return (ok == nshards) ? 0 : -1;
}

static PyMethodDef Methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "pulse_rt_ext",
    "Pulse helpers (copy256 + post-timing sharded writer).",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_pulse_rt_ext(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
