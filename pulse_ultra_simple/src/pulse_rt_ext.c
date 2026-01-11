// src/pulse_rt_ext.c
#include <Python.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RING_SIZE 256
#define RECORD_SZ 24
#define VIRTUAL_MAX (1024ULL * 1024ULL * 1024ULL * 8ULL)

typedef struct {
    uint8_t data[RING_SIZE * RECORD_SZ];
    _Atomic int32_t state;
} Ring;

#define NUM_RINGS 4096

typedef struct {
    Ring *rings;
    uint8_t *backing_store;
    _Atomic uint64_t producer_count;
    _Atomic uint64_t consumer_count;
    int stop;
} Runtime;

static Runtime g_rt = {0};

static void* consumer_main(void* arg) {
    struct timespec ts = {0, 1000}; // 1 microsecond
    while (!g_rt.stop) {
        uint64_t p_count = atomic_load(&g_rt.producer_count);
        uint64_t c_count = atomic_load(&g_rt.consumer_count);

        if (c_count >= p_count) {
            syscall(SYS_futex, &g_rt.producer_count, FUTEX_WAIT, p_count, NULL, NULL, 0);
            continue;
        }

        Ring *r = &g_rt.rings[c_count % NUM_RINGS];
        while(atomic_load(&r->state) != 2 && !g_rt.stop) {
            nanosleep(&ts, NULL);
        }

        if (g_rt.stop) break;

        memcpy(g_rt.backing_store + (c_count * RING_SIZE * RECORD_SZ), r->data, RING_SIZE * RECORD_SZ);

        atomic_store(&r->state, 0);
        atomic_fetch_add(&g_rt.consumer_count, 1);
    }
    return NULL;
}

static PyObject* py_init(PyObject* self, PyObject* args) {
    g_rt.backing_store = mmap(NULL, VIRTUAL_MAX, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    g_rt.rings = (Ring*)calloc(NUM_RINGS, sizeof(Ring));
    g_rt.stop = 0;
    g_rt.producer_count = 0;
    g_rt.consumer_count = 0;

    for(int i=0; i<4; i++) {
        pthread_t thread;
        pthread_create(&thread, NULL, consumer_main, NULL);
        pthread_detach(thread);
    }

    return PyLong_FromUnsignedLongLong((uintptr_t)g_rt.rings);
}

static PyObject* py_save(PyObject* self, PyObject* args) {
    const char* path;
    if (!PyArg_ParseTuple(args, "s", &path)) return NULL;
    g_rt.stop = 1;

    // Wake consumers one last time
    atomic_fetch_add(&g_rt.producer_count, 1);
    syscall(SYS_futex, &g_rt.producer_count, FUTEX_WAKE, 4, NULL, NULL, 0);

    usleep(100000); // Wait for drain

    uint64_t total_written = atomic_load(&g_rt.consumer_count) * RING_SIZE * RECORD_SZ;
    int fd = open(path, O_CREAT|O_TRUNC|O_WRONLY, 0644);
    if (fd >= 0) {
        write(fd, g_rt.backing_store, total_written);
        close(fd);
    }
    Py_RETURN_TRUE;
}

static PyMethodDef Methods[] = {
    {"init", py_init, METH_NOARGS, ""},
    {"save", py_save, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "pulse_rt_ext", NULL, -1, Methods};
PyMODINIT_FUNC PyInit_pulse_rt_ext(void) { return PyModule_Create(&module); }