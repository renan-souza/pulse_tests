// src/pulse_rt_ext.c
#include <Python.h>
#include <sys/mman.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

typedef struct {
    float *acc, *loss;
    int64_t *seq;
    uint64_t *ticks;
    size_t n;
} BackingStore;

static BackingStore g_store = {NULL, NULL, NULL, NULL, 0};

static PyObject* py_init(PyObject* self, PyObject* args) {
    long long n;
    if (!PyArg_ParseTuple(args, "L", &n)) return NULL;
    g_store.n = (size_t)n;

    // Allocate 100M records for all fields including ticks
    g_store.acc = mmap(NULL, g_store.n * 4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
    g_store.loss = mmap(NULL, g_store.n * 4, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
    g_store.seq = mmap(NULL, g_store.n * 8, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
    g_store.ticks = mmap(NULL, g_store.n * 8, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);

    // Pre-fault the memory to prevent page faults in the hot loop
    memset(g_store.acc, 0, g_store.n * 4);
    memset(g_store.loss, 0, g_store.n * 4);
    memset(g_store.seq, 0, g_store.n * 8);
    memset(g_store.ticks, 0, g_store.n * 8);

    return Py_BuildValue("KKKK", (uintptr_t)g_store.acc, (uintptr_t)g_store.loss,
                         (uintptr_t)g_store.seq, (uintptr_t)g_store.ticks);
}

static PyObject* py_save(PyObject* self, PyObject* args) {
    const char* path;
    if (!PyArg_ParseTuple(args, "s", &path)) return NULL;
    int fd = open(path, O_CREAT|O_TRUNC|O_WRONLY, 0644);
    if (fd >= 0) {
        write(fd, g_store.acc, g_store.n * 4);
        write(fd, g_store.loss, g_store.n * 4);
        write(fd, g_store.seq, g_store.n * 8);
        write(fd, g_store.ticks, g_store.n * 8);
        close(fd);
    }
    Py_RETURN_TRUE;
}

static PyMethodDef Methods[] = {
    {"init", py_init, METH_VARARGS, ""},
    {"save", py_save, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "pulse_rt_ext", NULL, -1, Methods};
PyMODINIT_FUNC PyInit_pulse_rt_ext(void) { return PyModule_Create(&module); }