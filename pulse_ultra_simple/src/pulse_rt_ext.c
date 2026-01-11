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
    size_t n;
} BackingStore;

static BackingStore g_store = {NULL, NULL, NULL, 0};

// Simplified: Takes destination offset and source pointers for the 3 buffers
void pulse_copy256(uint64_t off, float* s_acc, float* s_loss, int64_t* s_seq) {
    memcpy(g_store.acc + off, s_acc, 256 * sizeof(float));
    memcpy(g_store.loss + off, s_loss, 256 * sizeof(float));
    memcpy(g_store.seq + off, s_seq, 256 * sizeof(int64_t));
}

static PyObject* py_init(PyObject* self, PyObject* args) {
    long long n;
    if (!PyArg_ParseTuple(args, "L", &n)) return NULL;
    g_store.n = (size_t)n;

    // Allocate 100M record buffers (Pre-faulted via MAP_POPULATE)
    g_store.acc = mmap(NULL, g_store.n * sizeof(float), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
    g_store.loss = mmap(NULL, g_store.n * sizeof(float), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
    g_store.seq = mmap(NULL, g_store.n * sizeof(int64_t), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);

    return PyLong_FromUnsignedLongLong((uintptr_t)pulse_copy256);
}

static PyObject* py_save(PyObject* self, PyObject* args) {
    const char* path;
    if (!PyArg_ParseTuple(args, "s", &path)) return NULL;
    int fd = open(path, O_CREAT|O_TRUNC|O_WRONLY, 0644);
    if (fd < 0) Py_RETURN_FALSE;
    write(fd, g_store.acc, g_store.n * sizeof(float));
    write(fd, g_store.loss, g_store.n * sizeof(float));
    write(fd, g_store.seq, g_store.n * sizeof(int64_t));
    close(fd);
    Py_RETURN_TRUE;
}

static PyMethodDef Methods[] = {
    {"init", py_init, METH_VARARGS, ""},
    {"save", py_save, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "pulse_rt_ext", NULL, -1, Methods};
PyMODINIT_FUNC PyInit_pulse_rt_ext(void) { return PyModule_Create(&module); }