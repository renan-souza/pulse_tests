#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stddef.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#if defined(_MSC_VER)
#define PULSE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define PULSE_INLINE __attribute__((always_inline)) inline
#else
#define PULSE_INLINE inline
#endif

#define PULSE_QUANTIZE_10(x) ((x) * 10.0 + 0.5)

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t nrecords;
} PulseFileHeader;
#pragma pack(pop)

#define PULSE_MAGIC 0x31534C50u
#define PULSE_VERSION 1u

typedef struct {
    PyObject_HEAD
    Py_ssize_t capacity;
    Py_ssize_t cursor;

    PyObject* acc_arr;
    PyObject* loss_arr;

    uint8_t* acc_ptr;
    uint8_t* loss_ptr;
} PulseObject;

static int
Pulse_init(PulseObject* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {"capacity", NULL};
    Py_ssize_t cap = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "n", kwlist, &cap)) {
        return -1;
    }
    if (cap <= 0) {
        PyErr_SetString(PyExc_ValueError, "capacity must be > 0");
        return -1;
    }

    self->capacity = cap;
    self->cursor = 0;

    self->acc_arr = NULL;
    self->loss_arr = NULL;

    self->acc_ptr = NULL;
    self->loss_ptr = NULL;

    npy_intp dims[1];
    dims[0] = (npy_intp)cap;

    self->acc_arr   = (PyObject*)PyArray_SimpleNew(1, dims, NPY_UINT8);
    self->loss_arr  = (PyObject*)PyArray_SimpleNew(1, dims, NPY_UINT8);

    if (!self->acc_arr || !self->loss_arr) {
        Py_XDECREF(self->acc_arr);
        Py_XDECREF(self->loss_arr);
        self->acc_arr = self->loss_arr = NULL;
        PyErr_SetString(PyExc_MemoryError, "failed to allocate numpy arrays");
        return -1;
    }

    self->acc_ptr   = (uint8_t*)PyArray_DATA((PyArrayObject*)self->acc_arr);
    self->loss_ptr  = (uint8_t*)PyArray_DATA((PyArrayObject*)self->loss_arr);

    return 0;
}

static void
Pulse_dealloc(PulseObject* self)
{
    Py_XDECREF(self->acc_arr);
    Py_XDECREF(self->loss_arr);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PULSE_INLINE PyObject*
Pulse_append(PulseObject* self, PyObject* const* args, Py_ssize_t nargs)
{
    Py_ssize_t i = self->cursor;
    self->cursor++;

    uint8_t* acc_ptr  = self->acc_ptr;
    uint8_t* loss_ptr = self->loss_ptr;

    double acc_d  = PyFloat_AS_DOUBLE(args[0]);
    double loss_d = PyFloat_AS_DOUBLE(args[1]);

    acc_ptr[i]  = (uint8_t)PULSE_QUANTIZE_10(acc_d);
    loss_ptr[i] = (uint8_t)PULSE_QUANTIZE_10(loss_d);

    Py_RETURN_NONE;
}

static PyObject*
Pulse_size(PulseObject* self, PyObject* Py_UNUSED(ignored))
{
    return PyLong_FromSsize_t(self->cursor);
}

static PyObject*
Pulse_arrays(PulseObject* self, PyObject* Py_UNUSED(ignored))
{
    Py_INCREF(self->acc_arr);
    Py_INCREF(self->loss_arr);
    return PyTuple_Pack(2, self->acc_arr, self->loss_arr);
}

static PULSE_INLINE int
pulse_write_all(int fd, const void* buf, size_t n)
{
    const uint8_t* p = (const uint8_t*)buf;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w > 0) {
            p += (size_t)w;
            n -= (size_t)w;
            continue;
        }
        if (w < 0 && errno == EINTR) {
            continue;
        }
        return -1;
    }
    return 0;
}

static PyObject*
Pulse_flush(PulseObject* self, PyObject* args)
{
    const char* path = NULL;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        return NULL;
    }

    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
        return NULL;
    }

    PulseFileHeader h;
    h.magic = PULSE_MAGIC;
    h.version = PULSE_VERSION;
    h.nrecords = (uint64_t)self->cursor;

    int rc = 0;
    int saved_errno = 0;

    Py_BEGIN_ALLOW_THREADS

    if (pulse_write_all(fd, &h, sizeof(h)) != 0) rc = -1;


    if (rc == 0 && self->cursor > 0) {
        size_t n = (size_t)self->cursor;

        if (pulse_write_all(fd, self->acc_ptr, n) != 0) rc = -1;
        if (rc == 0 && pulse_write_all(fd, self->loss_ptr, n) != 0) rc = -1;
    }

    saved_errno = errno;
    close(fd);
    errno = saved_errno;

    Py_END_ALLOW_THREADS

    if (rc != 0) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef Pulse_methods[] = {
    {"append", (PyCFunction)Pulse_append, METH_FASTCALL, "append(accuracy:float, loss:float) -> None"},
    {"size",   (PyCFunction)Pulse_size,   METH_NOARGS,   "size() -> int"},
    {"arrays", (PyCFunction)Pulse_arrays, METH_NOARGS,   "arrays() -> (acc_u8, loss_u8)"},
    {"flush",  (PyCFunction)Pulse_flush,  METH_VARARGS,  "flush(path:str) -> None"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject PulseType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pulse_fast_ext.Pulse",
    .tp_basicsize = sizeof(PulseObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Pulse_init,
    .tp_dealloc = (destructor)Pulse_dealloc,
    .tp_methods = Pulse_methods,
};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pulse_fast_ext",
    .m_doc = "Fast Pulse logger implemented as a CPython extension using FASTCALL and NumPy buffers.",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_pulse_fast_ext(void)
{
    import_array();

    if (PyType_Ready(&PulseType) < 0) return NULL;

    PyObject* m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    Py_INCREF(&PulseType);
    if (PyModule_AddObject(m, "Pulse", (PyObject*)&PulseType) < 0) {
        Py_DECREF(&PulseType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
