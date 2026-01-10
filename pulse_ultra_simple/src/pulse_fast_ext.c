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

// --- FILE FORMAT PROTOCOL ---
#define PULSE_MAGIC 0x31534C50u
#define PULSE_VERSION 4u

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;           // 0x31534C50
    uint32_t version;         // 4
    uint64_t nrecords;        // Number of rows written

    // Calibration Metadata (Optional, set to 0 if unused)
    double   base_wall_time;  // Start t0 (Unix Timestamp)
    uint64_t base_tsc;        // Start RDTSC value
    double   tps;             // Ticks Per Second (CPU Frequency)

    // Mode Flags
    uint8_t  has_time_column; // 1 = TS column included, 0 = Acc/Loss only
    uint8_t  _pad[7];         // Padding for 64-bit alignment
} PulseFileHeader;
#pragma pack(pop)

typedef struct {
    PyObject_HEAD
    Py_ssize_t capacity;
    Py_ssize_t cursor;
    int has_time;             // Flag: Is TS enabled?

    // Python Objects (for ref counting)
    PyObject* acc_arr;
    PyObject* loss_arr;
    PyObject* ts_arr;

    // Raw Pointers (for fast C access)
    uint8_t* acc_ptr;
    uint8_t* loss_ptr;
    uint64_t* ts_ptr;
} PulseObject;

// --- INITIALIZATION ---
static int Pulse_init(PulseObject* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {"capacity", "enable_time", NULL};
    Py_ssize_t cap = 0;
    int enable_time = 0; // Default: False (Production Mode)

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "n|p", kwlist, &cap, &enable_time)) {
        return -1;
    }

    if (cap <= 0) {
        PyErr_SetString(PyExc_ValueError, "Capacity must be > 0");
        return -1;
    }

    self->capacity = cap;
    self->cursor = 0;
    self->has_time = enable_time;

    npy_intp dims[1] = {(npy_intp)cap};

    // 1. Allocate Core Buffers (Acc, Loss)
    self->acc_arr  = (PyObject*)PyArray_SimpleNew(1, dims, NPY_UINT8);
    self->loss_arr = (PyObject*)PyArray_SimpleNew(1, dims, NPY_UINT8);

    // 2. Conditionally Allocate Timestamp Buffer
    if (self->has_time) {
        self->ts_arr = (PyObject*)PyArray_SimpleNew(1, dims, NPY_UINT64);
    } else {
        self->ts_arr = NULL;
    }

    // Check for allocation failures
    if (!self->acc_arr || !self->loss_arr || (self->has_time && !self->ts_arr)) {
        Py_XDECREF(self->acc_arr);
        Py_XDECREF(self->loss_arr);
        Py_XDECREF(self->ts_arr);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate NumPy arrays");
        return -1;
    }

    // 3. Cache Pointers
    self->acc_ptr  = (uint8_t*)PyArray_DATA((PyArrayObject*)self->acc_arr);
    self->loss_ptr = (uint8_t*)PyArray_DATA((PyArrayObject*)self->loss_arr);

    if (self->has_time) {
        self->ts_ptr = (uint64_t*)PyArray_DATA((PyArrayObject*)self->ts_arr);
    } else {
        self->ts_ptr = NULL;
    }

    return 0;
}

static void Pulse_dealloc(PulseObject* self)
{
    Py_XDECREF(self->acc_arr);
    Py_XDECREF(self->loss_arr);
    Py_XDECREF(self->ts_arr); // Safe even if NULL
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// --- ARRAY EXPOSURE ---
static PyObject* Pulse_arrays(PulseObject* self, PyObject* Py_UNUSED(ignored))
{
    // Must set Writeable flags because Numba requires it
    PyArray_ENABLEFLAGS((PyArrayObject*)self->acc_arr, NPY_ARRAY_WRITEABLE);
    PyArray_ENABLEFLAGS((PyArrayObject*)self->loss_arr, NPY_ARRAY_WRITEABLE);
    Py_INCREF(self->acc_arr);
    Py_INCREF(self->loss_arr);

    if (self->has_time) {
        PyArray_ENABLEFLAGS((PyArrayObject*)self->ts_arr, NPY_ARRAY_WRITEABLE);
        Py_INCREF(self->ts_arr);
        // Return (Acc, Loss, TS)
        return PyTuple_Pack(3, self->acc_arr, self->loss_arr, self->ts_arr);
    } else {
        // Return (Acc, Loss)
        return PyTuple_Pack(2, self->acc_arr, self->loss_arr);
    }
}

// --- FLUSHING ---
static PULSE_INLINE int pulse_write_all(int fd, const void* buf, size_t n)
{
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

static PyObject* Pulse_flush(PulseObject* self, PyObject* args)
{
    const char* path = NULL;
    Py_ssize_t n_to_write = -1;

    // Optional Metadata Arguments
    double base_time = 0.0;
    unsigned long long base_tsc = 0;
    double tps = 0.0;

    // Parse: path, optional n, optional metadata
    if (!PyArg_ParseTuple(args, "s|ndKd", &path, &n_to_write, &base_time, &base_tsc, &tps)) {
        return NULL;
    }

    // Determine how many records to write
    size_t n;
    if (n_to_write < 0) {
        n = (size_t)self->capacity; // Default: Dump everything
    } else {
        if (n_to_write > self->capacity) n_to_write = self->capacity;
        n = (size_t)n_to_write;
    }

    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
        return NULL;
    }

    // Prepare Header
    PulseFileHeader h;
    memset(&h, 0, sizeof(h)); // Zero out padding
    h.magic = PULSE_MAGIC;
    h.version = PULSE_VERSION;
    h.nrecords = (uint64_t)n;
    h.base_wall_time = base_time;
    h.base_tsc = (uint64_t)base_tsc;
    h.tps = tps;
    h.has_time_column = (uint8_t)self->has_time;

    int rc = 0;
    Py_BEGIN_ALLOW_THREADS
    // 1. Write Header
    if (pulse_write_all(fd, &h, sizeof(h)) != 0) rc = -1;

    // 2. Write Data Bodies
    if (rc == 0 && n > 0) {
        // Always write Acc and Loss
        if (pulse_write_all(fd, self->acc_ptr, n) != 0) rc = -1;
        if (pulse_write_all(fd, self->loss_ptr, n) != 0) rc = -1;

        // Conditionally write Timestamp column
        if (self->has_time && self->ts_ptr) {
            if (pulse_write_all(fd, self->ts_ptr, n * sizeof(uint64_t)) != 0) rc = -1;
        }
    }
    close(fd);
    Py_END_ALLOW_THREADS

    if (rc != 0) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* Pulse_size(PulseObject* self, PyObject* Py_UNUSED(ignored))
{
    return PyLong_FromSsize_t(self->cursor);
}

// --- MODULE SETUP ---
static PyMethodDef Pulse_methods[] = {
    {"size",   (PyCFunction)Pulse_size,   METH_NOARGS,   "Get current cursor position"},
    {"arrays", (PyCFunction)Pulse_arrays, METH_NOARGS,   "Get tuple of numpy views (2 or 3)"},
    {"flush",  (PyCFunction)Pulse_flush,  METH_VARARGS,  "Write to disk with optional metadata"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject PulseType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pulse_fast_ext.Pulse",
    .tp_basicsize = sizeof(PulseObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Pulse_init,
    .tp_dealloc = (destructor)Pulse_dealloc,
    .tp_methods = Pulse_methods,
};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pulse_fast_ext",
    .m_doc = "Fast Pulse logger with Dynamic Time Column Strategy",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_pulse_fast_ext(void) {
    import_array();
    if (PyType_Ready(&PulseType) < 0) return NULL;
    PyObject* m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    Py_INCREF(&PulseType);
    PyModule_AddObject(m, "Pulse", (PyObject*)&PulseType);
    return m;
}