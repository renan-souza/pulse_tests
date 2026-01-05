#include <pybind11/pybind11.h>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>

namespace py = pybind11;

#pragma pack(push, 1)
struct LogRecord {
    uint8_t epoch;
    uint8_t accuracy; // 1 byte: Store (acc * 100)
    uint8_t loss;     // 1 byte: Store (loss * 10)
};
#pragma pack(pop)

class Pulse {
public:
    Pulse(size_t capacity) : capacity_(capacity), cursor_(0) {
        if (posix_memalign((void**)&data_, 64, capacity * sizeof(LogRecord)) != 0) {
            throw std::bad_alloc();
        }
    }

    ~Pulse() { free(data_); }

    // Logic: Multiply by 100/10 to fit in uint8
    inline void append(int32_t epoch, float accuracy, float loss) {
        LogRecord& r = data_[cursor_];
        
        // Fast rounding and quantization
        r.accuracy = static_cast<uint8_t>(accuracy * 100.0f + 0.5f);
        r.loss = static_cast<uint8_t>(loss * 10.0f + 0.5f);
        r.epoch = static_cast<uint8_t>(epoch % 256);
        
        cursor_++;
    }

    size_t size() const { return cursor_; }

private:
    LogRecord* data_;
    size_t capacity_;
    size_t cursor_;
};

PYBIND11_MODULE(pulse_fast_ext, m) {
    py::class_<Pulse>(m, "Pulse")
        .def(py::init<size_t>())
        .def("append", &Pulse::append)
        .def("size", &Pulse::size);
}