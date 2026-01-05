#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // Required for py::array_t
#include <cstdint>
#include <cstdlib>

namespace py = pybind11;

#pragma pack(push, 1)
struct LogRecord {
    uint8_t epoch;
    uint8_t accuracy; 
    uint8_t loss;
};
#pragma pack(pop)

class Pulse {
public:
    Pulse(size_t capacity) : capacity_(capacity), cursor_(0) {
        // macOS Sequoia: aligned_alloc is standard for cache-line alignment
        data_ = (LogRecord*)aligned_alloc(64, capacity * sizeof(LogRecord));
        if (!data_) throw std::bad_alloc();
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


    void push_batch(py::array_t<int32_t> e, py::array_t<float> a, py::array_t<float> l) {
        auto re = e.unchecked<1>();
        auto ra = a.unchecked<1>();
        auto rl = l.unchecked<1>();
        size_t n = re.shape(0);

        if (cursor_ + n > capacity_) throw std::runtime_error("Buffer overflow");

        // Use a local pointer for the loop to facilitate compiler vectorization
        LogRecord* out = &data_[cursor_];
        for (size_t i = 0; i < n; ++i) {
            out[i].epoch = static_cast<uint8_t>(re(i) % 256);
            out[i].accuracy = static_cast<uint8_t>(ra(i) * 10.0f + 0.5f);
            out[i].loss = static_cast<uint8_t>(rl(i) * 10.0f + 0.5f);
        }
        cursor_ += n;
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
        .def("push_batch", &Pulse::push_batch)
        .def("size", &Pulse::size);
}