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
    uint16_t accuracy; // 2 bytes: Store (acc * 1000)
    uint16_t loss;     // 2 bytes: Store (loss * 1000)
    char filename[32]; // 32 bytes
};
#pragma pack(pop)

class Pulse {
public:
    Pulse(size_t capacity) {
        data_.reserve(capacity);
    }


    // Logic: Multiply by 1000 to keep 3 decimal points
    inline void append(int32_t epoch, float accuracy, float loss, const std::string& filename) {
        LogRecord r;

        
        r.accuracy = static_cast<uint16_t>(accuracy * 1000.0f + 0.5f);
        r.loss = static_cast<uint16_t>(loss * 1000.0f + 0.5f);
        r.epoch = static_cast<uint8_t>(epoch % 256);
        
        std::strncpy(r.filename, filename.c_str(), 31);
        r.filename[31] = '\0';
        
        data_.push_back(r);
    }

    size_t size() const {
        return data_.size();
    }


private:
    std::vector<LogRecord> data_;    
};

PYBIND11_MODULE(pulse_fast_ext, m) {
    py::class_<Pulse>(m, "Pulse")
        .def(py::init<size_t>())
        .def("append", &Pulse::append)
        .def("size", &Pulse::size);
}