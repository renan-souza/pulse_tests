#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace py = pybind11;

class Pulse {
public:
    Pulse(size_t capacity)
        : capacity_(capacity),
          cursor_(0),
          epochs_(),
          acc_(),
          loss_(),
          epochs_ptr_(nullptr),
          acc_ptr_(nullptr),
          loss_ptr_(nullptr)
    {
        epochs_ = py::array_t<int32_t>(capacity_);
        acc_ = py::array_t<float>(capacity_);
        loss_ = py::array_t<float>(capacity_);

        epochs_ptr_ = static_cast<int32_t*>(epochs_.mutable_data());
        acc_ptr_ = static_cast<float*>(acc_.mutable_data());
        loss_ptr_ = static_cast<float*>(loss_.mutable_data());
    }

    inline void append(int32_t epoch, float accuracy, float loss) {
        const size_t i = cursor_;
        if (i >= capacity_) {
            throw std::runtime_error("Buffer overflow");
        }
        epochs_ptr_[i] = epoch;
        acc_ptr_[i] = accuracy;
        loss_ptr_[i] = loss;
        cursor_ = i + 1;
    }

    size_t size() const { return cursor_; }

    py::tuple arrays() const {
        return py::make_tuple(epochs_, acc_, loss_);
    }

private:
    size_t capacity_;
    size_t cursor_;

    py::array_t<int32_t> epochs_;
    py::array_t<float> acc_;
    py::array_t<float> loss_;

    int32_t* epochs_ptr_;
    float* acc_ptr_;
    float* loss_ptr_;
};

PYBIND11_MODULE(pulse_fast_ext, m) {
    py::class_<Pulse>(m, "Pulse")
        .def(py::init<size_t>()) // Constructor
        .def("append", &Pulse::append) // Method to append data
        .def("size", &Pulse::size)     // Method to get current size
        .def("arrays", &Pulse::arrays); // Method to retrieve data as NumPy arrays
}
