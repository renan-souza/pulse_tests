#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cmath>

namespace py = pybind11;

class Pulse final {
public:
    explicit Pulse(size_t capacity)
        : cursor_(0),
          epoch_(py::array_t<uint8_t>(capacity)),
          acc_q_(py::array_t<uint8_t>(capacity)),
          loss_q_(py::array_t<uint8_t>(capacity)),
          epoch_ptr_(static_cast<uint8_t*>(epoch_.mutable_data())),
          acc_ptr_(static_cast<uint8_t*>(acc_q_.mutable_data())),
          loss_ptr_(static_cast<uint8_t*>(loss_q_.mutable_data()))
    {}

    inline void append(int32_t epoch, float accuracy, float loss) noexcept {
        const size_t i = cursor_++;

        epoch_ptr_[i] = static_cast<uint8_t>(epoch);
        acc_ptr_[i] = static_cast<uint8_t>(std::fmaf(accuracy, 10.0f, 0.5f));
        loss_ptr_[i] = static_cast<uint8_t>(std::fmaf(loss, 10.0f, 0.5f));
    }

    size_t size() const noexcept { return cursor_; }

    py::tuple arrays() const {
        return py::make_tuple(epoch_, acc_q_, loss_q_);
    }

private:
    size_t cursor_;
    py::array_t<uint8_t> epoch_;
    py::array_t<uint8_t> acc_q_;
    py::array_t<uint8_t> loss_q_;
    uint8_t* epoch_ptr_;
    uint8_t* acc_ptr_;
    uint8_t* loss_ptr_;
};

PYBIND11_MODULE(pulse_fast_ext, m) {
    py::class_<Pulse>(m, "Pulse")
        .def(py::init<size_t>())
        .def("append", &Pulse::append)
        .def("size", &Pulse::size)
        .def("arrays", &Pulse::arrays);
}
