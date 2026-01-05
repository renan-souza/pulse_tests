#!/usr/bin/env python3
from pathlib import Path

def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")

def main():
    root = Path("pulse_ultra_simple")
    if root.exists():
        for p in sorted(root.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            else:
                try:
                    p.rmdir()
                except OSError:
                    pass

    (root / "src").mkdir(parents=True)
    (root / "pulse").mkdir(parents=True)
    (root / "examples").mkdir(parents=True)

    write(root / "pyproject.toml", """
[build-system]
requires = ["setuptools>=68", "wheel", "pybind11>=2.10"]
build-backend = "setuptools.build_meta"
""")

    write(root / "setup.py", """
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "pulse_simple_ext",
        ["src/pulse_simple.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]

setup(
    name="pulse_ultra_simple",
    version="0.0.1",
    packages=["pulse"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
""")

    write(root / "pulse/__init__.py", """
from pulse_simple_ext import Pulse

__all__ = ["Pulse"]
""")

    write(root / "src/pulse_simple.cpp", r"""
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace py = pybind11;

#pragma pack(push, 1)
struct LogRecord {
    int32_t epoch;
    int32_t step;
    float loss;
};
#pragma pack(pop)

static_assert(sizeof(LogRecord) == 12);

class Pulse {
public:
    Pulse() = default;

    void reserve(size_t n) {
        buf_.reserve(n);
    }

    void append(int32_t epoch, int32_t step, float loss) {
        LogRecord r;
        r.epoch = epoch;
        r.step = step;
        r.loss = loss;
        buf_.push_back(r);
    }

    void flush(const std::string& path) {
        FILE* fp = std::fopen(path.c_str(), "wb");
        if (!fp) throw std::runtime_error("failed to open file");
        if (!buf_.empty()) {
            size_t n = std::fwrite(buf_.data(), sizeof(LogRecord), buf_.size(), fp);
            if (n != buf_.size()) throw std::runtime_error("short write");
        }
        std::fclose(fp);
    }

    size_t size() const {
        return buf_.size();
    }

private:
    std::vector<LogRecord> buf_;
};

PYBIND11_MODULE(pulse_simple_ext, m) {
    py::class_<Pulse>(m, "Pulse")
        .def(py::init<>())
        .def("reserve", &Pulse::reserve)
        .def("append", &Pulse::append)
        .def("flush", &Pulse::flush)
        .def("size", &Pulse::size);
}
""")

    write(root / "examples/run.py", """
import random
from pulse import Pulse

def main():
    p = Pulse()
    p.reserve(10_000)

    loss = 5.0
    for i in range(10_000):
        epoch = i // 1000
        loss *= 0.999 + random.random() * 0.0005
        p.append(epoch, i, float(loss))

    p.flush("pulse_simple.bin")
    print("records:", p.size())
    print("wrote pulse_simple.bin")

if __name__ == "__main__":
    main()
""")

    write(root / "examples/read_into_pandas.py", """
import numpy as np
import pandas as pd

DT = np.dtype([
    ("epoch", np.int32),
    ("step", np.int32),
    ("loss", np.float32),
])

def main():
    arr = np.fromfile("pulse_simple.bin", dtype=DT)
    df = pd.DataFrame(arr)
    print(df.head())
    print(df.tail())
    print("rows:", len(df))

if __name__ == "__main__":
    main()
""")

    write(root / "README.txt", """
Pulse ultra-simple (educational)

What this shows:
- fixed C struct
- append() -> push_back()
- flush() writes once
- zero background threads
- zero timing
- zero telemetry

Build:
  python3 -m pip install -U pip setuptools wheel pybind11
  python3 -m pip install .

Run:
  python3 examples/run.py
  python3 examples/read_into_pandas.py
""")

    print("Created pulse_ultra_simple")
    print("Next:")
    print("  cd pulse_ultra_simple")
    print("  python3 -m pip install -U pip setuptools wheel pybind11")
    print("  python3 -m pip install .")
    print("  python3 examples/run.py")
    print("  python3 examples/read_into_pandas.py")

if __name__ == "__main__":
    main()
