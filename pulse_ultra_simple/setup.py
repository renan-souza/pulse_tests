from setuptools import setup, Extension
import numpy as np
ext_modules = [
    Extension(
        "pulse_fast_ext",
        ["src/pulse_fast_ext.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",
            "-ffast-math",
            "-funroll-loops",
            "-fopt-info-vec-optimized",
        ],
    )
]


setup(
    name="pulse_ultra_simple",
    version="0.0.1",
    packages=["pulse"],
    ext_modules=ext_modules,
)
