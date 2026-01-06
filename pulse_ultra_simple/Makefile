.PHONY: build install run clean precompile

build:
	python -m pip install -e .
	$(MAKE) precompile

precompile:
	python -m src.compiler

c:
	gcc \
	  -O3 \
	  -ffast-math \
	  -funroll-loops \
	  -fopt-info-vec-optimized \
	  -march=native \
	  -std=c11 \
	  -Wall \
	  -Wextra \
	  ./examples/c/pure_c.c \
	  -o ./examples/c/pure_c
	echo "run ./examples/c/pure_c"

run:
	python src/pulse_ultra_simple/cli.py

clean:
	rm -rf build/ dist/ *.egg-info pulse_log.bin *.so *.dylib
	find . -name "__pycache__" -exec rm -rf {} +
