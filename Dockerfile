FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

RUN python -m pip install -U pip setuptools wheel pybind11 numpy pandas

COPY pulse_ultra_simple /work/pulse_ultra_simple
WORKDIR /work/pulse_ultra_simple

RUN python -m pip install -U pip setuptools wheel pybind11
RUN python -m pip install .
RUN python examples/run.py
RUN test -f pulse_simple.bin && test -s pulse_simple.bin
RUN python examples/read_into_pandas.py
