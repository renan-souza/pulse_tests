#!/usr/bin/env bash
set -euo pipefail
docker build -t pulse-ultra-simple:py312 pulse_ultra_simple
docker run --rm -v "$(pwd)/pulse_ultra_simple:/work/pulse_ultra_simple" -it pulse-ultra-simple:py312 /bin/bash