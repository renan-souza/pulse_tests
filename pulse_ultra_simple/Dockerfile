FROM python:3.12-slim

# Install system build dependencies (gcc, g++, make)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies and compile the extension
RUN pip install --upgrade pip setuptools wheel pybind11 && \
    pip install .

CMD ["python3", "examples/run.py"]