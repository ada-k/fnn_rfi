# Dockerfile to build AOFlagger and its dependencies in an isolated container.
# Usage: docker build -t aoflagger:local .

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget ca-certificates pkg-config \
    python3 python3-dev python3-pip python3-numpy \
    libfftw3-dev libboost-all-dev libcfitsio-dev libhdf5-dev \
    libeigen3-dev libgtkmm-4.0-dev libsigc++-2.0-dev libtool \
    automake autotools-dev autoconf flex bison libncurses5-dev \
    libreadline-dev libbz2-dev libz-dev unzip procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Build casacore from source (required by AOFlagger)
RUN git clone --depth 1 https://github.com/casacore/casacore.git casacore-src \
 && mkdir -p casacore-build && cd casacore-build \
 && cmake ../casacore-src -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_SHARED_LIBS=ON \
 && cmake --build . --parallel $(nproc) \
 && cmake --install .

# Clone and build AOFlagger
RUN git clone --depth 1 https://gitlab.astron.nl/AOFlagger/AOFlagger.git aoflagger-src \
 && mkdir -p aoflagger-build && cd aoflagger-build \
 && cmake ../aoflagger-src -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON \
 && cmake --build . --parallel $(nproc) \
 && cmake --install .

ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /data

ENTRYPOINT ["/bin/bash"]
