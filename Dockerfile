FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install the packages that shouldn't affect the build process as all
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    git \
    cmake \
    build-essential \
    wget \
    curl && \ 
    apt-get clean

# Install Miniconda
WORKDIR /opt
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    rm miniconda.sh

# Set up the conda environment
ENV PATH="/opt/miniconda/bin:$PATH"
RUN conda init && \
    conda config --set auto_activate_base false

# Ensure conda is available in non-interactive shells
RUN echo "source /opt/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc

# Conda install python version which works
RUN conda install python=3.11

# Install Fortran
RUN apt install -y gfortran-11
RUN ln -s /usr/bin/gfortran-11 /usr/bin/gfortran

RUN pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
RUN ln -s /opt/miniconda/lib/python3.11/site-packages/torch /usr/local/libtorch

# Install FTorch
WORKDIR /opt
RUN git clone https://github.com/Cambridge-ICCS/FTorch.git /opt/FTorch

WORKDIR /opt/FTorch/src/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DUSE_CUDA=ON \
    -DCUDA_BIN_PATH=/usr/local/cuda/bin \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCUDA_LIB_DIR=/usr/local/cuda/lib64 \
    -DCMAKE_PREFIX_PATH=/usr/local/libtorch \
    -DCMAKE_INSTALL_PREFIX=/usr/local/ftorch

RUN make && make install

# Add To LD_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/ftorch/lib:$LD_LIBRARY_PATH 
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

ENV LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH
ENV Torch_DIR=/usr/local/libtorch/share/cmake/Torch

WORKDIR /root