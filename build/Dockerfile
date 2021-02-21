FROM nvidia/cuda:11.0-devel-ubuntu18.04

ARG python_version=3.7
RUN apt-get update

#&& apt-get install -y --no-install-recommends \
#        cuda-nvml-dev-$CUDA_PKG_VERSION \
#        cuda-command-line-tools-$CUDA_PKG_VERSION \
#cuda-libraries-dev-$CUDA_PKG_VERSION \
#        cuda-minimal-build-$CUDA_PKG_VERSION \
#        libnccl-dev=$NCCL_VERSION-1+cuda10.2 \
#libcublas-dev=10.2.2.89-1 \
#&& \/ /var/lib/apt/lists/*

#ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    htop \
 && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

 # Install Miniconda
RUN mkdir -p $CONDA_DIR && \
    apt-get update && \
    apt-get install -y wget git vim htop zip libhdf5-dev g++ graphviz libgtk2.0-dev libgl1-mesa-glx \
    openmpi-bin nano cmake libopenblas-dev liblapack-dev libx11-dev && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh


# CT: 4/17


# Install Miniconda
# RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \

RUN conda config --set remote_read_timeout_secs 100000.0 && \
    conda init && \
    conda update -n base -c defaults conda && \
    conda env create -f project/environment.yml && \
    conda clean -yt && \
    echo "conda activate tracker" >> /root/.bashrc

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/user/miniconda/bin/conda clean -ya


# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install Torchnet, a high-level framework for PyTorch
#RUN pip install torchnet==0.0.4

# Install Requests, a Python library for making HTTP requests
#RUN conda install -y requests=2.19.1 \
# && conda clean -ya

# Install Graphviz
#RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
# && conda clean -ya

## Install OpenCV3 Python bindings
#RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
#    libgtk2.0-0 \
#    libcanberra-gtk-module \
# && sudo rm -rf /var/lib/apt/lists/*
#RUN conda install -y -c menpo opencv3=3.1.0 \
# && conda clean -ya

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LIBRARY_PATH
ENV CUDA_HOME /usr/local/cuda

RUN echo '#!/bin/bash\n \
echo "export CONDA_DIR=$CONDA_DIR" >> /etc/profile\n \
echo "export PATH=$CONDA_DIR/bin:$PATH" >> /etc/profile\n \
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" >> /etc/profile\n \
echo "export LIBRARY_PATH=/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LIBRARY_PATH" >> /etc/profile\n \
echo "export CUDA_HOME=/usr/local/cuda" >> /etc/profile\n \
/usr/sbin/sshd -D' \
>> /bin/start.sh

COPY . /torchok

EXPOSE 8888 6006 22
ENTRYPOINT bash /bin/start.sh
