# Approximately 10 min to build

FROM nvidia/cuda:10.2-devel-ubuntu18.04
# Python
ARG python_version=3.7
ARG SSH_PASSWORD=password

# https://docs.docker.com/engine/examples/running_ssh_service/
# Last is SSH login fix. Otherwise user is kicked off after login
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir /var/run/sshd && echo "root:$SSH_PASSWORD" | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    echo "export VISIBLE=now" >> /etc/profile && \
    mkdir /root/.ssh && chmod 700 /root/.ssh && touch /root/.ssh/authorized_keys && \
    chmod 644 /root/.ssh/authorized_keys

ENV NOTVISIBLE "in users profile"
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

COPY ./environment.yml /src/environment.yml

# Install Data Science essential
RUN conda config --set remote_read_timeout_secs 100000.0 && \
    conda init && \
    conda update -n base -c defaults conda && \
    conda env create -f /src/environment.yml && \
    conda clean -yt && \
    echo "conda activate mot" >> /root/.bashrc

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LIBRARY_PATH
ENV CUDA_HOME /usr/local/cuda

# To access the container from the outer world
ENV SSH_PUBLIC_KEY ""

# To be able to add SSH key on docker run --env ... and to get important environment variables in SSH's bash
# writing env variables to /etc/profile as mentioned here:
# https://docs.docker.com/engine/examples/running_ssh_service/#environment-variables
RUN echo '#!/bin/bash\n \
echo $SSH_PUBLIC_KEY >> /root/.ssh/authorized_keys\n \
echo "export CONDA_DIR=$CONDA_DIR" >> /etc/profile\n \
echo "export PATH=$CONDA_DIR/bin:$PATH" >> /etc/profile\n \
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" >> /etc/profile\n \
echo "export LIBRARY_PATH=/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LIBRARY_PATH" >> /etc/profile\n \
echo "export CUDA_HOME=/usr/local/cuda" >> /etc/profile\n \
echo "alias juplabstart=\"nohup jupyter lab --ip 0.0.0.0 --allow-root > jup.log 2>&1 &\"" >> /etc/profile\n \
echo "alias jupnotestart=\"nohup jupyter notebook --ip 0.0.0.0 --allow-root > jup.log 2>&1 &\"" >> /etc/profile\n \
echo "alias jupkill=\"kill -9 \$(pgrep -f jupyter)\"" >> /etc/profile\n \
echo "alias tbkill=\"kill -9 \$(pgrep -f tensorboard)\"" >> /etc/profile\n \
/usr/sbin/sshd -D' \
>> /bin/start.sh

RUN echo '#!/bin/bash\n \
nohup tensorboard --bind_all --logdir=$1 > tb.log 2>&1 & echo "see tb.log for address"' \
>> /bin/tbstart.sh && chmod +x /bin/tbstart.sh

COPY . /src

EXPOSE 8888 6006 22
ENTRYPOINT bash /bin/start.sh
