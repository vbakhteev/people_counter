FROM pytorch/pytorch:latest

WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apt install git -y
RUN git clone https://github.com/Media-Smart/vedadet.git /vedadet
#RUN cd /vedadet && vedadet_root=${PWD} && pip install -r requirements/build.txt && pip install -v -e .
