FROM fair_mot

WORKDIR /app

COPY . .


#RUN git clone https://github.com/Media-Smart/vedadet.git /vedadet
#RUN cd /vedadet && vedadet_root=${PWD} && pip install -r requirements/build.txt && pip install -v -e .
