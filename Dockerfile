FROM  nvidia/cuda:10.0-base-ubuntu18.04

RUN apt-get update

RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN apt install -y python3-pip

RUN pip3 install --upgrade pip

RUN apt install -y git

RUN apt install -y python-opencv

RUN apt-get clean

RUN pip3 install numpy==1.18.0

RUN pip install flask  requests
    
RUN apt-get install -y gunicorn3

RUN pip3 install opencv-python

ADD . mask_scanner/

RUN mkdir -p mask_scanner/inference/output

COPY weights/* mask_scanner/weights/

RUN pip3 install -U -r mask_scanner/requirements.txt

WORKDIR 'mask_scanner'

CMD exec gunicorn3 --timeout 600 -b 0.0.0.0:80 --chdir /mask_scanner zero_app:app
