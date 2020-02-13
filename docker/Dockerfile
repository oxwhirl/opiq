FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER Tabish Rashid

RUN apt-get update
RUN apt-get install -y software-properties-common 
RUN apt-get install -y python3-pip
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip

RUN apt-get install -y vim screen htop

# Needed for some gym dependencies
RUN apt-get install -y --no-install-recommends zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb

RUN ln -sf /usr/lib/x86_64-linux-gnu/libboost_python-py35.so /usr/lib/x86_64-linux-gnu/libboost_python-py27.so.1.58.0

RUN pip install gym==0.12.1
#RUN pip install gym==0.10.5
RUN pip install pygame
RUN pip install numpy
#RUN pip install numpy==1.15.4
RUN pip install scipy pyyaml matplotlib
RUN pip install tensorboard-logger
RUN pip install scikit-image
RUN pip install pymongo
RUN pip install sacred==0.7.4
RUN pip install gym[atari]
RUN pip install opencv-python
RUN apt-get install -y zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev wget unzip
RUN pip install vizdoom
RUN pip install textworld

#RUN pip install torch==1.0.1.post2
RUN pip install torch==1.1.0

RUN pip install imageio
RUN pip install tensorboard_logger

RUN pip install gym-retro==0.6.0
RUN apt-get install -y git
RUN git clone https://github.com/MaxStrange/retrowrapper /repos/retrowrapper
RUN pip install -e /repos/retrowrapper

WORKDIR "/src"
