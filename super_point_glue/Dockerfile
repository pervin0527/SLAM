# CUDA 11.7, TensorRT 8.4.1
FROM nvcr.io/nvidia/tensorrt:22.07-py3 

ARG DEBIAN_FRONTEND=noninteractive

# 필수 패키지 설치 및 OpenCV 빌드
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libeigen3-dev wget build-essential cmake git \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev unzip && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.4.0.zip && \
    unzip opencv.zip && \
    cd opencv-4.4.0 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../../ && \
    rm -rf opencv.zip opencv-4.4.0

RUN apt-get install libyaml-cpp-dev -y
    
RUN cd / && \
    git clone https://github.com/changh95/fastcampus_slam_codes.git && \
    cd fastcampus_slam_codes/3_3 &&\
    mkdir build && cd build && cmake .. && make -j4
