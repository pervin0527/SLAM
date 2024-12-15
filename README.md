# SLAM

## Install

```bash

## docker 이미지 생성
## docker build -t {image_name} {Dockerfile 경로}
docker build -t custom_slam_image . ## .는 현재 디렉토리에 dockerfile이 있음을 뜻함.

## container 실행
docker run -it \
    --name slam \
    --privileged \
    -e DISPLAY=host.docker.internal:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -p 8877:22 \
    custom_slam_iamge
```

## Run

``` bash
xhost +local:docker
