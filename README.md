# SLAM

## Install

```bash

## docker 이미지 생성
## docker build -t {image_name} {Dockerfile 경로}:{태그}
docker build -t custom_slam_image:latest . ## .는 현재 디렉토리의 도커파일을 사용.

## container 실행
## docker run -it --name <컨테이너 이름> <이미지 이름>:<태그>
docker run -it --name slam --privileged -v /home/pervinco/:/home/pervinco/ custom_slam_image:latest
```

## Run

``` bash
xhost +local:docker