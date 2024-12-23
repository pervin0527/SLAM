# SLAM

## Install

### docker 이미지 생성
```bash

## docker 이미지 생성
## docker build -t {image_name} {Dockerfile 경로}:{태그}
## docker run -it --name <컨테이너 이름> <이미지 이름>:<태그>
docker build -t slam_image:latest . ## .는 현재 디렉토리의 도커파일을 사용.
```

### X11 Server
```bash
## X11이 설치되지 않은 경우
sudo apt update
sudo apt install -y xorg x11-apps

sudo vi /etc/ssh/sshd_config

## 아래 항목들을 찾아 수정, 없으면 추가.
X11Forwarding yes
X11DisplayOffset 10
X11UseLocalhost yes

sudo service ssh restart
ssh -X username@server_ip
```

### docker container 생성.

```bash
echo $DISPLAY # DISPLAY 환경 변수 설정확인
export DISPLAY=:0 # 위에서 아무것도 출력되지 않은 경우 디스플레이 변수 설정.

xhost +local:docker 
docker run -it --name slam \
    --privileged \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/pervinco/:/home/pervinco/ \
    slam_image:latest

apt install libgtk2.0-dev libgtk-3-dev libqt5gui5 libqt5widgets5 libqt5core5a -y
```

## Run

``` bash
xhost +local:docker