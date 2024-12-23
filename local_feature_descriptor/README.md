# local feature extractor + descriptor + matching
- 매칭이 잘된 경우 -> 수많은 직선들이 그려지게 된다.  
- 대각선이 포함되면 잘못된 매칭이 발생한 것.

# SetUp

## 1.dataset
```bash
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip
# wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip
```

## 2.Build

### CMakeLists.txt
```bash
cmake_minimum_required(VERSION 3.10)
project("orb feature matching" LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "Found OpenCV library: " ${OpenCV_INCLUDE_DIRS})
    include_directories(${OpenCV_INCLUDE_DIRS})
endif (OpenCV_FOUND)

add_executable(orb_feature_matching examples/orb_feature_matching.cpp)
target_link_libraries(orb_feature_matching ${OpenCV_LIBS})
```

- CMakeLists.txt 파일에 ```add_executable()```에 명시하는 cpp 파일 경로가 일치해야한다.

### build

```bash
### for docker
FROM slam:latest

RUN cd fastcampus_slam_codes/3_2 &&\
    mkdir build && cd build && \
    cmake -GNinja ../ && \
    ninja
```

```bash
## for local build
mkdir build && cd build
cmake ..
make -j
./orb_feature_matching /home/pervinco/Datasets/KITTI/dataset/sequences/00/image_0 /home/pervinco/Datasets/KITTI/dataset/sequences/00/image_1 100
```

## code

### 1.실행 인자

```bash
./orb_feature_matching /home/pervinco/Datasets/KITTI/dataset/sequences/00/image_0 /home/pervinco/Datasets/KITTI/dataset/sequences/00/image_1 100
```

- argv[0]: 실행 파일 이름 (./orb_feature_matching)
- argv[1]: /home/pervinco/Datasets/KITTI/sequences/00/image_0 (왼쪽 이미지 경로)
- argv[2]: /home/pervinco/Datasets/KITTI/sequences/00/image_1 (오른쪽 이미지 경로)
- argv[3]: 100 (프레임 수)

### 2.문자열 벡터

```cpp
    std::vector<std::string> left_image_filenames, right_image_filenames;
    left_image_filenames.reserve(5000);
    right_image_filenames.reserve(5000);
```
- 실행 인자로 전달한 경로에 있는 파일들을 문자열 벡터에 저장.
- ```reserve```는 벡터의 용량을 미리 할당하는 함수.
- 벡터는 동적 배열로, 원소를 추가할 때 메모리를 점진적으로 확장한다. 메모리 확장이 필요한 경우, 기존 데이터의 복사와 추가 할당이 일어나므로 비용이 크기 때문에 미리 할당한다.


```cpp
for (const auto &entry : std::filesystem::directory_iterator(left_image_path)) {
    left_image_filenames.push_back(entry.path());
}
```

- ```left_image_path``` 디렉토리 내의 파일 경로들을 순회
- 발견된 파일 경로들을 left_image_filenames 벡터에 저장.