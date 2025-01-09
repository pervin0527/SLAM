cmake_minimum_required(VERSION 3.10) ## CMake 최소 버전 명시
project("Introduction to point cloud processing" LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17) ## CPP 17을 표준으로 설정

find_package(PCL REQUIRED) ## PCL 라이브러리가 있는지 탐색
if (PCL_FOUND)
    message(STATUS "Found PCL library: " ${PCL_INCLUDE_DIRS})
    include_directories(${PCL_INCLUDE_DIRS}) ## 라이브러리 경로를 include
endif (PCL_FOUND)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(visualization examples/visualization.cpp) ## 실행 코드 1
target_link_libraries(visualization ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES})

add_executable(visualization_kitti examples/visualization_kitti.cpp) ## 실행 코드 2
target_link_libraries(visualization_kitti ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES})