# Introduction to point cloud processing

- Load a lidar point cloud datum from KITTI dataset
- Use basic PCL data structures
- Visualize the point cloud

---

# How to build & run

Requirement: PCL

## Local build

```
mkdir build && cd build
cmake ..
make -j
./visualization
./visualization /data/sequences/00/velodyne
```

## Docker build 

Requires base build

```
docker build . -t slam:4_2
docker run -it --env DISPLAY=$DISPLAY -v /kitti:/data -v /tmp/.X11-unix/:/tmp/.X11-unix:ro slam:4_2

# Inside docker container
cd fastcampus_slam_codes/4_2
./build/visualization
./build/visualization_kitti /home/pervinco/Datasets/KITTI/dataset/sequences/00/velodyne/
```

---

# Output

## Two point clouds visualization

![](./output.gif)

## KITTI dataset visualization

![](./output_kitti.gif)