import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 큐브의 꼭짓점 정의
def cube_vertices(center, size):
    c = np.array(center)
    s = size / 2
    vertices = [
        c + [-s, -s, -s], c + [s, -s, -s], c + [s, s, -s], c + [-s, s, -s],  # 아래 면
        c + [-s, -s, s], c + [s, -s, s], c + [s, s, s], c + [-s, s, s]       # 위 면
    ]
    return np.array(vertices)

# 큐브의 면 정의
def cube_faces(vertices):
    return [
        [vertices[j] for j in [0, 1, 2, 3]],  # 아래 면
        [vertices[j] for j in [4, 5, 6, 7]],  # 위 면
        [vertices[j] for j in [0, 1, 5, 4]],  # 앞면
        [vertices[j] for j in [2, 3, 7, 6]],  # 뒷면
        [vertices[j] for j in [1, 2, 6, 5]],  # 오른쪽 면
        [vertices[j] for j in [0, 3, 7, 4]],  # 왼쪽 면
    ]

# 애니메이션 초기화 함수
def init():
    ax.clear()
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_box_aspect([1, 1, 1])  # 1:1:1 비율 유지

# 애니메이션 업데이트 함수
def update(frame):
    init()
    translated_center = [frame * 0.1, 0, 0]  # x축 방향으로 이동
    vertices = cube_vertices(translated_center, size=2)
    faces = cube_faces(vertices)
    ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', edgecolors='r', alpha=0.6))

# Figure 및 3D Axes 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 애니메이션 생성
ani = FuncAnimation(fig, update, frames=100, interval=50, init_func=init)

plt.show()
