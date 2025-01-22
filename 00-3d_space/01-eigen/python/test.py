import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_rectangular_prism(center, size):
    x, y, z = center
    dx, dy, dz = size
    vertices = np.array([
        [x - dx / 2, y - dy / 2, z - dz / 2],
        [x + dx / 2, y - dy / 2, z - dz / 2],
        [x + dx / 2, y + dy / 2, z - dz / 2],
        [x - dx / 2, y + dy / 2, z - dz / 2],
        [x - dx / 2, y - dy / 2, z + dz / 2],
        [x + dx / 2, y - dy / 2, z + dz / 2],
        [x + dx / 2, y + dy / 2, z + dz / 2],
        [x - dx / 2, y + dy / 2, z + dz / 2],
    ])
    return vertices

def get_faces(vertices):
    return [
        [vertices[j] for j in [0, 1, 2, 3]],  # 아래면
        [vertices[j] for j in [4, 5, 6, 7]],  # 윗면
        [vertices[j] for j in [0, 3, 7, 4]],  # 왼쪽면
        [vertices[j] for j in [1, 2, 6, 5]],  # 오른쪽면
        [vertices[j] for j in [0, 1, 5, 4]],  # 앞면
        [vertices[j] for j in [3, 2, 6, 7]],  # 뒷면
    ]

def rotation_matrix_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

# 직육면체 생성
center = (0, 0, 0)
size = (2, 3, 1)
vertices = create_rectangular_prism(center, size)

# 회전 변환
theta = np.pi / 4
rotation_matrix = rotation_matrix_x(theta)
rotated_vertices = np.dot(vertices, rotation_matrix.T)

# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 원래 직육면체 그리기
faces = get_faces(vertices)
original_box = Poly3DCollection(faces, alpha=0.3, color='blue')
ax.add_collection3d(original_box)

# 회전된 직육면체 그리기
rotated_faces = get_faces(rotated_vertices)
rotated_box = Poly3DCollection(rotated_faces, alpha=0.3, color='red')
ax.add_collection3d(rotated_box)

# 범례를 직접 추가
ax.text2D(0.05, 0.95, "Original: Blue\nRotated: Red", transform=ax.transAxes)

# 축 설정
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# 제목 추가
ax.set_title('3D Rectangular Prism Rotation Around X-axis')

plt.show()
