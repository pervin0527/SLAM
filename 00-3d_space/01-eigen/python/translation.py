import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 원점 정의
origin = np.array([0, 0, 0])

# Translation 벡터 정의
translation_vector = np.array([7, 7, 7])

# 이동된 점 계산
translated_point = origin + translation_vector

# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 원점과 이동된 점 표시
ax.scatter(*origin, color='red', s=100, label='Origin')  # 원점
ax.scatter(*translated_point, color='blue', s=100, label='Translated Point')  # 이동된 점

# 이동 벡터를 선으로 표시
ax.plot(
    [origin[0], translated_point[0]], 
    [origin[1], translated_point[1]], 
    [origin[2], translated_point[2]], 
    color='green', linestyle='dashed', label='Translation Vector'
)

# 축 설정
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_xlim([0, 10]) ## 0 ~ 9까지 눈금 표현
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])

# 제목과 범례 추가
ax.set_title('3D Point Translation')
ax.legend()

# 플롯 표시
plt.show()
