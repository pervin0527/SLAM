import cv2
import numpy as np

img = np.zeros(300, 400, 3)
cv2.imshow("Test Image", img)
cv2.waitKey(0)
