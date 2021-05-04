import cv2
import numpy as np

image = cv2.imread('test_image.jpeg')
cv2.imshow("result", image)
cv2.waitKey(0)
