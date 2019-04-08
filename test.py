# !/usr/bin/env python2.7
import cv2
import numpy as np

# load image
prev = cv2.imread('0000000000.png')
next = cv2.imread('0000000001.png')

# change RGB to gray
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

# calculate optical flow
flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# calculate mat
w = int(prev.shape[1])
h = int(prev.shape[0])
y_coords, x_coords = np.mgrid[0:h, 0:w]
coords = np.float32(np.dstack([x_coords, y_coords]))
pixel_map = coords + flow
print(pixel_map.shape)
new_frame = cv2.remap(prev, pixel_map, None, cv2.INTER_LINEAR)
new_frame = new_frame[:,371:871]

cv2.imwrite('new_frame.png', new_frame)
