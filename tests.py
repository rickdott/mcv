import numpy as np
import cv2 as cv

arr = np.random.rand(56,2)
print(arr)
print(arr.shape)

arr2 = np.expand_dims(arr, 1)
print(arr2)
print(arr2.shape)

for item in arr2:
    print(item[0][0])
    print(item[0][1])
# img = cv.imread('resources\img_20_X.png')

# # xy = np.mgrid[155:284:6j, 245:344:9j].reshape(2, -1).T
# # xy = np.mgrid[155:245:6j, 284:344:9j].reshape(2, -1).T
# x1 = np.linspace(155, 134, 6)
# y1 = np.linspace(284, 388, 6)
# x2 = np.linspace(261, 245, 6)
# y2 = np.linspace(224, 344, 6)




# for point in xy:
#     print(point)
#     cv.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 0))

# cv.imshow('img', img)
# cv.waitKey(0)

# print(xy)