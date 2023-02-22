import os
import cv2 as cv
import numpy as np

erosion_kernel = np.ones((5, 5), np.uint8)
dilatation_kernel = np.ones((5, 5), np.uint8)
for cam in range(1, 5):
    base_path = os.path.join('assignments', 'data', f'cam{cam}')
    background_path = os.path.join(base_path, 'background.avi')
    video_path = os.path.join(base_path, 'video.avi')

    background_subtractor = cv.createBackgroundSubtractorKNN(detectShadows=True)

    vid = cv.VideoCapture(background_path)
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            background_subtractor.apply(frame, learningRate=0.1)
        else:
            vid.release()
            break
    cv.imshow('bg', background_subtractor.getBackgroundImage())

    vid = cv.VideoCapture(video_path)
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            fg = background_subtractor.apply(frame, learningRate=0)
            # Remove detected shadows
            fg[fg == 127] = 0
            fg = cv.erode(fg, erosion_kernel)

            cv.imshow('video', frame)
            cv.imshow('fg', fg)
            cv.waitKey(0)
        else:
            vid.release()
            break