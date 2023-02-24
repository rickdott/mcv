import os
import cv2 as cv
import numpy as np

class BackgroundSubtractor:

    def __init__(self, path):
        self.subtractor = cv.createBackgroundSubtractorKNN(detectShadows=True)
        vid = cv.VideoCapture(path)
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                self.subtractor.apply(frame, learningRate=0.1)
            else:
                vid.release()
                break
        pass

    def get_foreground(self, frame):
        fg = self.subtractor.apply(frame, 0)
        # Remove detected shadows
        fg[fg == 127] = 0

        # fg = cv.erode(fg, erosion_kernel, iterations=1)
        # fg = cv.dilate(fg, erosion_kernel, iterations=1)
        contours, hierarchy = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv.contourArea)

        mask = np.zeros(fg.shape, np.uint8)
        
        # Draw filled version of biggest contour onto empty mask, resulting in final foreground
        cv.drawContours(mask, [biggest_contour], 0, 255, -1)

        return mask
        # cv.imshow('image', frame)
        # cv.imshow('fg', mask)
        # cv.waitKey(0)
        # Perhaps dilate or erode? Better to overestimate area so dilate? Check during task 3