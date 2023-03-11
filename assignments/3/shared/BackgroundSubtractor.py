import cv2 as cv
import numpy as np

# The BackgroundSubtractor class handles subtracting background (or creating foregrounds) from images
class BackgroundSubtractor:

    def __init__(self, path):
        # Use OpenCV's KNN background subtractor
        self.subtractor = cv.createBackgroundSubtractorKNN(detectShadows=True)
        vid = cv.VideoCapture(path)
        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                # Learn from every frame
                self.subtractor.apply(frame, learningRate=0.5)
            else:
                vid.release()
                break
        pass

    def get_foreground(self, frame):
        # Do not learn as much from actual video frames
        fg = self.subtractor.apply(frame, learningRate=0.00001)

        # Remove detected shadows
        fg[fg == 127] = 0

        # Find the biggest contour in the image
        contours, hierarchy = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        biggest_contours = []
        for contour in contours:
            if cv.contourArea(contour) > 1000:
                biggest_contours.append(contour)

        # Draw filled version of biggest contour onto empty mask, resulting in final foreground
        mask = np.zeros(fg.shape, np.uint8)

        cv.drawContours(mask, biggest_contours, -1, 255, -1)

        # Applying the morphologic closing operation (dilating, then eroding) gave results
        # that helped denoising the edges of the foreground without losing much detail, if any
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)

        return mask