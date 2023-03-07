from shared.VoxelCam import VoxelCam
import cv2 as cv

vc = VoxelCam(4)
while True:
    vc.next_frame()
    cv.imshow('frame', vc.frame)
    cv.imshow('fg', vc.fg)
    print(vc.video.get(cv.CAP_PROP_POS_FRAMES))
    cv.waitKey(0)