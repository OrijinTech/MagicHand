import cv2
import numpy as np
import hand_detector as hdt
import time
import autopy

cam_w, cam_h = 640, 480


cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)

while True:
    success, img = cap.read()
    cv2.imshow("image", img)
    cv2.waitKey(1)









