import autopy as autopy
import cv2
import numpy as np
import HandDetector
import time

cam_w, cam_h = 640, 480
frameR = 100

cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
smooth = 6

detector = HandDetector.HandDetector(maxHands=1)
wScreen, hScreen = autopy.screen.size()

while True:
    # Find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPos(img)

    # Get the tip of the index and middle fingers.
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # Check which fingers are up
        fingers = detector.fingerUp()
        print(fingers)
        # Only index finger: moving mode
        if fingers[1] == 1 and fingers[0] == 0:
            cv2.rectangle(img, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (255, 0, 255), 2)
            # Convert coordinates
            x3 = np.interp(x1, (frameR, cam_w - frameR), (0, wScreen))
            y3 = np.interp(y1, (frameR, cam_h - frameR), (0, hScreen))
            # Smoothing values
            clocX = plocX + (x3 - plocX) / smooth
            clocY = plocY + (y3 - plocY) / smooth

            # Move mouse
            autopy.mouse.move(wScreen - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        # Both index and thumb up: Click Mode
        if fingers[1] == 1 and fingers[0] == 1:
            # Find distance between fingers (4 and 8 are the tips of the fingers)
            length, img, lineInfo = detector.getDist(4, 8, img)
            print(length)
            # Click when dist of the two fingers are short.
            if length < 95:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                #autopy.mouse.click()

    # Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (250, 0, 0), 3)
    # Display
    cv2.imshow("image", img)
    cv2.waitKey(1)
