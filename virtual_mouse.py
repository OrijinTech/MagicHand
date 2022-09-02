import autopy as autopy
import cv2
import numpy as np
import HandDetector
import time

cam_w, cam_h = 640, 480
frameR = 140

cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
smooth = 6
lastClick = 0

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
        # print(fingers)
        clickLength, clickImg, clickInfo = detector.getDist(4, 8, img)
        dragLength, dragImg, dragInfo = detector.getDist(8, 12, img, draw=False)
        rightCLength, rcImg, rcInfo = detector.getDist(4, 12, img)
        print(rightCLength)
        cv2.rectangle(img, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (255, 0, 255), 2)
        # Convert coordinates
        x3 = np.interp(x1, (frameR, cam_w - frameR), (0, wScreen))
        y3 = np.interp(y1, (frameR, cam_h - frameR), (0, hScreen))
        # Smoothing values
        clocX = plocX + (x3 - plocX) / smooth
        clocY = plocY + (y3 - plocY) / smooth

        # Move mode
        if clickLength >= 50 and dragLength >= 50:
            # Move mouse
            try:
                autopy.mouse.move(wScreen - clocX, clocY)
            except ValueError:
                print("ValueError")
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Click Mode (Go into lock position)
        elif 50 > clickLength > 0:
            # Find distance between fingers (4 and 8 are the tips of the fingers)
            # clickLength, img, clickInfo = detector.getDist(4, 8, img)
            # Click when dist of the two fingers are short.
            if clickLength < 18:
                curClick = time.time()
                cv2.circle(img, (clickInfo[4], clickInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                if curClick > lastClick + 0.2:
                    autopy.mouse.click()
                    lastClick = curClick

        # Drag Mode
        elif 50 > dragLength > 0:
            # Move mouse
            try:
                autopy.mouse.toggle(down=True)
                autopy.mouse.move(wScreen - clocX, clocY)
            except ValueError:
                print("ValueError")

        # Right Click
        # elif rightCLength < 20 and clickLength > 50:
        #     if rightCLength < 15:
        #         curClick = time.time()
        #         if curClick > lastClick + 0.2:
        #             mouse.right_click()
        #             lastClick = curClick

    # Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (250, 0, 0), 3)
    # Display
    cv2.imshow("image", img)
    cv2.waitKey(1)
