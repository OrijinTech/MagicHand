import math

import cv2
import mediapipe as mp
import time


# More on Google Media Pipe

class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, minDetect=0.5, maxTracking=0.5):
        self.results = None
        self.jointList = None
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.minDetect = minDetect
        self.maxTracking = maxTracking
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.minDetect,
                                        self.maxTracking)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # Ids for the tip of the fingers.

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hd in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hd, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPos(self, img, hand_num=0, draw=True):
        xList = []
        yList = []
        boundBox = []
        self.jointList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_num]
            for idx, lm in enumerate(myHand.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                # print(idx, cx, cy)
                xList.append(cx)
                yList.append(cy)
                self.jointList.append([id, cx, cy])
                if draw:
                    # The color is B,G,R instead of RGB
                    cv2.circle(img, (cx, cy), 7, (169, 82, 169), cv2.FILLED)
            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            boundBox = xMin, yMin, xMax, yMax
            if draw:
                cv2.rectangle(img, (xMin - 20, yMin - 20), (xMax + 20, yMax + 20), (0, 255, 0), 2)

        return self.jointList, boundBox

    def fingerUp(self):
        fingers = []
        # This checks for the thumb
        if self.jointList[self.tipIds[0]][1] > self.jointList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # This is for the rest of the fingers.
        for id in range(1, 5):
            if self.jointList[self.tipIds[id]][2] < self.jointList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def thumbIndexDiff(self):
        fingers = []
        # This checks for the thumb
        if self.jointList[self.tipIds[0]][1] > self.jointList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        return fingers

    def getDist(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.jointList[p1][1:]
        x2, y2 = self.jointList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        dist = math.hypot(x2 - x1, y2 - y1)
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)
        return dist, img, [x1, y1, x2, y2, cx, cy]


