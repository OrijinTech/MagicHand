import cv2
import mediapipe as mp
import time

# More on Google Media Pipe

class hand_detector():
    def __init__(self, mode=False, max_hands=2, model_complexity=1, min_detect=0.5, max_tracking=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.min_detect = min_detect
        self.max_tracking = max_tracking
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.model_complexity, self.min_detect, self.max_tracking)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # Ids for the tip of the fingers.

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hd in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hd, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_pos(self, img, hand_num=0, draw=True):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]
            for idx, lm in enumerate(my_hand.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                # print(idx, cx, cy)
                self.lm_list.append([id, cx, cy])
                if draw:
                    # The color is B,G,R instead of RGB
                    cv2.circle(img, (cx, cy), 7, (169, 82, 169), cv2.FILLED)
        return self.lm_list


    def fingerUp(self):
        fingers = []
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)



