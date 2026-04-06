import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Screen size
screen_w, screen_h = 1366, 768

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Smoothing
prev_x, prev_y = 0, 0
smoothening = 6

# Click cooldown
click_delay = 0.4
last_left_click = 0
last_right_click = 0

# Scroll
prev_scroll_y = 0
scroll_speed = 2

# Control box margin
frame_margin = 100

# Drag state
left_dragging = False
right_dragging = False

def finger_half_folded(lm, tip, middle_joint):
    """Returns True if finger is half-folded (tip below middle_joint but not fully folded)"""
    return middle_joint[1] < lm[tip][1] < middle_joint[1] + 40  # adjust 40 px if needed

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            lm_list = [(int(p.x * w), int(p.y * h)) for p in lm]

            # Finger states
            index_up = lm_list[8][1] < lm_list[6][1]
            middle_up = lm_list[12][1] < lm_list[10][1]
            ring_up = lm_list[16][1] < lm_list[14][1]
            pinky_up = lm_list[20][1] < lm_list[18][1]

            index_half = finger_half_folded(lm_list, 8, lm_list[6])
            middle_half = finger_half_folded(lm_list, 12, lm_list[10])

            # Cursor position
            x, y = lm_list[8]
            control_x1, control_y1 = frame_margin, frame_margin
            control_x2, control_y2 = w - frame_margin, h - frame_margin
            screen_x = np.interp(x, [control_x1, control_x2], [0, screen_w])
            screen_y = np.interp(y, [control_y1, control_y2], [0, screen_h])
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            # ✋ Pause (all fingers up)
            if index_up and middle_up and ring_up and pinky_up:
                cv2.putText(img, "PAUSED", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if left_dragging:
                    pyautogui.mouseUp()
                    left_dragging = False
                if right_dragging:
                    pyautogui.mouseUp(button='right')
                    right_dragging = False

            # ✌️ Scroll (index + middle up)
            elif index_up and middle_up:
                if prev_scroll_y != 0:
                    dy = y - prev_scroll_y
                    pyautogui.scroll(int(-dy * scroll_speed))
                prev_scroll_y = y

            # ✊ Right click / drag (half-folded index + middle up)
            elif index_half and middle_up:
                if not right_dragging:
                    pyautogui.mouseDown(button='right')
                    right_dragging = True
                pyautogui.moveTo(curr_x, curr_y)
                last_right_click = time.time()

            # ✊ Left click / drag (half-folded middle + index up)
            elif middle_half and index_up:
                if not left_dragging:
                    pyautogui.mouseDown()
                    left_dragging = True
                pyautogui.moveTo(curr_x, curr_y)
                last_left_click = time.time()

            # Move cursor (index up, no dragging)
            elif index_up and not middle_up and not left_dragging and not right_dragging:
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

            # Release drag if finger opens
            else:
                if left_dragging and not middle_half:
                    pyautogui.mouseUp()
                    left_dragging = False
                if right_dragging and not index_half:
                    pyautogui.mouseUp(button='right')
                    right_dragging = False
                prev_scroll_y = 0

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("AirControl", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
