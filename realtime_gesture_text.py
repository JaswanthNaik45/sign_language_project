import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# ---------------- CONFIG ----------------
CONFIDENCE_THRESHOLD = 0.9
BUFFER_SIZE = 10
STABLE_COUNT = 8

# ---------------- LOAD MODEL ----------------
model = load_model("final_gesture_model.h5")
labels = np.load("final_labels.npy", allow_pickle=True)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- STABILITY BUFFER ----------------
gesture_buffer = deque(maxlen=BUFFER_SIZE)
stable_gesture = "Detecting..."

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # ---- DEFAULT STATE (NO HAND) ----
    gesture = "Show Gesture"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            data = []

            # -------- NORMALIZATION --------
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z

            for lm in hand_landmarks.landmark:
                data.extend([
                    lm.x - base_x,
                    lm.y - base_y,
                    lm.z - base_z
                ])

            # -------- PREDICTION --------
            prediction = model.predict(np.array([data]), verbose=0)

            # -------- NaN SAFETY --------
            if np.isnan(prediction).any():
                gesture = "No Gesture"
                continue

            confidence = np.max(prediction)
            predicted_label = labels[np.argmax(prediction)]

            # -------- CONFIDENCE FILTER --------
            if confidence >= CONFIDENCE_THRESHOLD:
                gesture = predicted_label
            else:
                gesture = "No Gesture"

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # -------- STABILITY LOGIC --------
    gesture_buffer.append(gesture)

    if gesture_buffer.count(gesture) >= STABLE_COUNT:
        stable_gesture = gesture
    else:
        stable_gesture = "Detecting..."

    # -------- DISPLAY --------
    cv2.putText(
        frame,
        f"Gesture: {stable_gesture}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Real-Time Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()














# import cv2
# import mediapipe as mp
# import numpy as np
# from tensorflow.keras.models import load_model

# model = load_model("final_gesture_model.h5")
# # labels = np.load("final_labels.npy")
# labels = np.load("final_labels.npy", allow_pickle=True)


# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)

#     gesture = "No Hand"

#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             data = []
#             for lm in hand_landmarks.landmark:
#                 data.extend([lm.x, lm.y, lm.z])

#             prediction = model.predict(np.array([data]))
#             gesture = labels[np.argmax(prediction)]

#             mp_draw.draw_landmarks(
#                 frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
#             )

#     cv2.putText(frame, f"Gesture: {gesture}",
#                 (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                 1.5, (0, 255, 0), 3)

#     cv2.imshow("Real-Time Gesture to Text", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows() can you add motion detection code for Z in the code