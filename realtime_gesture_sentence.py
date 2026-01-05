# import cv2
# import mediapipe as mp
# import numpy as np
# from tensorflow.keras.models import load_model
# from collections import deque

# # ---------------- CONFIG ----------------
# CONFIDENCE_THRESHOLD = 0.9
# BUFFER_SIZE = 10
# STABLE_COUNT = 8

# # ---------------- LOAD MODEL ----------------
# model = load_model("final_gesture_model.h5")
# labels = np.load("final_labels.npy", allow_pickle=True)

# # ---------------- MEDIAPIPE ----------------
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils

# # ---------------- CAMERA ----------------
# cap = cv2.VideoCapture(0)

# # ---------------- STABILITY ----------------
# gesture_buffer = deque(maxlen=BUFFER_SIZE)
# stable_gesture = None
# last_committed_gesture = None   # ⭐ edge trigger
# live_gesture = "Detecting..."

# # ---------------- TEXT STATE ----------------
# sentence_lines = [""]
# cursor_line = 0
# cursor_pos = 0


# # ---------------- APPLY GESTURE ----------------
# def apply_gesture(g):
#     global cursor_line, cursor_pos, sentence_lines

#     line = sentence_lines[cursor_line]

#     # LETTERS
#     if len(g) == 1 and g.isalpha():
#         line = line[:cursor_pos] + g + line[cursor_pos:]
#         cursor_pos += 1

#     # SPACE
#     elif g == "Like":
#         line = line[:cursor_pos] + " " + line[cursor_pos:]
#         cursor_pos += 1

#     # DELETE
#     elif g == "Dislike":
#         if cursor_pos > 0:
#             line = line[:cursor_pos - 1] + line[cursor_pos:]
#             cursor_pos -= 1

#     # NEW LINE
#     elif g == "Talk to hand":
#         remaining = line[cursor_pos:]
#         sentence_lines[cursor_line] = line[:cursor_pos]
#         sentence_lines.insert(cursor_line + 1, remaining)
#         cursor_line += 1
#         cursor_pos = 0
#         return

#     sentence_lines[cursor_line] = line


# # ---------------- MAIN LOOP ----------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)

#     gesture = None
#     live_gesture = "Detecting..."

#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             data = []

#             # -------- NORMALIZATION --------
#             base_x = hand_landmarks.landmark[0].x
#             base_y = hand_landmarks.landmark[0].y
#             base_z = hand_landmarks.landmark[0].z

#             for lm in hand_landmarks.landmark:
#                 data.extend([
#                     lm.x - base_x,
#                     lm.y - base_y,
#                     lm.z - base_z
#                 ])

#             # -------- PREDICTION --------
#             prediction = model.predict(np.array([data]), verbose=0)

#             if not np.isnan(prediction).any():
#                 confidence = np.max(prediction)
#                 predicted_label = str(labels[np.argmax(prediction)])
#                 live_gesture = predicted_label   # ⭐ live feedback

#                 if confidence >= CONFIDENCE_THRESHOLD:
#                     gesture = predicted_label

#             mp_draw.draw_landmarks(
#                 frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
#             )

#     # -------- STABILITY LOGIC (LIKE TEXT CODE) --------
#     gesture_buffer.append(gesture)

#     if gesture and gesture_buffer.count(gesture) >= STABLE_COUNT:
#         stable_gesture = gesture
#     else:
#         stable_gesture = None

#     # -------- EDGE-TRIGGERED APPLY --------
#     if stable_gesture and stable_gesture != last_committed_gesture:
#         apply_gesture(stable_gesture)
#         last_committed_gesture = stable_gesture

#     if stable_gesture is None:
#         last_committed_gesture = None

#     # -------- TEXT PANEL WITH CURSOR --------
#     text_panel = np.zeros_like(frame)
#     y = 50
#     for i, line in enumerate(sentence_lines):
#         if i == cursor_line:
#             display = line[:cursor_pos] + "|" + line[cursor_pos:]
#         else:
#             display = line

#         cv2.putText(
#             text_panel,
#             display,
#             (20, y),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (255, 255, 255),
#             2
#         )
#         y += 40

#     # -------- LIVE DETECTING DISPLAY --------
#     cv2.putText(
#         frame,
#         f"Detecting: {live_gesture}",
#         (20, 40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.2,
#         (0, 255, 255),
#         2
#     )

#     combined = np.hstack((frame, text_panel))
#     cv2.imshow("Sign Language Sentence Builder", combined)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # ---------------- CLEANUP ----------------
# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
from collections import deque

# ================= CONFIG =================
CONFIDENCE_THRESHOLD = 0.85
BUFFER_SIZE = 10
STABLE_COUNT = 8
PRED_SMOOTH = 5
COOLDOWN = 0.7

# ================= LOAD MODEL =================
model = load_model("final_gesture_model.h5")
labels = np.load("final_labels.npy", allow_pickle=True)

# ================= DISPLAY MAP =================
DISPLAY_MAP = {
    "Like": "[SPACE]",
    "Dislike": "[DELETE]",
    "Talk to hand": "[NEW LINE]"
}

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# ================= BUFFERS =================
gesture_buffer = deque(maxlen=BUFFER_SIZE)
pred_buffer = deque(maxlen=PRED_SMOOTH)
z_path = deque(maxlen=30)

last_committed_gesture = None
last_commit_time = 0

# ================= Z STATE =================
z_state = 0
z_start_y = None

# ================= TEXT STATE =================
sentence_lines = [""]
cursor_line = 0
cursor_pos = 0


# ================= APPLY GESTURE =================
def apply_gesture(g):
    global cursor_line, cursor_pos, sentence_lines
    line = sentence_lines[cursor_line]

    if len(g) == 1 and g.isalpha():
        line = line[:cursor_pos] + g + line[cursor_pos:]
        cursor_pos += 1

    elif g == "Like":
        line = line[:cursor_pos] + " " + line[cursor_pos:]
        cursor_pos += 1

    elif g == "Dislike":
        if cursor_pos > 0:
            line = line[:cursor_pos - 1] + line[cursor_pos:]
            cursor_pos -= 1

    elif g == "Talk to hand":
        remaining = line[cursor_pos:]
        sentence_lines[cursor_line] = line[:cursor_pos]
        sentence_lines.insert(cursor_line + 1, remaining)
        cursor_line += 1
        cursor_pos = 0
        return

    sentence_lines[cursor_line] = line


# ================= STATE-BASED Z DETECTION =================
def detect_z_motion_state(path):
    global z_state, z_start_y

    if len(path) < 2:
        return False

    x_prev, y_prev = path[-2]
    x_curr, y_curr = path[-1]

    dx = x_curr - x_prev
    dy = y_curr - y_prev

    # Phase 0 → move right
    if z_state == 0 and dx > 0.01:
        z_state = 1
        z_start_y = y_curr

    # Phase 1 → diagonal down-left
    elif z_state == 1 and dx < -0.005 and dy > 0.005:
        z_state = 2

    # Phase 2 → move right again + downward progress
    elif z_state == 2 and dx > 0.01 and (y_curr - z_start_y) > 0.02:
        z_state = 3
        return True

    return False


# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    live_label = "Detecting..."
    gesture = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # -------- NORMALIZATION --------
            base = hand_landmarks.landmark[0]
            ref = hand_landmarks.landmark[9]

            scale = np.linalg.norm([
                ref.x - base.x,
                ref.y - base.y,
                ref.z - base.z
            ]) + 1e-6

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([
                    (lm.x - base.x) / scale,
                    (lm.y - base.y) / scale,
                    (lm.z - base.z) / scale
                ])

            # -------- STATIC MODEL --------
            pred = model.predict(np.array([data]), verbose=0)[0]
            pred_buffer.append(pred)

            avg_pred = np.mean(pred_buffer, axis=0)
            confidence = np.max(avg_pred)
            label = str(labels[np.argmax(avg_pred)])

            if confidence >= CONFIDENCE_THRESHOLD:
                gesture = label
                live_label = label
            else:
                gesture = None

            # -------- Z MOTION (INDEX FINGER) --------
            index_tip = hand_landmarks.landmark[8]
            z_path.append((index_tip.x, index_tip.y))

            # Prevent static interference while drawing Z
            if len(z_path) >= 5:
                if abs(z_path[-1][0] - z_path[0][0]) > 0.02:
                    gesture = None

            if detect_z_motion_state(z_path):
                gesture = "Z"
                live_label = "Z"

                z_path.clear()
                pred_buffer.clear()
                gesture_buffer.clear()
                z_state = 0

            if len(z_path) >= z_path.maxlen:
                z_path.clear()
                z_state = 0

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # -------- STABILITY --------
    gesture_buffer.append(gesture)
    stable_gesture = None

    if gesture is not None and gesture_buffer.count(gesture) >= STABLE_COUNT:
        stable_gesture = gesture

    # -------- EDGE TRIGGER --------
    now = time.time()
    if (
        stable_gesture and
        stable_gesture != last_committed_gesture and
        now - last_commit_time > COOLDOWN
    ):
        apply_gesture(stable_gesture)
        last_committed_gesture = stable_gesture
        last_commit_time = now

    if stable_gesture is None:
        last_committed_gesture = None

    # -------- TEXT PANEL --------
    text_panel = np.zeros_like(frame)
    y = 50
    for i, line in enumerate(sentence_lines):
        if i == cursor_line:
            display = line[:cursor_pos] + "|" + line[cursor_pos:]
        else:
            display = line

        cv2.putText(
            text_panel, display, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 255, 255), 2
        )
        y += 40

    # -------- LIVE UI --------
    display_live = DISPLAY_MAP.get(live_label, live_label)
    cv2.putText(
        frame, f"Live: {display_live}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        2
    )

    combined = np.hstack((frame, text_panel))
    cv2.imshow("Sign Language Sentence Builder (A–Z)", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()




