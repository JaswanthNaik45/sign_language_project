import cv2
import mediapipe as mp
import csv
import os

gesture_name = input("Enter gesture name: ")

save_dir = f"own_gestures/{gesture_name}"
os.makedirs(save_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

count = 0
TOTAL_SAMPLES = 300   # per gesture

print("Press 's' to save sample, 'q' to quit")

with open(f"{save_dir}/landmarks.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    writer.writerow(header)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                if cv2.waitKey(1) & 0xFF == ord('s') and count < TOTAL_SAMPLES:
                    row = []
                    # for lm in hand_landmarks.landmark:
                    #     row.extend([lm.x, lm.y, lm.z]) 
                    # replacing
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    base_z = hand_landmarks.landmark[0].z

                    for lm in hand_landmarks.landmark:
                        row.extend([
                            lm.x - base_x,
                            lm.y - base_y,
                            lm.z - base_z
                        ])

                    writer.writerow(row)
                    count += 1
                    print(f"Saved sample {count}/{TOTAL_SAMPLES}")

        cv2.putText(frame, f"Samples: {count}/{TOTAL_SAMPLES}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow("Collect Gesture Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= TOTAL_SAMPLES:
            break

cap.release()
cv2.destroyAllWindows()
print("Data collection completed")