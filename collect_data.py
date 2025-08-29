import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Output file
csv_path = "data/gesture_data.csv"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create file if not exists
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')] + ['label']
        writer.writerow(header)

# Webcam
cap = cv2.VideoCapture(0)
print("Press 's' to save sample, 'q' to quit.\n")

label = input("Enter gesture label (e.g., 'fist', 'palm', 'peace', 'thumbs'): ")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 63 values
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            # Save when 's' is pressed
            key = cv2.waitKey(1)
            if key == ord('s'):
                with open(csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmark_list + [label])
                print(f"✅ Sample saved for label: {label}")

    # Show video
    cv2.imshow("Collecting Hand Data", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()