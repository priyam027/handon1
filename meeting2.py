import mediapipe as mp

import cv2

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=2,

    min_detection_confidence=0.7,

    min_tracking_confidence=0.5

)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")

    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Frame capture failed.")

        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(

                frame,

                hand_landmarks,

                mp_hands.HAND_CONNECTIONS,

                mp_drawing_styles.get_default_hand_landmarks_style(),

                mp_drawing_styles.get_default_hand_connections_style()

            )

    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

