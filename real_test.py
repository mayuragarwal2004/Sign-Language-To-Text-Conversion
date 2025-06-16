import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the model and label encoder
model = load_model('gesture_lstm_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# Landmark indices used for pose (based on your dataset)
POSE_INDICES = [
    0, 2, 5, 7, 8,     # Head & eyes
    11, 12,            # Shoulders
    13, 14,            # Elbows
    15, 16             # Wrists
]
POSE_LEN = len(POSE_INDICES) * 3
HAND_LEN = 21 * 3 * 2
INPUT_LEN = POSE_LEN + HAND_LEN

# Helper to normalize pose + hand landmarks
def extract_features(pose_landmarks, hands_results):
    if not pose_landmarks:
        return None, False

    lm = pose_landmarks.landmark
    if any(lm[i].visibility < 0.5 for i in POSE_INDICES):
        return None, False

    l_shoulder = lm[11]
    r_shoulder = lm[12]
    ref_x = (l_shoulder.x + r_shoulder.x) / 2
    ref_y = (l_shoulder.y + r_shoulder.y) / 2
    scale = ((l_shoulder.x - r_shoulder.x) ** 2 + (l_shoulder.y - r_shoulder.y) ** 2) ** 0.5
    if scale == 0:
        scale = 1

    data = []
    for i in POSE_INDICES:
        data += [
            (lm[i].x - ref_x) / scale,
            (lm[i].y - ref_y) / scale,
            lm[i].z / scale
        ]

    for h in range(2):  # Up to 2 hands
        if hands_results.multi_hand_landmarks and len(hands_results.multi_hand_landmarks) > h:
            hlms = hands_results.multi_hand_landmarks[h].landmark
            data += [((l.x - ref_x) / scale, (l.y - ref_y) / scale, l.z / scale) for l in hlms]
            data = [v for point in data for v in (point if isinstance(point, tuple) else [point])]
        else:
            data += [0] * (21 * 3)

    return np.array(data), True


# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)

    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    features, visible = extract_features(pose_results.pose_landmarks, hand_results)

    if visible and features is not None and len(features) == INPUT_LEN:
        input_tensor = np.expand_dims(features, axis=0)
        predictions = model.predict(input_tensor, verbose=0)[0]
        top_label = label_encoder.inverse_transform([np.argmax(predictions)])[0]

        # Draw all class probabilities
        for idx, prob in enumerate(predictions):
            label = label_encoder.inverse_transform([idx])[0]
            text = f"{label}: {prob:.2f}"
            color = (0, 255, 0) if label == top_label else (200, 200, 200)
            y = 30 + idx * 25
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    else:
        cv2.putText(frame, "Upper body not fully visible", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Gesture Predictor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
