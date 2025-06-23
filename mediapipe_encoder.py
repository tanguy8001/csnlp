import mediapipe as mp
import numpy as np

POSE_IDX = [11, 12, 13, 14, 23, 24]
FACE_IDX = [
    0, 4, 13, 14, 17, 33, 37, 39, 46, 52, 55, 61, 64, 81, 82,
    93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276, 282,
    285, 291, 294, 311, 323, 362, 386, 397, 468, 473
]

def create_holistic_model(static_image_mode=False):
    return mp.solutions.holistic.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True
    )

def extract_landmarks(frames, holistic_model):
    all_features = []
    for frame in frames:
        res = holistic_model.process(frame[..., ::-1])  # BGR → RGB

        pose_feats = np.zeros((len(POSE_IDX), 3))
        face_feats = np.zeros((len(FACE_IDX), 3))

        if res.pose_landmarks:
            pose_landmarks = res.pose_landmarks.landmark
            pose_feats = np.array([[
                pose_landmarks[i].x,
                pose_landmarks[i].y,
                pose_landmarks[i].z
            ] for i in POSE_IDX])

        if res.face_landmarks:
            face_landmarks = res.face_landmarks.landmark
            face_feats = np.array([[
                face_landmarks[i].x,
                face_landmarks[i].y,
                face_landmarks[i].z
            ] for i in FACE_IDX])

        # Concatenate both sets of features
        combined = np.concatenate([pose_feats, face_feats], axis=0)  # shape: (N, 3)
        all_features.append(combined.flatten())  # shape: (N*3,)
    
    return np.stack(all_features)  # shape: (T, N*3)
