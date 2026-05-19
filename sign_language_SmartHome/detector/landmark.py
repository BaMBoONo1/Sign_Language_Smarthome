import cv2
import numpy as np
import os
import shutil
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solution_base as mp_solution_base


def _configure_mediapipe_assets():
    source_modules = os.path.join(os.path.dirname(mp.__file__), "modules")
    candidates = []
    if os.environ.get("MEDIAPIPE_ASSET_ROOT"):
        candidates.append(os.environ["MEDIAPIPE_ASSET_ROOT"])
    candidates.extend(
        [
            os.path.join("C:\\", "SignLanguageSmartHome", "mediapipe_root"),
            os.path.join("C:\\", "Temp", "SignLanguageSmartHome", "mediapipe_root"),
            os.path.join("C:\\", "Users", "Public", "Documents", "SignLanguageSmartHome", "mediapipe_root"),
            os.path.join("C:\\", "Users", "Public", "SignLanguageSmartHome", "mediapipe_root"),
        ]
    )

    last_error = None
    for asset_root in candidates:
        target_modules = os.path.join(asset_root, "mediapipe", "modules")
        target_solution_base = os.path.join(asset_root, "mediapipe", "python", "solution_base.py")
        required_graph = os.path.join(target_modules, "hand_landmark", "hand_landmark_tracking_cpu.binarypb")
        try:
            if not os.path.exists(required_graph):
                os.makedirs(os.path.dirname(target_modules), exist_ok=True)
                shutil.copytree(source_modules, target_modules, dirs_exist_ok=True)

            os.makedirs(os.path.dirname(target_solution_base), exist_ok=True)
            if not os.path.exists(target_solution_base):
                with open(target_solution_base, "w", encoding="utf-8") as file:
                    file.write("# MediaPipe resource path anchor for non-ASCII project paths.\n")

            mp_solution_base.__file__ = target_solution_base
            return
        except Exception as exc:
            last_error = exc

    print(f"MediaPipe asset path fallback failed: {last_error}")


_configure_mediapipe_assets()

class Detector:
    def __init__(self):
        # Hands for finger joint recognition (Primary)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            model_complexity=0
        )
        self.hand_results = None
        self.width = 0
        self.height = 0

    def process_hands(self, frame):
        self.height, self.width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.hand_results = self.hands.process(rgb)
        return self.hand_results.multi_hand_landmarks is not None

    # 사람 기준 왼손 (카메라 기준 Right 레이블)
    def get_left_hand_pos(self):
        """Returns the (x, y) coordinates of the hand palm centroid using wrist and MCP joints."""
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == 'Right':
                    landmarks = self.hand_results.multi_hand_landmarks[i].landmark
                    # 손바닥 중심을 결정하는 주요 관절 (손목 0, 검지~새끼 기저부 5, 9, 13, 17)
                    palm_indices = [0, 5, 9, 13, 17]
                    avg_x = sum(landmarks[idx].x for idx in palm_indices) / len(palm_indices)
                    avg_y = sum(landmarks[idx].y for idx in palm_indices) / len(palm_indices)
                    return avg_x, avg_y
        return None

    def get_left_index_pos(self):
        """Returns the (x, y) coordinates of the left index finger tip (landmark 8)."""
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == 'Right':
                    landmark = self.hand_results.multi_hand_landmarks[i].landmark[8]
                    return landmark.x, landmark.y
        return None

    def get_right_index_pos(self):
        """Returns the (x, y) coordinates of the right index finger tip (landmark 8)."""
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == 'Left':
                    landmark = self.hand_results.multi_hand_landmarks[i].landmark[8]
                    return landmark.x, landmark.y
        return None

    # 사람 기준 오른손 (카메라 기준 Left 레이블)
    def get_right_hand_pos(self):
        """Returns the (x, y) coordinates of the hand palm centroid using wrist and MCP joints."""
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == 'Left':
                    landmarks = self.hand_results.multi_hand_landmarks[i].landmark
                    # 손바닥 중심을 결정하는 주요 관절 (손목 0, 검지~새끼 기저부 5, 9, 13, 17)
                    palm_indices = [0, 5, 9, 13, 17]
                    avg_x = sum(landmarks[idx].x for idx in palm_indices) / len(palm_indices)
                    avg_y = sum(landmarks[idx].y for idx in palm_indices) / len(palm_indices)
                    return avg_x, avg_y
        return None

    def get_left_thumb_pos(self):
        """Returns the (x, y) coordinates of the left thumb tip (landmark 4)."""
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == 'Right':
                    landmark = self.hand_results.multi_hand_landmarks[i].landmark[4]
                    return landmark.x, landmark.y
        return None

    def get_right_thumb_pos(self):
        """Returns the (x, y) coordinates of the right thumb tip (landmark 4)."""
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == 'Left':
                    landmark = self.hand_results.multi_hand_landmarks[i].landmark[4]
                    return landmark.x, landmark.y
        return None

    def _get_hand_idx_by_label(self, label):
        """Returns the index of the hand with the given label ('Left' or 'Right')."""
        if self.hand_results and self.hand_results.multi_handedness:
            for i, handedness in enumerate(self.hand_results.multi_handedness):
                if handedness.classification[0].label == label:
                    return i
        return None

    def get_hand_center(self):
        """
        Returns [Right_X, Right_Y, Left_X, Left_Y] of the hand centroids.
        Note: MediaPipe 'Left' label = Human Right hand, 'Right' label = Human Left hand.
        """
        res = [0.0, 0.0, 0.0, 0.0]
        
        # Human Right (MediaPipe Left)
        r_idx = self._get_hand_idx_by_label('Left')
        if r_idx is not None:
            landmarks = self.hand_results.multi_hand_landmarks[r_idx].landmark
            palm_indices = [0, 5, 9, 13, 17]
            res[0] = sum(landmarks[idx].x for idx in palm_indices) / len(palm_indices)
            res[1] = sum(landmarks[idx].y for idx in palm_indices) / len(palm_indices)
            
        # Human Left (MediaPipe Right)
        l_idx = self._get_hand_idx_by_label('Right')
        if l_idx is not None:
            landmarks = self.hand_results.multi_hand_landmarks[l_idx].landmark
            palm_indices = [0, 5, 9, 13, 17]
            res[2] = sum(landmarks[idx].x for idx in palm_indices) / len(palm_indices)
            res[3] = sum(landmarks[idx].y for idx in palm_indices) / len(palm_indices)
            
        return res

    def get_joint_angles(self):
        """
        Returns a list of 30 joint angles: [Right_Angles(15), Left_Angles(15)].
        If a hand is missing, its angles are filled with 0.0.
        """
        all_angles = []
        
        for label in ['Left', 'Right']: # MediaPipe Left=HumanRight, MediaPipe Right=HumanLeft
            idx = self._get_hand_idx_by_label(label)
            if idx is not None:
                landmark_list = self.hand_results.multi_hand_landmarks[idx].landmark
                joint = np.zeros((21, 3))
                for j, lm in enumerate(landmark_list):
                    joint[j] = [lm.x, lm.y, lm.z]
                    
                v1_idx = np.array([0, 1, 2, 0, 5, 6, 0, 9, 10, 0, 13, 14, 0, 17, 18])
                v2_idx = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19])
                v3_idx = np.array([2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20])

                v1 = joint[v2_idx] - joint[v1_idx]
                v2 = joint[v3_idx] - joint[v2_idx]
                
                v1_norm = np.linalg.norm(v1, axis=1)
                v2_norm = np.linalg.norm(v2, axis=1)
                
                # Zero-division protection
                v1_norm[v1_norm == 0] = 1e-6
                v2_norm[v2_norm == 0] = 1e-6
                
                v1 = v1 / v1_norm[:, np.newaxis]
                v2 = v2 / v2_norm[:, np.newaxis]
                
                dot_product = np.sum(v1 * v2, axis=1)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angles = np.arccos(dot_product) * (180.0 / np.pi)
                all_angles.extend(np.round(angles, 2).tolist())
            else:
                all_angles.extend([0.0] * 15)
                
        return all_angles

