import cv2
import time
import numpy as np
import json
import os
from collections import deque
from core.camera import get_camera
from core.ui import UIManager
from detector.landmark import Detector

class InteractionApp:
    def __init__(self):
        self.camera = get_camera(width=1280, height=720, fps=30)
        self.detector = Detector()
        self.ui = UIManager()
        
        self.gesture_timer_start = None
        
        # 제스처 인식 관련 초기화
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        try:
            model_path = os.path.join(BASE_DIR, 'gesture_svm_model.xml')
            label_path = os.path.join(BASE_DIR, 'gesture_labels.json')
            self.gesture_model = cv2.ml.SVM_load(model_path)
            with open(label_path, 'r') as f:
                self.label_map = json.load(f)
        except Exception as e:
            print(f"Failed to load gesture model: {e}")
            self.gesture_model = None
            self.label_map = {}
            
        self.gesture_buffer = deque(maxlen=20) # 20프레임(약 0.66초)으로 조정 (안정성과 속도 균형)
        self.current_gesture = "None"
        self.last_gesture_text = ""
        self.gesture_display_time = 0
        self.last_triggered_gesture = "None"
        self.gesture_locked = False  # 제스처 인식 잠금 플래그

    def start(self):
        self.camera.start()
        try:
            self.run_loop()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()

    def run_loop(self):
        # 구글 네스트허브 사양 (1024x600)
        WINDOW_W, WINDOW_H = 1024, 600
        CAM_W = 512 # 화면의 가로 절반
        
        while True:
            ir_image = self.camera.get_frame()
            if ir_image is None:
                continue
                
            if len(ir_image.shape) == 2 or (len(ir_image.shape) == 3 and ir_image.shape[2] == 1):
                display_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            else:
                display_image = ir_image.copy()
            
            # 1. 카메라 화면 처리 (512x600 비율에 맞춤)
            # 1280x720 -> 512x600 비율 (target_w = 720 * 512/600 = 614)
            h_full, w_full = display_image.shape[:2]
            target_w = int(h_full * (CAM_W / WINDOW_H))
            start_x = (w_full - target_w) // 2
            
            # 크롭 및 리사이즈 (512x600)
            cam_crop = display_image[:, start_x : start_x + target_w]
            cam_view = cv2.resize(cam_crop, (CAM_W, WINDOW_H))
            
            # 2. 메인 캔버스 생성 (1024x600)
            canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
            canvas[:, CAM_W:WINDOW_W] = (30, 30, 30) # 우측 대시보드 배경색
            
            # 3. 핸드 트래킹 (카메라 뷰 기준)
            self.detector.process_hands(cam_view)
            self.ui.draw_landmarks(cam_view, self.detector, CAM_W, WINDOW_H)
            
            # 4. 제스처 인식 로직
            hand_pos = self.detector.get_left_hand_pos()
            if hand_pos is None:
                hand_pos = self.detector.get_right_hand_pos()
            
            # [수정] 손이 감지되지 않으면 상태를 초기화하고 스킵합니다.
            if hand_pos is None:
                self.gesture_locked = False
                self.gesture_buffer.clear()
                self.current_gesture = "None"
            else:
                # 손이 감지된 경우에만 인식 로직 진행
                if not self.gesture_locked:
                    angles = self.detector.get_joint_angles() # 30 angles
                    center = self.detector.get_hand_center() # 4 coords [rx, ry, lx, ly]
                    
                    if angles is not None and center is not None:
                        # [rx, ry, lx, ly, r_angle0...l_angle14]
                        features_single = center + angles
                        self.gesture_buffer.append(features_single)
                        
                        if len(self.gesture_buffer) == 20 and self.gesture_model is not None:
                            buffer_array = np.array(self.gesture_buffer, dtype=np.float32)
                            
                            # 1. Scaling: Normalize angles (0~180) to 0~1 range
                            buffer_array[:, 4:] = buffer_array[:, 4:] / 180.0
                            
                            # 2. Relative Position Normalization (Reference: first detected hand)
                            ref_x, ref_y = 0.0, 0.0
                            for frame in buffer_array:
                                if frame[0] != 0 or frame[1] != 0: # Right hand first
                                    ref_x, ref_y = frame[0], frame[1]
                                    break
                                elif frame[2] != 0 or frame[3] != 0: # Left hand fallback
                                    ref_x, ref_y = frame[2], frame[3]
                                    break

                            if ref_x != 0 or ref_y != 0:
                                buffer_array[:, 0] = np.where(buffer_array[:, 0] != 0, buffer_array[:, 0] - ref_x, 0)
                                buffer_array[:, 1] = np.where(buffer_array[:, 1] != 0, buffer_array[:, 1] - ref_y, 0)
                                buffer_array[:, 2] = np.where(buffer_array[:, 2] != 0, buffer_array[:, 2] - ref_x, 0)
                                buffer_array[:, 3] = np.where(buffer_array[:, 3] != 0, buffer_array[:, 3] - ref_y, 0)
                            
                            resampled = np.zeros((20, 34), dtype=np.float32) # 34 features
                            orig_idx = np.linspace(0, 1, 20)
                            target_idx = np.linspace(0, 1, 20)
                            for i in range(34):
                                resampled[:, i] = np.interp(target_idx, orig_idx, buffer_array[:, i])
                            
                            features = resampled.flatten().reshape(1, -1)
                            _, result = self.gesture_model.predict(features)
                            pred_class = int(result[0][0])
                            pred_label = self.label_map.get(str(pred_class), "Unknown")
                            self.current_gesture = pred_label
                    else:
                        self.gesture_buffer.clear()
                        self.current_gesture = "None"
                else:
                    # 제스처가 잠긴 상태(인식 직후)라면 버퍼를 비워둡니다.
                    self.gesture_buffer.clear()
                    self.current_gesture = "None"
            
            if self.current_gesture != "None" and self.current_gesture.lower() != "normal":
                self.last_gesture_text = self.current_gesture
                self.gesture_display_time = time.time()
                self.gesture_locked = True
            
            # 5. UI 합성
            # 카메라 뷰를 캔버스 좌측에 배치
            canvas[0:WINDOW_H, 0:CAM_W] = cam_view
            
            # 우측 대시보드 UI (Smart Home Status)
            self.ui.draw_dashboard(canvas, CAM_W, 0, CAM_W, WINDOW_H)

            # 제스처 팝업 알림 (카메라 뷰 중앙 하단)
            if time.time() - self.gesture_display_time < 1.2 and self.last_gesture_text:
                self.ui.draw_gesture_popup(canvas, self.last_gesture_text, WINDOW_W, WINDOW_H)

            # 최종 출력
            cv2.imshow('Nest Hub Smart Home', canvas)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
