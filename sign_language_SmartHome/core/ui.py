import cv2

class UIManager:
    @staticmethod
    def draw_landmarks(display_image, detector, w_img, h_img):
        if detector.hand_results and detector.hand_results.multi_hand_landmarks:
            for hand_landmarks in detector.hand_results.multi_hand_landmarks:
                # 관절 연결선 그리기 (고급스러운 느낌을 위해 추가)
                import mediapipe.python.solutions.hands as mp_hands
                from mediapipe.python.solutions import drawing_utils, drawing_styles
                
                # 직접 원을 그리는 대신 스타일링된 랜드마크 사용 가능하지만, 
                # 여기서는 기존 방식을 유지하되 색상과 크기를 조정
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w_img), int(lm.y * h_img)
                    cv2.circle(display_image, (cx, cy), 3, (255, 255, 255), -1)
                    cv2.circle(display_image, (cx, cy), 5, (255, 170, 0), 1)

    @staticmethod
    def draw_dashboard(canvas, x_offset, y_offset, width, height):
        """네스트허브 스타일의 하단 또는 측면 대시보드 UI를 그립니다."""
        # 1. 시계 및 날짜
        import datetime
        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M")
        date_str = now.strftime("%A, %b %d")
        
        cv2.putText(canvas, time_str, (x_offset + 40, y_offset + 80), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3)
        cv2.putText(canvas, date_str, (x_offset + 40, y_offset + 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)
        
        # 2. 스마트홈 상태 카드 (세로 배치 또는 가로 배치)
        # 가로 폭이 좁으면 세로로 배치 (현재 512px)
        if width < 600:
            UIManager._draw_card(canvas, x_offset + 40, y_offset + 180, width - 80, 100, "Lights", "OFF", (100, 100, 100))
            UIManager._draw_card(canvas, x_offset + 40, y_offset + 300, width - 80, 100, "A/C", "24'C", (255, 100, 0))
            UIManager._draw_card(canvas, x_offset + 40, y_offset + 420, width - 80, 100, "Security", "Armed", (0, 150, 0))
        else:
            UIManager._draw_card(canvas, x_offset + 350, y_offset + 40, 180, 220, "Lights", "OFF", (100, 100, 100))
            UIManager._draw_card(canvas, x_offset + 550, y_offset + 40, 180, 220, "A/C", "24'C", (255, 100, 0))
            UIManager._draw_card(canvas, x_offset + 750, y_offset + 40, 180, 220, "Security", "Armed", (0, 150, 0))

    @staticmethod
    def _draw_card(canvas, x, y, w, h, title, status, accent_color):
        # 카드 배경 (둥근 사각형 효과를 위해 사각형 중첩)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 50), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (70, 70, 70), 1)
        
        # 강조 색 바 (상단)
        cv2.rectangle(canvas, (x, y), (x + w, y + 10), accent_color, -1)
        
        # 텍스트
        cv2.putText(canvas, title, (x + 20, y + 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(canvas, status, (x + 20, y + 120), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

    @staticmethod
    def draw_gesture_popup(canvas, gesture_text, win_w, win_h):
        """인식된 제스처를 세련된 팝업으로 표시합니다."""
        text = f" {gesture_text.upper()} "
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.2
        thickness = 2
        
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 카메라 뷰(좌측 절반) 중앙에 배치
        cam_w = win_w // 2
        pad = 25
        rx1 = (cam_w - tw) // 2 - pad
        ry1 = win_h - 150 - th - pad
        rx2 = rx1 + tw + 2 * pad
        ry2 = ry1 + th + baseline + 2 * pad
        
        # 둥근 사각형 느낌 (Blur 효과 등은 OpenCV로 구현하기 복잡하므로 반투명 대신 진한 배경)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 170, 0), 2)
        
        text_x = rx1 + pad
        text_y = ry1 + pad + th
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
