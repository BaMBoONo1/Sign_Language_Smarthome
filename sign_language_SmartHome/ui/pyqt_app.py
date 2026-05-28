import json
import os
import sys
import time
from collections import deque


def _configure_qt_runtime():
    qt_root = os.path.join(sys.prefix, "Lib", "site-packages", "PyQt5", "Qt5")
    plugin_root = os.path.join(qt_root, "plugins")
    platform_root = os.path.join(plugin_root, "platforms")
    bin_root = os.path.join(qt_root, "bin")

    if os.path.isdir(platform_root):
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", platform_root)
    if os.path.isdir(plugin_root):
        os.environ.setdefault("QT_PLUGIN_PATH", plugin_root)
    if hasattr(os, "add_dll_directory") and os.path.isdir(bin_root):
        os.add_dll_directory(bin_root)


_configure_qt_runtime()

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


APP_WIDTH = 1280
APP_HEIGHT = 800
CAMERA_WIDTH = 612
RIGHT_PANEL_WIDTH = 620


DEVICE_TYPES = ["조명", "에어컨", "보일러"]
DEFAULT_ROOMS = ["방", "거실", "부엌"]
DEFAULT_DEVICES = [
    {"room": "방", "type": "조명", "status": "ON", "active": True, "connected": True},
    {"room": "방", "type": "에어컨", "status": "OFF", "active": False, "connected": True, "temp" : 20},
    {"room": "거실", "type": "조명", "status": "ON", "active": True, "connected": True},
    {"room": "거실", "type": "보일러", "status": "ON", "active": True, "connected": True, "temp": 22},
    {"room": "부엌", "type": "조명", "status": "OFF", "active": False, "connected": True},
]


LABEL_TO_WORD = {
    "Room": "방",
    "Living_Room": "거실",
    "Kitchen": "부엌",
    "All": "전체",
    "AC": "에어컨",
    "Boiler": "보일러",
    "One": "1도",
    "Two": "2도",
    "Four": "4도",
    "Light_On": "조명 켜다",
    "Light_Off": "조명 끄다",
    "On": "켜다",
    "Off": "끄다",
    "Temp_Up": "온도 높이기",
    "Temp_Down": "온도 낮추기",
    "Enter": "실행",
    "Check": "확인",
    "Erase": "지우기",
    "Start": "시작",
}


def refresh_style(widget):
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def make_scroll_area(widget):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setWidget(widget)
    return scroll


def device_default(room, device_type):
    data = {
        "room": room,
        "type": device_type,
        "status": "OFF",
        "active": False,
        "connected": True,
    }
    if device_type in ("에어컨", "보일러"):
        data["temp"] = 22
    return data


class GestureRecognizer:
    def __init__(self):
        from detector.landmark import Detector

        self.detector = Detector()
        self.gesture_model = None
        self.label_map = {}
        self.gesture_buffer = deque(maxlen=20)
        self.current_gesture = "None"
        self.gesture_locked = False
        self.last_emit_time = 0

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            model_path = os.path.join(base_dir, "gesture_svm_model.xml")
            label_path = os.path.join(base_dir, "gesture_labels.json")
            if os.path.exists("gesture_svm_model.xml"):
                model_path = "gesture_svm_model.xml"
            if os.path.exists("gesture_labels.json"):
                label_path = "gesture_labels.json"
            self.gesture_model = cv2.ml.SVM_load(model_path)
            with open(label_path, "r", encoding="utf-8") as file:
                self.label_map = json.load(file)
        except Exception as exc:
            print(f"Failed to load gesture model: {exc}")

    def process_frame(self, frame):
        self.detector.process_hands(frame)
        self._draw_landmarks(frame)
        hand_pos = self.detector.get_left_hand_pos() or self.detector.get_right_hand_pos()

        if hand_pos is None:
            self.gesture_locked = False
            self.gesture_buffer.clear()
            self.current_gesture = "None"
            return None

        if self.gesture_locked:
            self.gesture_buffer.clear()
            self.current_gesture = "None"
            return None

        angles = self.detector.get_joint_angles()
        center = self.detector.get_hand_center()
        if angles is None or center is None:
            self.gesture_buffer.clear()
            self.current_gesture = "None"
            return None

        self.gesture_buffer.append(center + angles)
        if len(self.gesture_buffer) < 20 or self.gesture_model is None:
            return None

        features = self._make_features()
        if features is None:
            return None

        _, result = self.gesture_model.predict(features)
        pred_class = int(result[0][0])
        pred_label = self.label_map.get(str(pred_class), "Unknown")
        self.current_gesture = pred_label

        if pred_label in ("None", "Normal", "Unknown"):
            return None

        self.gesture_locked = True
        if time.time() - self.last_emit_time < 0.8:
            return None

        self.last_emit_time = time.time()
        return LABEL_TO_WORD.get(pred_label, pred_label)

    def _make_features(self):
        buffer_array = np.array(self.gesture_buffer, dtype=np.float32)
        if buffer_array.shape != (20, 34):
            return None

        buffer_array[:, 4:] = buffer_array[:, 4:] / 180.0
        ref_x, ref_y = 0.0, 0.0
        for frame in buffer_array:
            if frame[0] != 0 or frame[1] != 0:
                ref_x, ref_y = frame[0], frame[1]
                break
            if frame[2] != 0 or frame[3] != 0:
                ref_x, ref_y = frame[2], frame[3]
                break

        if ref_x != 0 or ref_y != 0:
            buffer_array[:, 0] = np.where(buffer_array[:, 0] != 0, buffer_array[:, 0] - ref_x, 0)
            buffer_array[:, 1] = np.where(buffer_array[:, 1] != 0, buffer_array[:, 1] - ref_y, 0)
            buffer_array[:, 2] = np.where(buffer_array[:, 2] != 0, buffer_array[:, 2] - ref_x, 0)
            buffer_array[:, 3] = np.where(buffer_array[:, 3] != 0, buffer_array[:, 3] - ref_y, 0)

        resampled = np.zeros((20, 34), dtype=np.float32)
        orig_idx = np.linspace(0, 1, 20)
        target_idx = np.linspace(0, 1, 20)
        for i in range(34):
            resampled[:, i] = np.interp(target_idx, orig_idx, buffer_array[:, i])
        return resampled.flatten().reshape(1, -1)

    def _draw_landmarks(self, frame):
        results = self.detector.hand_results
        if not results or not results.multi_hand_landmarks:
            return

        height, width = frame.shape[:2]
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                cx = int(landmark.x * width)
                cy = int(landmark.y * height)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 7, (37, 99, 255), 2)


class CameraWorker(QThread):
    frame_ready = pyqtSignal(object)
    recognized_word = pyqtSignal(str)
    camera_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._recognition_enabled = True
        self.recognizer = None
        self.is_active = False
        self.flat_hands_start_time = None
        self.last_action_time = time.time()

    def set_recognition_enabled(self, enabled):
        self._recognition_enabled = enabled

    def stop(self):
        self._running = False

    def run(self):
        camera = None
        try:
            from core.camera import get_camera

            camera = get_camera(width=1280, height=720, fps=30)
            camera.start()
            self.camera_message.emit("카메라가 연결되었습니다")

            try:
                self.recognizer = GestureRecognizer()
                self.camera_message.emit("대기 상태 · 양손을 쫙 펴서 3초간 유지하여 활성화")
            except Exception as exc:
                self.recognizer = None
                self.camera_message.emit(f"카메라 연결됨 · 인식기 준비 실패: {exc}")

            while self._running:
                frame = camera.get_frame()
                if frame is None:
                    self.msleep(10)
                    continue

                if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    display_frame = frame.copy()

                if self._recognition_enabled and self.recognizer is not None:
                    try:
                        now = time.time()
                        if not self.is_active:
                            # 대기 상태: 랜드마크 검출 및 렌더링만 진행
                            self.recognizer.detector.process_hands(display_frame)
                            self.recognizer._draw_landmarks(display_frame)
                            
                            is_flat = self.recognizer.detector.check_flat_hands()
                            if is_flat:
                                if self.flat_hands_start_time is None:
                                    self.flat_hands_start_time = now
                                elapsed = now - self.flat_hands_start_time
                                if elapsed >= 3.0:
                                    self.is_active = True
                                    self.last_action_time = now
                                    self.flat_hands_start_time = None
                                    self.camera_message.emit("수화 인식 활성화됨 (20초 미입력 시 대기 전환)")
                                else:
                                    self.camera_message.emit(f"양손 감지됨 · 활성화까지 {3.0 - elapsed:.1f}초...")
                            else:
                                if self.flat_hands_start_time is not None:
                                    self.flat_hands_start_time = None
                                    self.camera_message.emit("대기 상태 · 양손을 쫙 펴서 3초간 유지하여 활성화")
                        else:
                            # 활성화 상태: 전체 프레임 처리(제스처 분류 포함)
                            word = self.recognizer.process_frame(display_frame)
                            is_flat = self.recognizer.detector.check_flat_hands()
                            
                            if is_flat:
                                self.last_action_time = now
                                
                            if now - self.last_action_time >= 20.0:
                                self.is_active = False
                                self.flat_hands_start_time = None
                                self.camera_message.emit("대기 상태 · 양손을 쫙 펴서 3초간 유지하여 활성화")
                            else:
                                if word:
                                    self.last_action_time = now
                                    self.camera_message.emit(f"수화 인식 중 · 감지: {word} (대기 시간 초기화)")
                                    self.recognized_word.emit(word)
                                else:
                                    remaining = 20.0 - (now - self.last_action_time)
                                    self.camera_message.emit(f"수화 인식 활성화 중 (남은 대기 시간: {remaining:.1f}초)")

                    except Exception as exc:
                        self.recognizer = None
                        self.camera_message.emit(f"카메라 연결됨 · 인식 처리 중지: {exc}")

                self.frame_ready.emit(display_frame)
                self.msleep(12)
        except Exception as exc:
            self.camera_message.emit(f"카메라 대기 중: {exc}")
        finally:
            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass


class CameraPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("CameraPanel")
        self.setFixedWidth(CAMERA_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)

        self.video_label = QLabel()
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("카메라 피드")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label)

        self.status_label = QLabel("카메라를 준비하고 있습니다")
        self.status_label.setObjectName("CameraStatus")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

    def set_message(self, message):
        self.status_label.setText(message)

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        image = QImage(rgb.data, width, height, channel * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        target_size = self.video_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        scaled = pixmap.scaled(target_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        x = max(0, (scaled.width() - target_size.width()) // 2)
        y = max(0, (scaled.height() - target_size.height()) // 2)
        self.video_label.setPixmap(scaled.copy(x, y, target_size.width(), target_size.height()))


class TopNavigation(QFrame):
    mode_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("TopNavigation")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(12)

        self.recognition_button = self._make_button("문장 인식", "recognition")
        self.status_button = self._make_button("기기 상태", "status")
        self.setup_button = self._make_button("설정", "setup")

        layout.addWidget(self.recognition_button)
        layout.addWidget(self.status_button)
        layout.addWidget(self.setup_button)
        self.update_active("recognition")

    def _make_button(self, text, mode):
        button = QPushButton(text)
        button.setProperty("navButton", True)
        button.clicked.connect(lambda: self.mode_requested.emit(mode))
        return button

    def update_active(self, mode):
        for button, button_mode in (
            (self.recognition_button, "recognition"),
            (self.status_button, "status"),
            (self.setup_button, "setup"),
        ):
            button.setProperty("active", button_mode == mode)
            button.setProperty("setupTab", button_mode == "setup")
            refresh_style(button)


class RoomCard(QFrame):
    delete_requested = pyqtSignal(str)

    def __init__(self, room):
        super().__init__()
        self.room = room
        self.setObjectName("ListCard")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 12, 12, 12)

        name = QLabel(room)
        name.setObjectName("ListCardTitle")
        layout.addWidget(name, 1)

        delete_button = QPushButton("삭제")
        delete_button.setProperty("dangerButton", True)
        delete_button.clicked.connect(lambda: self.delete_requested.emit(self.room))
        layout.addWidget(delete_button)


class SetupScreen(QWidget):
    go_back = pyqtSignal()
    data_changed = pyqtSignal()

    def __init__(self, rooms, devices):
        super().__init__()
        self.rooms = rooms
        self.devices = devices
        self.selected_device_type = DEVICE_TYPES[0]

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(28, 24, 28, 24)
        self.content_layout.setSpacing(18)

        header = QHBoxLayout()
        title = QLabel("초기 설정")
        title.setObjectName("ScreenTitle")
        back = QPushButton("돌아가기")
        back.setProperty("primaryButton", True)
        back.clicked.connect(self.go_back.emit)
        header.addWidget(title, 1)
        header.addWidget(back)
        self.content_layout.addLayout(header)

        self._build_room_section()
        self._build_device_section()
        self.content_layout.addStretch()

        root.addWidget(make_scroll_area(content))

    def _build_room_section(self):
        section = QFrame()
        section.setObjectName("SetupSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("방 추가")
        title.setObjectName("SectionTitle")
        self.room_input = QLineEdit()
        self.room_input.setPlaceholderText("방 이름 입력")
        add_button = QPushButton("+ 방 추가")
        add_button.setProperty("primaryButton", True)
        add_button.clicked.connect(self.add_room)

        self.room_list_layout = QVBoxLayout()
        self.room_list_layout.setSpacing(10)

        layout.addWidget(title)
        layout.addWidget(self.room_input)
        layout.addWidget(add_button)
        layout.addLayout(self.room_list_layout)
        self.content_layout.addWidget(section)
        self.update_room_list()

    def _build_device_section(self):
        section = QFrame()
        section.setObjectName("SetupSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("가전 추가")
        title.setObjectName("SectionTitle")
        self.room_combo = QComboBox()

        type_row = QHBoxLayout()
        type_row.setSpacing(10)
        self.type_buttons = []
        for device_type in DEVICE_TYPES:
            button = QPushButton(device_type)
            button.setProperty("choiceButton", True)
            button.clicked.connect(lambda checked=False, value=device_type: self.select_device_type(value))
            type_row.addWidget(button)
            self.type_buttons.append(button)

        add_button = QPushButton("+ 가전 추가")
        add_button.setProperty("primaryButton", True)
        add_button.clicked.connect(self.add_device)

        self.device_list_layout = QVBoxLayout()
        self.device_list_layout.setSpacing(10)

        layout.addWidget(title)
        layout.addWidget(QLabel("방 선택"))
        layout.addWidget(self.room_combo)
        layout.addWidget(QLabel("가전 종류"))
        layout.addLayout(type_row)
        layout.addWidget(add_button)
        layout.addLayout(self.device_list_layout)
        self.content_layout.addWidget(section)

        self.update_room_combo()
        self.update_device_type_buttons()
        self.update_device_list()

    def add_room(self):
        name = self.room_input.text().strip()
        if not name or name in self.rooms:
            return
        self.rooms.append(name)
        self.room_input.clear()
        self.update_room_list()
        self.update_room_combo()
        self.data_changed.emit()

    def delete_room(self, room):
        if room not in self.rooms:
            return
        self.rooms.remove(room)
        self.devices[:] = [device for device in self.devices if device["room"] != room]
        self.update_room_list()
        self.update_room_combo()
        self.update_device_list()
        self.data_changed.emit()

    def select_device_type(self, device_type):
        self.selected_device_type = device_type
        self.update_device_type_buttons()

    def add_device(self):
        if not self.rooms:
            return
        room = self.room_combo.currentText()
        self.devices.append(device_default(room, self.selected_device_type))
        self.update_device_list()
        self.data_changed.emit()

    def delete_device(self, index):
        if 0 <= index < len(self.devices):
            del self.devices[index]
            self.update_device_list()
            self.data_changed.emit()

    def update_room_list(self):
        self._clear_layout(self.room_list_layout)
        for room in self.rooms:
            card = RoomCard(room)
            card.delete_requested.connect(self.delete_room)
            self.room_list_layout.addWidget(card)

    def update_room_combo(self):
        current = self.room_combo.currentText() if hasattr(self, "room_combo") else ""
        self.room_combo.blockSignals(True)
        self.room_combo.clear()
        self.room_combo.addItems(self.rooms)
        if current in self.rooms:
            self.room_combo.setCurrentText(current)
        self.room_combo.blockSignals(False)

    def update_device_type_buttons(self):
        for button in self.type_buttons:
            button.setProperty("selected", button.text() == self.selected_device_type)
            refresh_style(button)

    def update_device_list(self):
        self._clear_layout(self.device_list_layout)
        for index, device in enumerate(self.devices):
            card = QFrame()
            card.setObjectName("ListCard")
            row = QHBoxLayout(card)
            row.setContentsMargins(18, 12, 12, 12)
            label = QLabel(f"{device['room']} - {device['type']}")
            label.setObjectName("ListCardTitle")
            delete_button = QPushButton("삭제")
            delete_button.setProperty("dangerButton", True)
            delete_button.clicked.connect(lambda checked=False, idx=index: self.delete_device(idx))
            row.addWidget(label, 1)
            row.addWidget(delete_button)
            self.device_list_layout.addWidget(card)

    def refresh_all(self):
        self.update_room_list()
        self.update_room_combo()
        self.update_device_list()

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


class WordButton(QPushButton):
    clicked_word = pyqtSignal(str)

    def __init__(self, word, category):
        super().__init__(word)
        self.word = word
        self.setProperty("wordCategory", category)
        self.clicked.connect(lambda: self.clicked_word.emit(self.word))

    def set_selected(self, selected):
        self.setProperty("selected", selected)
        refresh_style(self)


class TouchInputPanel(QFrame):
    word_selected = pyqtSignal(str)
    open_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("TouchInputPanel")
        self.sequence = []
        self.word_buttons = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header_button = QPushButton("TOUCH INPUT PANEL  접기/펼치기")
        self.header_button.setObjectName("TouchHeader")
        self.header_button.clicked.connect(self.toggle)
        layout.addWidget(self.header_button)

        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(20, 16, 20, 16)
        body_layout.setSpacing(12)

        self.sequence_title = QLabel("SELECTED SEQUENCE")
        self.sequence_title.setObjectName("SmallTitle")
        self.sequence_box = QLabel("단어를 선택하여 문장을 구성하세요")
        self.sequence_box.setObjectName("SequenceBox")
        self.sequence_box.setWordWrap(True)
        body_layout.addWidget(self.sequence_title)
        body_layout.addWidget(self.sequence_box)

        words_container = QWidget()
        words_layout = QGridLayout(words_container)
        words_layout.setContentsMargins(0, 0, 0, 0)
        words_layout.setHorizontalSpacing(12)
        words_layout.setVerticalSpacing(12)

        groups = [
            ("장소", "place", ["방", "거실", "부엌"]),
            ("가전", "device", ["에어컨", "보일러"]),
            ("정도", "degree", ["1도", "2도", "4도"]),
            ("동작", "action", ["켜다", "끄다", "조명 켜다", "조명 끄다", "온도 높이기", "온도 낮추기"]),
        ]
        for index, (title, category, words) in enumerate(groups):
            group = QFrame()
            group.setObjectName("WordGroupCard")
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(12, 12, 12, 12)
            group_layout.setSpacing(8)
            label = QLabel(title)
            label.setProperty("groupLabel", category)
            group_layout.addWidget(label)

            button_grid = QGridLayout()
            button_grid.setContentsMargins(0, 0, 0, 0)
            button_grid.setHorizontalSpacing(8)
            button_grid.setVerticalSpacing(8)
            for word_index, word in enumerate(words):
                button = WordButton(word, category)
                button.clicked_word.connect(self.word_selected.emit)
                self.word_buttons.append(button)
                button_grid.addWidget(button, word_index // 2, word_index % 2)

            group_layout.addLayout(button_grid)
            group_layout.addStretch()
            words_layout.addWidget(group, index // 2, index % 2)

        body_layout.addWidget(make_scroll_area(words_container), 1)
        layout.addWidget(self.body, 1)
        self.set_open(False)

    def toggle(self):
        self.set_open(not self.body.isVisible())

    def set_open(self, open_state):
        self.body.setVisible(open_state)
        self.header_button.setProperty("open", open_state)
        self.header_button.setText("TOUCH INPUT PANEL  ▲" if open_state else "TOUCH INPUT PANEL  ▼")
        refresh_style(self.header_button)
        self.open_changed.emit(open_state)

    def set_sequence(self, sequence):
        self.sequence = list(sequence)
        if self.sequence:
            self.sequence_box.setText("  ".join(self.sequence))
        else:
            self.sequence_box.setText("단어를 선택하여 문장을 구성하세요")
        for button in self.word_buttons:
            button.set_selected(button.word in self.sequence)


class RecognitionScreen(QWidget):
    command_executed = pyqtSignal(list, bool)
    mode_change = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.sequence = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sentence_panel = QFrame()
        self.sentence_panel.setObjectName("SentencePanel")
        sentence_layout = QVBoxLayout(self.sentence_panel)
        sentence_layout.setContentsMargins(28, 26, 28, 24)
        sentence_layout.setSpacing(16)

        title = QLabel("인식된 문장")
        title.setObjectName("ScreenTitle")
        self.recognized_text = QLabel("수화를 인식하면 여기에 표시됩니다")
        self.recognized_text.setObjectName("RecognizedText")
        self.recognized_text.setWordWrap(True)
        legend = QLabel("장소   가전   정도   동작   제어")
        legend.setObjectName("LegendText")
        sentence_layout.addWidget(title)
        sentence_layout.addWidget(self.recognized_text)
        sentence_layout.addWidget(legend)

        self.touch_panel = TouchInputPanel()
        self.touch_panel.word_selected.connect(self.handle_word)
        self.touch_panel.open_changed.connect(self.update_touch_layout)

        footer = QFrame()
        footer.setObjectName("FooterActions")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(28, 16, 28, 18)
        footer_layout.setSpacing(18)
        clear_button = QPushButton("전체 지우기")
        clear_button.setProperty("secondaryButton", True)
        clear_button.clicked.connect(self.clear_sequence)
        execute_button = QPushButton("명령 실행")
        execute_button.setProperty("primaryButton", True)
        execute_button.clicked.connect(lambda: self.execute_sequence(auto_clear=False))
        footer_layout.addWidget(clear_button)
        footer_layout.addWidget(execute_button)

        layout.addWidget(self.sentence_panel)
        layout.addWidget(self.touch_panel, 1)
        layout.addWidget(footer)

    def update_touch_layout(self, touch_open):
        self.sentence_panel.setVisible(not touch_open)

    def handle_word(self, word):
        # 확인 또는 check → 기기 상태 화면 전환
        if word in ("확인", "check"):
            self.mode_change.emit("status")
            return
        # 입력 또는 enter → 인식 화면 전환 (문장 입력 가능)
        if word in ("입력", "enter"):
            self.mode_change.emit("recognition")
            return
        if word == "지우기":
            self.clear_sequence()
            return
        if word in ("실행", "시작"):
            # 조명 조작, 전원 조작, 온도 조작 단어가 포함되어 있는지 확인
            action_words = ("조명 켜다", "조명 끄다", "켜다", "끄다", "온도 높이기", "온도 낮추기")
            has_action = any(w in self.sequence for w in action_words)
            if not has_action:
                return
            self.execute_sequence(auto_clear=(word == "시작"))
            return

        step = len(self.sequence)
        # 1단계: 위치
        if step == 0:
            allowed = ("방", "거실", "부엌", "전체")
            if word not in allowed:
                return
        # 2단계: 조명 조작 또는 가전
        elif step == 1:
            allowed = ("조명 켜다", "조명 끄다", "에어컨", "보일러")
            if word not in allowed:
                return
        # 3단계: 전원 조작 또는 온도
        elif step == 2:
            if self.sequence[1] in ("조명 켜다", "조명 끄다"):
                return
            allowed = ("켜다", "끄다", "1도", "2도", "4도")
            if word not in allowed:
                return
        # 4단계: 온도 조작
        elif step == 3:
            if self.sequence[2] in ("켜다", "끄다"):
                return
            allowed = ("온도 높이기", "온도 낮추기")
            if word not in allowed:
                return
        else:
            return

        self.sequence.append(word)
        self.touch_panel.set_sequence(self.sequence)
        self.recognized_text.setText(" ".join(self.sequence))

    def clear_sequence(self):
        self.sequence = []
        self.touch_panel.set_sequence(self.sequence)
        self.recognized_text.setText("수화를 인식하면 여기에 표시됩니다")

    def execute_sequence(self, auto_clear=False):
        is_auto = (auto_clear is True)
        if not self.sequence:
            self.recognized_text.setText("실행할 명령이 없습니다")
            return
        self.recognized_text.setText("명령 실행: " + " ".join(self.sequence))
        self.command_executed.emit(list(self.sequence), is_auto)


class DeviceCard(QFrame):
    def __init__(self, device):
        super().__init__()
        self.setObjectName("DeviceCard")
        self.setProperty("active", bool(device.get("active")))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(14)

        icon = QLabel(self._icon_for(device["type"]))
        icon.setObjectName("DeviceIcon")
        icon.setProperty("active", bool(device.get("active")))
        icon.setAlignment(Qt.AlignCenter)

        middle = QVBoxLayout()
        name = QLabel(device["type"])
        name.setObjectName("DeviceName")
        state = QLabel(self._state_text(device))
        state.setObjectName("DeviceState")
        middle.addWidget(name)
        middle.addWidget(state)

        connection = QLabel("연결됨" if device.get("connected", True) else "연결 안됨")
        connection.setObjectName("ConnectionText")
        connection.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(icon)
        layout.addLayout(middle, 1)
        layout.addWidget(connection)

    @staticmethod
    def _icon_for(device_type):
        if device_type == "에어컨":
            return "A/C"
        if device_type == "보일러":
            return "℃"
        return "ON"

    @staticmethod
    def _state_text(device):
        status = device.get("status", "OFF")
        # 온도가 표시되는 조건: 장치가 켜져 있을 때만
        if device.get("active", False) and "temp" in device:
            return f"{status} · {device['temp']}℃"
        return status


class DeviceStatusScreen(QWidget):
    def __init__(self, rooms, devices):
        super().__init__()
        self.rooms = rooms
        self.devices = devices

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(28, 24, 28, 24)
        self.content_layout.setSpacing(18)
        root.addWidget(make_scroll_area(content))
        self.rebuild()

    def rebuild(self):
        # 기존 위젯과 레이아웃을 모두 정리하여 UI 뒤틀림 방지
        self._clear_layout(self.content_layout)

        header = QLabel("SYSTEM STATUS")
        header.setObjectName("ScreenTitle")
        self.content_layout.addWidget(header)

        for room in self.rooms:
            room_devices = [device for device in self.devices if device["room"] == room]
            if not room_devices:
                continue

            room_label = QLabel(room)
            room_label.setObjectName("RoomTitle")
            self.content_layout.addWidget(room_label)

            grid = QGridLayout()
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(14)
            for index, device in enumerate(room_devices):
                grid.addWidget(DeviceCard(device), index // 2, index % 2)
            self.content_layout.addLayout(grid)

    def _clear_layout(self, layout):
        """재귀적으로 레이아웃과 그 안의 위젯을 모두 삭제합니다."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                child_layout = item.layout()
                if child_layout:
                    self._clear_layout(child_layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.rooms = list(DEFAULT_ROOMS)
        self.devices = [dict(device) for device in DEFAULT_DEVICES]
        self.current_mode = "recognition"

        self.setWindowTitle("수화 기반 스마트홈 보조 시스템")
        self.setFixedSize(APP_WIDTH, APP_HEIGHT)

        root = QWidget()
        root.setObjectName("AppRoot")
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(0)

        self.camera_panel = CameraPanel()
        self.right_panel = QFrame()
        self.right_panel.setObjectName("RightPanel")
        self.right_panel.setFixedWidth(RIGHT_PANEL_WIDTH)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.navigation = TopNavigation()
        self.navigation.mode_requested.connect(self.set_mode)

        self.stack = QStackedWidget()
        self.setup_screen = SetupScreen(self.rooms, self.devices)
        self.setup_screen.go_back.connect(lambda: self.set_mode("recognition"))
        self.setup_screen.data_changed.connect(self.refresh_data_views)
        self.recognition_screen = RecognitionScreen()
        self.recognition_screen.command_executed.connect(self.apply_command)
        self.recognition_screen.mode_change.connect(self.set_mode)
        self.status_screen = DeviceStatusScreen(self.rooms, self.devices)
        self.status_screen.rebuild()

        self.stack.addWidget(self.recognition_screen)
        self.stack.addWidget(self.status_screen)
        self.stack.addWidget(self.setup_screen)

        right_layout.addWidget(self.navigation)
        right_layout.addWidget(self.stack, 1)

        main_layout.addWidget(self.camera_panel)
        main_layout.addWidget(self.right_panel)
        self.setCentralWidget(root)
        self.setStyleSheet(STYLE_SHEET)

        self.camera_worker = CameraWorker(self)
        self.camera_worker.frame_ready.connect(self.camera_panel.update_frame)
        self.camera_worker.recognized_word.connect(self.handle_recognized_word)
        self.camera_worker.camera_message.connect(self.camera_panel.set_message)
        self.camera_worker.start()

    def set_mode(self, mode):
        self.current_mode = mode
        index = {"recognition": 0, "status": 1, "setup": 2}[mode]
        self.stack.setCurrentIndex(index)
        self.navigation.update_active(mode)
        self.camera_worker.set_recognition_enabled(mode in ("recognition", "status"))
        if mode == "status":
            self.status_screen.rebuild()
        if mode == "setup":
            self.setup_screen.refresh_all()

    def handle_recognized_word(self, word):
        # 현재 모드가 "status"일 경우, "입력"/"enter"/"실행"만 인식하여 인식 화면으로 전환하도록 허용
        if self.current_mode == "status":
            word_lower = word.lower()
            if word_lower in ("입력", "enter", "실행"):
                self.set_mode("recognition")
            return
        # "입력"/"enter" 로 인식 화면 전환 (기존 동작)
        if word in ("입력", "enter"):
            self.recognition_screen.handle_word(word)
            return
        # 그 외 모든 단어를 인식 화면에 전달 (명령 실행 등) – 현재 화면에 관계없이 동작
        self.recognition_screen.handle_word(word)


    def refresh_data_views(self):
        self.status_screen.rebuild()

    def apply_command(self, sequence, auto_clear=False):
        joined = " ".join(sequence)
        target_room = next((room for room in self.rooms if room in sequence), None)
        target_type = next((device_type for device_type in DEVICE_TYPES if device_type in joined), None)

        for device in self.devices:
            if target_room and device["room"] != target_room:
                continue
            if target_type and device["type"] != target_type:
                continue

            if "끄다" in joined:
                device["status"] = "OFF"
                device["active"] = False
            elif "켜다" in joined:
                device["status"] = "ON"
                device["active"] = True

            if "온도 높이기" in joined and "temp" in device:
                device["temp"] += self._degree_from(sequence)
                device["status"] = "ON"
                device["active"] = True
            elif "온도 낮추기" in joined and "temp" in device:
                device["temp"] -= self._degree_from(sequence)
                device["status"] = "ON"
                device["active"] = True

        self.status_screen.rebuild()
        # Auto-clear sequence after 2 seconds if requested
        if auto_clear:
            QTimer.singleShot(2000, self.recognition_screen.clear_sequence)




    @staticmethod
    def _degree_from(sequence):
        for word in sequence:
            if word.endswith("도"):
                try:
                    return int(word.replace("도", ""))
                except ValueError:
                    return 1
        return 1






STYLE_SHEET = """
* {
    font-family: "Malgun Gothic", "Noto Sans CJK KR", Arial;
    letter-spacing: 0;
}
#AppRoot {
    background: #eef2f6;
}
#CameraPanel {
    background: #142033;
    border-top-left-radius: 14px;
    border-bottom-left-radius: 14px;
}
#VideoLabel {
    color: #aab4c2;
    background: #10192a;
    border: 1px solid #22304a;
    border-radius: 12px;
    font-size: 30px;
    font-weight: 700;
}
#CameraStatus {
    color: #c8d0dc;
    font-size: 16px;
    padding-top: 12px;
}
#RightPanel {
    background: white;
    border-top-right-radius: 14px;
    border-bottom-right-radius: 14px;
}
#TopNavigation {
    background: white;
    border-bottom: 1px solid #dfe4ec;
}
QPushButton[navButton="true"] {
    min-height: 66px;
    padding: 0 22px;
    color: #697386;
    background: #f0f2f6;
    border: 0;
    border-radius: 18px;
    font-size: 24px;
    font-weight: 800;
}
QPushButton[navButton="true"][active="true"] {
    color: white;
    background: #2563ff;
}
QPushButton[navButton="true"][setupTab="true"][active="true"] {
    background: #a313f3;
}
#ScreenTitle {
    color: #111827;
    font-size: 31px;
    font-weight: 900;
}
#SectionTitle {
    color: #111827;
    font-size: 25px;
    font-weight: 900;
}
#SetupSection {
    background: #f7f9fc;
    border: 1px solid #e4e9f1;
    border-radius: 12px;
}
QLineEdit, QComboBox {
    min-height: 52px;
    padding: 0 14px;
    color: #111827;
    background: white;
    border: 2px solid #d7dde7;
    border-radius: 10px;
    font-size: 21px;
}
QLabel {
    color: #374151;
    font-size: 18px;
}
QPushButton[primaryButton="true"] {
    min-height: 56px;
    padding: 0 20px;
    color: white;
    background: #2563ff;
    border: 0;
    border-radius: 12px;
    font-size: 22px;
    font-weight: 900;
}
QPushButton[secondaryButton="true"] {
    min-height: 56px;
    padding: 0 20px;
    color: #7b8494;
    background: #f8fafc;
    border: 1px solid #dfe4ec;
    border-radius: 12px;
    font-size: 22px;
    font-weight: 900;
}
QPushButton[dangerButton="true"] {
    min-height: 38px;
    padding: 0 14px;
    color: #ef233c;
    background: #fff1f3;
    border: 1px solid #ffd2d8;
    border-radius: 9px;
    font-size: 16px;
    font-weight: 800;
}
QPushButton[choiceButton="true"] {
    min-height: 58px;
    color: #374151;
    background: white;
    border: 2px solid #d7dde7;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 800;
}
QPushButton[choiceButton="true"][selected="true"] {
    color: #2563ff;
    border-color: #2563ff;
    background: #eef4ff;
}
#ListCard {
    background: white;
    border: 1px solid #e1e7ef;
    border-radius: 10px;
}
#ListCardTitle {
    color: #111827;
    font-size: 20px;
    font-weight: 800;
}
#SentencePanel {
    background: white;
    border-bottom: 1px solid #e5e7eb;
}
#RecognizedText {
    min-height: 84px;
    padding: 18px 20px;
    color: #000000;
    background: #f7f8fa;
    border-radius: 12px;
    font-size: 27px;
    font-weight: 700;
}
#LegendText {
    color: #6b7280;
    font-size: 17px;
    font-weight: 700;
}
#TouchInputPanel {
    background: #fbfcfe;
}
#TouchHeader {
    min-height: 68px;
    color: #8c96a6;
    background: #f8fafc;
    border: 0;
    border-bottom: 1px solid #e5e7eb;
    font-size: 22px;
    font-weight: 900;
}
#TouchHeader[open="true"] {
    color: white;
    background: #2563ff;
}
#SmallTitle {
    color: #97a1b1;
    font-size: 15px;
    font-weight: 900;
}
#SequenceBox {
    min-height: 48px;
    padding: 12px 14px;
    color: #111827;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    font-size: 20px;
    font-weight: 800;
}
#WordGroupCard {
    background: white;
    border: 1px solid #e5e9f1;
    border-radius: 12px;
}
QPushButton[wordCategory] {
    min-height: 50px;
    padding: 0 8px;
    color: #374151;
    background: white;
    border: 1px solid #dfe4ec;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 900;
}
QPushButton[wordCategory][selected="true"] {
    color: white;
    background: #2563ff;
    border-color: #2563ff;
}
QLabel[groupLabel="place"] {
    color: #2f80ff;
    font-weight: 900;
}
QLabel[groupLabel="device"] {
    color: #a855f7;
    font-weight: 900;
}
QLabel[groupLabel="degree"] {
    color: #ff7300;
    font-weight: 900;
}
QLabel[groupLabel="action"] {
    color: #0ac75a;
    font-weight: 900;
}
#FooterActions {
    background: white;
    border-top: 1px solid #e5e7eb;
}
#RoomTitle {
    color: #8b95a6;
    font-size: 19px;
    font-weight: 900;
}
#DeviceCard {
    min-height: 92px;
    background: #f5f7fb;
    border: 1px solid #e9edf4;
    border-radius: 12px;
}
#DeviceCard[active="true"] {
    background: #f2fff7;
    border-color: #b9f3d0;
}
#DeviceIcon {
    min-width: 52px;
    min-height: 52px;
    color: white;
    background: #09c95b;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 900;
}
#DeviceIcon[active="false"] {
    color: #9aa3b2;
    background: #e8ecf2;
}
#DeviceName {
    color: #1f2937;
    font-size: 22px;
    font-weight: 900;
}
#DeviceState {
    color: #0aa64a;
    font-size: 16px;
    font-weight: 900;
}
#ConnectionText {
    color: #6b7280;
    font-size: 14px;
    font-weight: 800;
}
QScrollBar:vertical {
    width: 10px;
    background: #eef2f6;
}
QScrollBar::handle:vertical {
    background: #a7afbd;
    border-radius: 5px;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}
"""


def run_app():
    app = QApplication(sys.argv)
    app.setFont(QFont("Malgun Gothic", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
