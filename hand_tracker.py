"""
Hand tracking — MediaPipe compatibility wrapper and threaded camera capture.
"""

import cv2
import os
import sys
import threading
import time

import mediapipe as mp
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.core import base_options as mp_base_options


def _find_model():
    """Locate hand_landmarker.task inside the mediapipe package."""
    for p in sys.path:
        candidate = os.path.join(p, "mediapipe", "modules", "hand_landmark", "hand_landmarker.task")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("Could not find hand_landmarker.task model in mediapipe package.")


class _Landmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class HandTracker:
    """Wraps MediaPipe HandLandmarker (v0.10+) behind a simple API.

    Call ``start()`` to begin background capture + detection,
    then ``latest()`` to grab the most recent landmarks list (or *None*).
    Call ``stop()`` when done.
    """

    def __init__(self, *, max_hands=1, detection_confidence=0.3,
                 tracking_confidence=0.3, cam_width=640, cam_height=480):
        model_path = _find_model()
        options = HandLandmarkerOptions(
            base_options=mp_base_options.BaseOptions(model_asset_path=model_path),
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = HandLandmarker.create_from_options(options)
        self._cam_width = cam_width
        self._cam_height = cam_height

        self._lock = threading.Lock()
        self._landmarks = None  # list[_Landmark] | None
        self._running = False
        self._thread = None
        self._cap = None

    # ------------------------------------------------------------------
    def start(self):
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open camera")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cam_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cam_height)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()

    # ------------------------------------------------------------------
    def latest(self):
        """Return the latest detected hand landmarks or *None*."""
        with self._lock:
            return self._landmarks

    # ------------------------------------------------------------------
    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)
            result = self._detector.detect(mp_img)

            landmarks = None
            if result.hand_landmarks:
                raw = result.hand_landmarks[0]
                landmarks = [_Landmark(l.x, l.y, l.z) for l in raw]

            with self._lock:
                self._landmarks = landmarks
            time.sleep(0.001)
