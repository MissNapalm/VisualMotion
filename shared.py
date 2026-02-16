"""
Shared utilities for the Gesture Carousel app.
MediaPipe compatibility wrapper, gesture detection, drawing helpers, and state classes.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import sys
from collections import deque
import subprocess
import time
import os
import threading

# --- Mediapipe Hands Compatibility Wrapper for v0.10+ ---
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.core import base_options as mp_base_options


def _find_hand_landmarker_model():
    for p in sys.path:
        candidate = os.path.join(p, 'mediapipe', 'modules', 'hand_landmark', 'hand_landmarker.task')
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError('Could not find hand_landmarker.task model in mediapipe package.')


class MPHandsCompat:
    """Wraps the new MediaPipe HandLandmarker API to mimic the old mp.solutions.hands interface."""

    def __init__(self, max_num_hands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, model_complexity=0):
        model_path = _find_hand_landmarker_model()
        options = HandLandmarkerOptions(
            base_options=mp_base_options.BaseOptions(model_asset_path=model_path),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = HandLandmarker.create_from_options(options)

    def process(self, image):
        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=image)
        result = self.detector.detect(mp_img)

        class Results:
            pass

        results = Results()
        results.multi_hand_landmarks = []
        results.multi_handedness = []
        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                class Landmark:
                    def __init__(self, x, y, z):
                        self.x = x
                        self.y = y
                        self.z = z

                hand = type('HandLandmarks', (), {})()
                hand.landmark = [Landmark(l.x, l.y, l.z) for l in hand_lms]
                results.multi_hand_landmarks.append(hand)
            for handedness in result.handedness:
                class Handedness:
                    def __init__(self, label):
                        self.classification = [type('Classification', (), {'label': label})()]

                label = handedness[0].category_name if handedness else 'Unknown'
                results.multi_handedness.append(Handedness(label))
        return results


# Patch for old code references
mp_hands = type('mp_hands', (), {'Hands': MPHandsCompat})


# ==============================
# Utility
# ==============================
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ==============================
# Font cache
# ==============================
_font_cache = {}


def get_font(size):
    if size not in _font_cache:
        _font_cache[size] = pygame.font.Font(None, size)
    return _font_cache[size]


# ==============================
# Carousel classes
# ==============================
class FingerSmoother:
    def __init__(self, window_size=5):
        self.thumb_history = deque(maxlen=window_size)
        self.index_history = deque(maxlen=window_size)

    def update(self, thumb_pos, index_pos):
        self.thumb_history.append(thumb_pos)
        self.index_history.append(index_pos)
        tx = sum(p[0] for p in self.thumb_history) / len(self.thumb_history)
        ty = sum(p[1] for p in self.thumb_history) / len(self.thumb_history)
        ix = sum(p[0] for p in self.index_history) / len(self.index_history)
        iy = sum(p[1] for p in self.index_history) / len(self.index_history)
        return (tx, ty), (ix, iy)

    def reset(self):
        self.thumb_history.clear()
        self.index_history.clear()


class HandState:
    def __init__(self):
        self.card_offset = 0.0
        self.category_offset = 0.0
        self.smooth_card_offset = 0.0
        self.smooth_category_offset = 0.0
        self.scroll_smoothing = 0.25
        self.scroll_gain = 5.0

        self.is_pinching = False
        self.last_pinch_x = None
        self.last_pinch_y = None
        self.pinch_start_pos = None
        self.movement_threshold = 10

        self.selected_card = None
        self.selected_category = None
        self.zoom_progress = 0.0
        self.zoom_target = 0.0
        self.finger_smoother = FingerSmoother(window_size=5)

        # zoom wheel
        self.wheel_active = False
        self.wheel_angle = math.pi
        self.last_finger_angle = None
        self.wheel_center_x = 0
        self.wheel_center_y = 0
        self.wheel_radius = 110
        self.gui_scale = 1.00
        self.gui_scale_min = 0.60
        self.gui_scale_max = 1.80
        self.gui_scale_sensitivity = 0.32

        # pinch timing - now for launch detection only
        self.pinch_threshold = 0.08
        self.pinch_prev = False
        self.last_pinch_time = 0
        self.double_pinch_window = 0.4
        self.pinch_hold_start = 0          # when current pinch began
        self.scroll_unlocked = False       # True once held 1.5s
        self.pinch_hold_delay = 0.45       # seconds before scroll activates

        # misc
        self.current_fps = 0.0


# ==============================
# Gesture detection functions
# ==============================
def get_pinch_distance(landmarks):
    if not landmarks:
        return None
    a = landmarks[4]
    b = landmarks[8]
    return math.hypot(a.x - b.x, a.y - b.y)


def is_pinching_gesture(landmarks, thresh):
    d = get_pinch_distance(landmarks)
    return (d is not None) and (d < thresh)


def get_pinch_position(landmarks):
    if not landmarks:
        return None
    a = landmarks[4]
    b = landmarks[8]
    return ((a.x + b.x) / 2, (a.y + b.y) / 2)


def is_finger_extended(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y


def detect_three_finger_gesture(landmarks):
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    wrist = landmarks[0]
    thumb_ext = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) * 0.8
    index_ext = is_finger_extended(landmarks, 8, 6)
    middle_ext = is_finger_extended(landmarks, 12, 10)
    ring_fold = landmarks[16].y > landmarks[14].y - 0.02
    pinky_fold = landmarks[20].y > landmarks[18].y - 0.02
    return thumb_ext and index_ext and middle_ext and ring_fold and pinky_fold


def get_hand_center(landmarks):
    return landmarks[9]


def calculate_finger_angle(landmarks):
    c = get_hand_center(landmarks)
    idx = landmarks[8]
    return math.atan2(idx.y - c.y, idx.x - c.x)


def lm_to_screen(lm, W, H):
    return (lm.x * W, lm.y * H)


# ==============================
# Drawing helpers
# ==============================
def draw_wheel(surface, state, window_width, window_height):
    if not state.wheel_active:
        return
    scale = state.gui_scale
    cx = state.wheel_center_x
    cy = state.wheel_center_y
    r = int(state.wheel_radius * scale)
    white = (255, 255, 255)
    margin = r + int(80 * scale)
    local_w = margin * 2
    local_h = margin * 2
    local_surf = pygame.Surface((local_w, local_h), pygame.SRCALPHA)
    lcx, lcy = margin, margin
    for i in range(5):
        rr = r + int(15 * scale) + i * int(10 * scale)
        op = int(100 - i * 20)
        pygame.draw.circle(local_surf, (*white, op), (lcx, lcy), rr, max(1, int(2 * scale)))
    surface.blit(local_surf, (cx - margin, cy - margin))
    pygame.draw.circle(surface, white, (cx, cy), r, max(1, int(4 * scale)))
    pygame.draw.circle(surface, white, (cx, cy), r - int(20 * scale), max(1, int(2 * scale)))
    segs = 48
    prog = int((state.wheel_angle / (2 * math.pi)) * segs) % segs
    ir = r - int(10 * scale)
    for i in range(prog):
        sa = math.radians(i * 360 / segs) - math.pi / 2
        ea = math.radians((i + 1) * 360 / segs) - math.pi / 2
        sx = cx + int(ir * math.cos(sa))
        sy = cy + int(ir * math.sin(sa))
        ex = cx + int(ir * math.cos(ea))
        ey = cy + int(ir * math.sin(ea))
        pygame.draw.line(surface, white, (sx, sy), (ex, ey), max(1, int(6 * scale)))
    pl = r - int(30 * scale)
    px = cx + int(pl * math.cos(state.wheel_angle))
    py = cy + int(pl * math.sin(state.wheel_angle))
    pygame.draw.line(surface, white, (cx, cy), (px, py), max(1, int(3 * scale)))
    pygame.draw.circle(surface, white, (px, py), max(2, int(6 * scale)))
    pygame.draw.circle(surface, white, (cx, cy), max(2, int(8 * scale)))
    font = get_font(max(18, int(40 * scale)))
    t = font.render(f"GUI {state.gui_scale:.2f}x", True, white)
    tr = t.get_rect(center=(cx, cy + r + int(44 * scale)))
    bg = pygame.Rect(tr.x - int(10 * scale), tr.y - int(5 * scale),
                     tr.width + int(20 * scale), tr.height + int(10 * scale))
    pygame.draw.rect(surface, (20, 20, 20), bg)
    pygame.draw.rect(surface, white, bg, max(1, int(2 * scale)))
    surface.blit(t, tr)
