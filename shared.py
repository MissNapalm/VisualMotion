"""
Shared utilities – MediaPipe wrapper, gesture detection, drawing helpers, state.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import sys
import os

# --- MediaPipe Hands compatibility wrapper (tasks API v0.10+) ---
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.core import base_options as mp_base_options


def _find_hand_landmarker_model():
    for p in sys.path:
        candidate = os.path.join(p, 'mediapipe', 'modules',
                                 'hand_landmark', 'hand_landmarker.task')
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        'Could not find hand_landmarker.task model in mediapipe package.')


class MPHandsCompat:
    """Wraps the new MediaPipe HandLandmarker tasks API."""

    def __init__(self, max_num_hands=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, model_complexity=0):
        model_path = _find_hand_landmarker_model()
        options = HandLandmarkerOptions(
            base_options=mp_base_options.BaseOptions(
                model_asset_path=model_path),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = HandLandmarker.create_from_options(options)

    def process(self, image):
        mp_img = mp_image.Image(
            image_format=mp_image.ImageFormat.SRGB, data=image)
        result = self.detector.detect(mp_img)

        class Results:
            pass

        results = Results()
        results.multi_hand_landmarks = []
        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                class Landmark:
                    def __init__(self, x, y, z):
                        self.x, self.y, self.z = x, y, z

                hand = type('HandLandmarks', (), {})()
                hand.landmark = [Landmark(l.x, l.y, l.z) for l in hand_lms]
                results.multi_hand_landmarks.append(hand)
        return results


mp_hands = type('mp_hands', (), {'Hands': MPHandsCompat})


# ==============================
# Utility
# ==============================
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


_font_cache = {}


def get_font(size):
    if size not in _font_cache:
        _font_cache[size] = pygame.font.Font(None, size)
    return _font_cache[size]


# ==============================
# One-Euro Filter  —  the gold-standard for pointer smoothing.
# Provides smooth tracking at rest AND responsive tracking during movement,
# which is exactly what no combination of exponential filters can do.
# ==============================
class OneEuroFilter:
    """1-D One-Euro filter (low-pass with adaptive cutoff)."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff   # minimum cutoff freq (Hz) – lower = smoother at rest
        self.beta = beta               # speed coefficient – higher = less lag when moving
        self.d_cutoff = d_cutoff       # cutoff for derivative filter
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None):
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t if t is not None else 0.0
            return x

        if t is None:
            t = self.t_prev + 1.0 / 60.0   # assume 60 fps
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1.0 / 60.0
        self.t_prev = t

        # derivative (speed)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class OneEuroFilter2D:
    """Convenience wrapper: two 1-D One-Euro filters for (x, y)."""

    def __init__(self, **kwargs):
        self.fx = OneEuroFilter(**kwargs)
        self.fy = OneEuroFilter(**kwargs)

    def __call__(self, x, y, t=None):
        return self.fx(x, t), self.fy(y, t)

    def reset(self):
        self.fx.reset()
        self.fy.reset()


# ==============================
# Gesture detection
# ==============================
def get_pinch_distance(landmarks):
    if not landmarks:
        return None
    a, b = landmarks[4], landmarks[8]
    return math.hypot(a.x - b.x, a.y - b.y)


def is_pinching(landmarks, thresh=0.08):
    d = get_pinch_distance(landmarks)
    return d is not None and d < thresh


def thumb_pos(landmarks):
    """Return normalised (x, y) of thumb tip."""
    if not landmarks:
        return None
    return (landmarks[4].x, landmarks[4].y)


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
    return (thumb_ext and index_ext and middle_ext
            and ring_fold and pinky_fold)


def get_hand_center(landmarks):
    return landmarks[9]


def calculate_finger_angle(landmarks):
    c = get_hand_center(landmarks)
    idx = landmarks[8]
    return math.atan2(idx.y - c.y, idx.x - c.x)


def lm_to_screen(lm, W, H):
    return (lm.x * W, lm.y * H)


# ==============================
# Wheel drawing
# ==============================
def draw_wheel(surface, state):
    if not state.wheel_active:
        return
    scale = state.gui_scale
    cx, cy = state.wheel_center_x, state.wheel_center_y
    r = int(state.wheel_radius * scale)
    white = (255, 255, 255)

    margin = r + int(80 * scale)
    glow = pygame.Surface((margin * 2, margin * 2), pygame.SRCALPHA)
    for i in range(5):
        rr = r + int(15 * scale) + i * int(10 * scale)
        pygame.draw.circle(glow, (*white, 100 - i * 20),
                           (margin, margin), rr, max(1, int(2 * scale)))
    surface.blit(glow, (cx - margin, cy - margin))

    pygame.draw.circle(surface, white, (cx, cy), r, max(1, int(4 * scale)))
    pygame.draw.circle(surface, white, (cx, cy),
                       r - int(20 * scale), max(1, int(2 * scale)))

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
        pygame.draw.line(surface, white, (sx, sy), (ex, ey),
                         max(1, int(6 * scale)))

    pl = r - int(30 * scale)
    px = cx + int(pl * math.cos(state.wheel_angle))
    py = cy + int(pl * math.sin(state.wheel_angle))
    pygame.draw.line(surface, white, (cx, cy), (px, py),
                     max(1, int(3 * scale)))
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
