"""
Gesture detection — pure functions that operate on a list of 21 hand landmarks.
"""

import math


def pinch_distance(landmarks):
    """Distance between thumb-tip (4) and index-tip (8)."""
    a, b = landmarks[4], landmarks[8]
    return math.hypot(a.x - b.x, a.y - b.y)


def is_pinching(landmarks, threshold=0.08):
    return pinch_distance(landmarks) < threshold


def pinch_position(landmarks):
    """Midpoint between thumb-tip and index-tip (normalised coords)."""
    a, b = landmarks[4], landmarks[8]
    return ((a.x + b.x) / 2, (a.y + b.y) / 2)


def _finger_extended(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y


def is_three_finger(landmarks):
    """Thumb + index + middle extended, ring + pinky folded."""
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    thumb_ext = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) * 0.8
    index_ext = _finger_extended(landmarks, 8, 6)
    middle_ext = _finger_extended(landmarks, 12, 10)
    ring_fold = landmarks[16].y > landmarks[14].y - 0.02
    pinky_fold = landmarks[20].y > landmarks[18].y - 0.02
    return thumb_ext and index_ext and middle_ext and ring_fold and pinky_fold


def hand_center(landmarks):
    return landmarks[9]


def finger_angle(landmarks):
    """Angle from hand centre to index-tip."""
    c = hand_center(landmarks)
    idx = landmarks[8]
    return math.atan2(idx.y - c.y, idx.x - c.x)


def lm_to_screen(lm, width, height):
    return (lm.x * width, lm.y * height)
