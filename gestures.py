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


def _remap(v, lo, hi):
    """Remap normalised coord from the [lo..hi] region to 0..1."""
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))

# Padding: asymmetric so right side isn't clipped early.
# Camera left 15%, right 8%, top 10%, bottom 10%.
_PAD_L, _PAD_R = 0.15, 0.08
_PAD_T, _PAD_B = 0.10, 0.10


def pinch_position(landmarks):
    """Thumb-tip position (the primary cursor). Remapped coords."""
    a = landmarks[4]
    rx = _remap(a.x, _PAD_L, 1.0 - _PAD_R)
    ry = _remap(a.y, _PAD_T, 1.0 - _PAD_B)
    return (rx, ry)


def _finger_extended(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y


def is_three_finger(landmarks):
    """Thumb + index + middle extended, ring + pinky clearly folded.

    Tightened thresholds to avoid false triggers at screen edges.
    """
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    thumb_ext = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) * 0.9

    index_ext = _finger_extended(landmarks, 8, 6)
    middle_ext = _finger_extended(landmarks, 12, 10)

    # Ring and pinky must be clearly curled: tip well below PIP
    ring_fold = landmarks[16].y > landmarks[14].y + 0.03
    pinky_fold = landmarks[20].y > landmarks[18].y + 0.03

    # Also reject if the hand is pinching (thumb+index close together)
    if pinch_distance(landmarks) < 0.10:
        return False

    return thumb_ext and index_ext and middle_ext and ring_fold and pinky_fold


def hand_center(landmarks):
    return landmarks[9]


def finger_angle(landmarks):
    """Angle from hand centre to index-tip."""
    c = hand_center(landmarks)
    idx = landmarks[8]
    return math.atan2(idx.y - c.y, idx.x - c.x)


def lm_to_screen(lm, width, height):
    """Map landmark to screen coords with edge padding.

    Uses the same asymmetric remap as pinch_position so cursors
    can comfortably reach all edges of the screen.
    """
    return (_remap(lm.x, _PAD_L, 1.0 - _PAD_R) * width,
            _remap(lm.y, _PAD_T, 1.0 - _PAD_B) * height)
