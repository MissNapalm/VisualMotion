"""
Application state — configuration constants, FingerSmoother, and HandState.
"""

import math
from collections import deque


# ==============================
# Carousel layout
# ==============================
CARD_COUNT = 7
CARD_WIDTH = 280
CARD_HEIGHT = 280
CARD_SPACING = 50
ROW_BASE_SPACING = CARD_HEIGHT + 80

CATEGORIES = [
    ["Mail", "Music", "Browser", "Messages", "Calendar", "Maps", "Camera"],
    ["Photos", "Notes", "Reminders", "Clock", "Weather", "Stocks", "News"],
    ["YouTube", "Netflix", "Twitch", "Spotify", "Podcasts", "Books", "Games"],
]
NUM_CATEGORIES = len(CATEGORIES)

APP_COLORS = {
    "Mail": (74, 144, 226),   "Music": (252, 61, 86),    "Safari": (35, 142, 250),
    "Messages": (76, 217, 100), "Calendar": (252, 61, 57), "Maps": (89, 199, 249),
    "Camera": (138, 138, 142), "Photos": (252, 203, 47),  "Notes": (255, 214, 10),
    "Reminders": (255, 69, 58), "Clock": (30, 30, 30),    "Weather": (99, 204, 250),
    "Stocks": (30, 30, 30),    "News": (252, 61, 86),     "YouTube": (255, 0, 0),
    "Netflix": (229, 9, 20),   "Twitch": (145, 70, 255),  "Spotify": (30, 215, 96),
    "Podcasts": (146, 72, 223), "Books": (255, 124, 45),  "Games": (255, 45, 85),
    "Browser": (35, 142, 250),
}

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


# ==============================
# Finger smoother
# ==============================
class FingerSmoother:
    def __init__(self, window_size=5):
        self._thumb = deque(maxlen=window_size)
        self._index = deque(maxlen=window_size)

    def update(self, thumb_pos, index_pos):
        self._thumb.append(thumb_pos)
        self._index.append(index_pos)
        tx = sum(p[0] for p in self._thumb) / len(self._thumb)
        ty = sum(p[1] for p in self._thumb) / len(self._thumb)
        ix = sum(p[0] for p in self._index) / len(self._index)
        iy = sum(p[1] for p in self._index) / len(self._index)
        return (tx, ty), (ix, iy)

    def reset(self):
        self._thumb.clear()
        self._index.clear()


# ==============================
# Hand state
# ==============================
class HandState:
    def __init__(self):
        # scroll offsets
        self.card_offset = 0.0
        self.category_offset = 0.0
        self.smooth_card_offset = 0.0
        self.smooth_category_offset = 0.0
        self.scroll_smoothing = 0.25
        self.scroll_gain = 5.0

        # pinch state
        self.is_pinching = False
        self.last_pinch_x = None
        self.last_pinch_y = None
        self.pinch_start_pos = None
        self.movement_threshold = 10
        self.pinch_threshold = 0.08
        self.pinch_prev = False
        self.last_pinch_time = 0
        self.double_pinch_window = 0.4
        self.pinch_hold_start = 0
        self.scroll_unlocked = False
        self.pinch_hold_delay = 0.45

        # selection
        self.selected_card = None
        self.selected_category = None
        self.zoom_progress = 0.0
        self.zoom_target = 0.0
        self.finger_smoother = FingerSmoother()

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

        # misc
        self.current_fps = 0.0

    def reset_pinch(self):
        self.is_pinching = False
        self.last_pinch_x = None
        self.last_pinch_y = None
        self.pinch_start_pos = None
        self.scroll_unlocked = False
