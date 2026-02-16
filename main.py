import cv2
import numpy as np
import math
import pygame
import sys
import time
import threading

from shared import (
    mp_hands, clamp, get_font,
    HandState,
    is_pinching_gesture, get_pinch_position,
    detect_three_finger_gesture,
    get_hand_center, calculate_finger_angle, lm_to_screen, draw_wheel,
)

# ==============================
# Carousel config
# ==============================
CARD_COUNT = 7
CARD_WIDTH = 280
CARD_HEIGHT = 280
CARD_SPACING = 50
ROW_BASE_SPACING = CARD_HEIGHT + 80

CAROUSEL_CATEGORIES = [
    ["Mail", "Music", "Browser", "Messages", "Calendar", "Maps", "Camera"],
    ["Photos", "Notes", "Reminders", "Clock", "Weather", "Stocks", "News"],
    ["YouTube", "Netflix", "Twitch", "Spotify", "Podcasts", "Books", "Games"]
]
NUM_CATEGORIES = len(CAROUSEL_CATEGORIES)

APP_COLORS = {
    "Mail": (74, 144, 226), "Music": (252, 61, 86), "Safari": (35, 142, 250),
    "Messages": (76, 217, 100), "Calendar": (252, 61, 57), "Maps": (89, 199, 249),
    "Camera": (138, 138, 142), "Photos": (252, 203, 47), "Notes": (255, 214, 10),
    "Reminders": (255, 69, 58), "Clock": (30, 30, 30), "Weather": (99, 204, 250),
    "Stocks": (30, 30, 30), "News": (252, 61, 86), "YouTube": (255, 0, 0),
    "Netflix": (229, 9, 20), "Twitch": (145, 70, 255), "Spotify": (30, 215, 96),
    "Podcasts": (146, 72, 223), "Books": (255, 124, 45), "Games": (255, 45, 85),
    "Browser": (35, 142, 250)
}


# ==============================
# Carousel drawing
# ==============================
def draw_app_icon(surface, app_name, x, y, base_w, base_h, is_selected=False, zoom_scale=1.0, gui_scale=1.0):
    width = int(base_w * gui_scale)
    height = int(base_h * gui_scale)
    if is_selected:
        width = int(width * zoom_scale)
        height = int(height * zoom_scale)
    br = max(12, int(50 * gui_scale))
    rect = pygame.Rect(x - width // 2, y - height // 2, width, height)
    color = tuple(min(255, int(APP_COLORS.get(app_name, (100, 100, 100))[i] * 1.2)) for i in range(3))
    pygame.draw.rect(surface, color, rect, border_radius=br)
    if is_selected:
        sel = pygame.Rect(rect.x - int(6 * gui_scale), rect.y - int(6 * gui_scale),
                          rect.width + int(12 * gui_scale), rect.height + int(12 * gui_scale))
        pygame.draw.rect(surface, (255, 255, 255), sel, width=max(2, int(8 * gui_scale)), border_radius=br)
    icon_size = max(24, int(120 * (width / max(1, int(base_w * gui_scale)))))
    icon_img = get_font(icon_size).render(app_name[0], True, (255, 255, 255, 180))
    surface.blit(icon_img, icon_img.get_rect(center=(x, y - int(20 * gui_scale))))
    text_size = max(12, int(36 * (width / max(1, int(base_w * gui_scale)))))
    text_img = get_font(text_size).render(app_name, True, (255, 255, 255))
    surface.blit(text_img, text_img.get_rect(center=(x, y + int(60 * gui_scale))))
    return rect


def draw_cards(surface, center_x, center_y, card_offset, category_idx,
               selected_card=None, selected_category=None, zoom_progress=0.0,
               window_width=1280, gui_scale=1.0, base_w=280, base_h=280, base_spacing=50):
    app_names = CAROUSEL_CATEGORIES[category_idx]
    card_rects = []
    sw = int(base_w * gui_scale)
    ss = int(base_spacing * gui_scale)
    stride = sw + ss
    first_vis = int((-card_offset - window_width // 2) / stride) - 1
    last_vis = int((-card_offset + window_width // 2) / stride) + 2
    first_vis = max(0, first_vis)
    last_vis = min(CARD_COUNT, last_vis)
    for i in range(first_vis, last_vis):
        x = int(center_x + (i * stride) + card_offset)
        y = int(center_y)
        sel = (selected_card == i and selected_category == category_idx)
        if not sel:
            rect = draw_app_icon(surface, app_names[i], x, y, base_w, base_h, False, 1.0, gui_scale)
            card_rects.append((rect, i, category_idx))
    for i in range(first_vis, last_vis):
        x = int(center_x + (i * stride) + card_offset)
        y = int(center_y)
        sel = (selected_card == i and selected_category == category_idx)
        if sel:
            rect = draw_app_icon(surface, app_names[i], x, y, base_w, base_h, True, 1.0 + (zoom_progress * 0.3), gui_scale)
            card_rects.append((rect, i, category_idx))
    return card_rects


# ==============================
# Carousel main
# ==============================
def carousel_main():
    pygame.init()
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Gesture Carousel")
    clock = pygame.time.Clock()

    print("=" * 50)
    print("GESTURE CAROUSEL STARTED")
    print("Pinch to select • Hold pinch to scroll • Three-finger wheel to zoom")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=0)

    state = HandState()
    tap_to_check = None
    double_pinch_to_check = None
    running = True

    # --- Performance: threaded camera + detection ---
    _font_status = pygame.font.Font(None, 48)
    _latest_results = [None]  # shared between threads
    _latest_rgb = [None]
    _detection_lock = threading.Lock()

    def _camera_detection_loop():
        """Runs in background thread: captures frames and runs MediaPipe."""
        while running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            with _detection_lock:
                _latest_results[0] = results
                _latest_rgb[0] = rgb
            time.sleep(0.001)  # yield to main thread

    _cam_thread = threading.Thread(target=_camera_detection_loop, daemon=True)
    _cam_thread.start()
    _cam_frame_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # --- Grab latest detection results from background thread ---
        with _detection_lock:
            results = _latest_results[0]
            rgb = _latest_rgb[0]
        if results is None or rgb is None:
            pygame.display.flip()
            clock.tick(60)
            continue
        _cam_frame_count += 1

        fps = clock.get_fps()
        if fps > 0:
            state.current_fps = (0.9 * state.current_fps + 0.1 * fps) if state.current_fps > 0 else fps

        right_hand = None
        if results.multi_hand_landmarks:
            # Use the first detected hand — handedness labels are unreliable
            # especially at screen edges and with mirrored frames
            right_hand = results.multi_hand_landmarks[0].landmark

        if right_hand is None:
            state.finger_smoother.reset()
            state.wheel_active = False
            state.last_finger_angle = None

        pinch_now = is_pinching_gesture(right_hand, state.pinch_threshold) if right_hand else False

        # Three-finger wheel gesture (zoom)
        if right_hand and detect_three_finger_gesture(right_hand):
            if not state.wheel_active:
                hc = get_hand_center(right_hand)
                state.wheel_active = True
                state.wheel_center_x = int(hc.x * WINDOW_WIDTH)
                state.wheel_center_y = int(hc.y * WINDOW_HEIGHT)
                state.last_finger_angle = None
            ang = calculate_finger_angle(right_hand)
            if state.last_finger_angle is not None:
                diff = ang - state.last_finger_angle
                if diff > math.pi:
                    diff -= 2 * math.pi
                elif diff < -math.pi:
                    diff += 2 * math.pi
                state.wheel_angle = (state.wheel_angle + diff * 2) % (2 * math.pi)
                state.gui_scale = clamp(
                    state.gui_scale + diff * state.gui_scale_sensitivity,
                    state.gui_scale_min, state.gui_scale_max
                )
            state.last_finger_angle = ang
        else:
            state.wheel_active = False
            state.last_finger_angle = None

        # Pinch handling: distinguish scroll from selection
        if right_hand and not state.wheel_active:
            pos = get_pinch_position(right_hand)
            
            # Pinch just started
            if pinch_now and not state.pinch_prev:
                if pos:
                    px = pos[0] * WINDOW_WIDTH
                    py = pos[1] * WINDOW_HEIGHT
                    state.pinch_start_pos = (px, py)
                    state.last_pinch_x = px
                    state.last_pinch_y = py
                    state.is_pinching = True
                    state.pinch_hold_start = time.time()
                    state.scroll_unlocked = False

            # Pinch continuing
            elif pinch_now and state.pinch_prev and pos:
                px = pos[0] * WINDOW_WIDTH
                py = pos[1] * WINDOW_HEIGHT
                if state.last_pinch_x is not None:
                    dx = px - state.last_pinch_x
                    dy = py - state.last_pinch_y
                    
                    # Unlock scrolling after holding pinch for 1.5 seconds
                    if not state.scroll_unlocked:
                        if time.time() - state.pinch_hold_start >= state.pinch_hold_delay:
                            state.scroll_unlocked = True
                            print("✓ Scroll unlocked — drag to scroll")
                    
                    # Only scroll if unlocked (held long enough)
                    if state.scroll_unlocked:
                        if state.pinch_start_pos:
                            total_dx = px - state.pinch_start_pos[0]
                            total_dy = py - state.pinch_start_pos[1]
                            total_move = math.hypot(total_dx, total_dy)
                            
                            if total_move > state.movement_threshold:
                                state.card_offset += dx * state.scroll_gain
                                state.category_offset += dy * state.scroll_gain
                                stride_x = int((CARD_WIDTH + CARD_SPACING) * state.gui_scale)
                                min_x = -(CARD_COUNT - 1) * stride_x
                                state.card_offset = clamp(state.card_offset, min_x, 0)
                                row_stride = int(ROW_BASE_SPACING * state.gui_scale)
                                min_y = -(NUM_CATEGORIES - 1) * row_stride
                                state.category_offset = clamp(state.category_offset, min_y, 0)
                
                state.last_pinch_x = px
                state.last_pinch_y = py

            # Pinch released
            elif not pinch_now and state.pinch_prev:
                if state.pinch_start_pos and state.last_pinch_x is not None:
                    total_dx = state.last_pinch_x - state.pinch_start_pos[0]
                    total_dy = state.last_pinch_y - state.pinch_start_pos[1]
                    total_move = math.hypot(total_dx, total_dy)

                    current_time = time.time()
                    dt = current_time - state.last_pinch_time

                    # If minimal movement and scroll was NOT unlocked, this is a SELECT action
                    if total_move <= state.movement_threshold and not state.scroll_unlocked:
                        tap_to_check = (state.last_pinch_x, state.last_pinch_y)
                        
                        # Check for double-pinch (for launch)
                        if 0.05 < dt < state.double_pinch_window:
                            double_pinch_to_check = (state.last_pinch_x, state.last_pinch_y)
                            print("✓ Double pinch detected — will launch card under finger")

                    state.last_pinch_time = current_time

                state.is_pinching = False
                state.last_pinch_x = None
                state.last_pinch_y = None
                state.pinch_start_pos = None
                state.scroll_unlocked = False
        else:
            state.is_pinching = False
            state.last_pinch_x = None
            state.last_pinch_y = None
            state.pinch_start_pos = None
            state.scroll_unlocked = False

        state.pinch_prev = pinch_now

        # Smooth offsets
        s = state.scroll_smoothing
        state.smooth_card_offset += (state.card_offset - state.smooth_card_offset) * s
        state.smooth_category_offset += (state.category_offset - state.smooth_category_offset) * s

        # Draw carousel
        screen.fill((20, 20, 30))
        cx = WINDOW_WIDTH // 2
        cy = WINDOW_HEIGHT // 2
        all_rects = []
        row_stride = int(ROW_BASE_SPACING * state.gui_scale)
        first_cat = max(0, int(-state.smooth_category_offset / row_stride) - 1)
        last_cat = min(NUM_CATEGORIES, int((-state.smooth_category_offset + WINDOW_HEIGHT) / row_stride) + 2)
        for cat_idx in range(first_cat, last_cat):
            y = cy + (cat_idx * row_stride) + state.smooth_category_offset
            all_rects += draw_cards(
                screen, cx, int(y), state.smooth_card_offset, cat_idx,
                state.selected_card, state.selected_category, state.zoom_progress,
                WINDOW_WIDTH, state.gui_scale, CARD_WIDTH, CARD_HEIGHT, CARD_SPACING
            )

        # Resolve single pinch tap
        if tap_to_check:
            tx, ty = tap_to_check
            hit = False
            for rect, ci, ca in all_rects:
                if rect.collidepoint(tx, ty):
                    app_name = CAROUSEL_CATEGORIES[ca][ci]
                    state.selected_card = ci
                    state.selected_category = ca
                    state.zoom_target = 1.0
                    print(f"✓ Selected: {app_name} (card {ci}, category {ca})")
                    hit = True
                    break
            if not hit:
                print("Pinch didn't hit a card")
            tap_to_check = None

        # Resolve double-pinch
        if double_pinch_to_check:
            dx, dy = double_pinch_to_check
            for rect, ci, ca in all_rects:
                if rect.collidepoint(dx, dy):
                    app_name = CAROUSEL_CATEGORIES[ca][ci]
                    print(f"✓✓ Double pinch on {app_name}")
                    break
            double_pinch_to_check = None

        # Wheel overlay
        draw_wheel(screen, state, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Hand HUD
        if right_hand:
            tt = right_hand[4]
            it = right_hand[8]
            (tx, ty), (ix, iy) = state.finger_smoother.update(
                lm_to_screen(tt, WINDOW_WIDTH, WINDOW_HEIGHT),
                lm_to_screen(it, WINDOW_WIDTH, WINDOW_HEIGHT)
            )
            if not state.wheel_active and pinch_now:
                pygame.draw.line(screen, (255, 255, 255), (int(tx), int(ty)), (int(ix), int(iy)), 2)
            pygame.draw.circle(screen, (255, 255, 255), (int(tx), int(ty)), 8)
            pygame.draw.circle(screen, (255, 255, 255), (int(ix), int(iy)), 8)
        else:
            state.finger_smoother.reset()

        # Status text
        if state.wheel_active:
            status = f"WHEEL • GUI {state.gui_scale:.2f}x"
        elif state.is_pinching:
            if state.scroll_unlocked:
                status = "SCROLLING"
            else:
                hold_elapsed = time.time() - state.pinch_hold_start
                remaining = max(0, state.pinch_hold_delay - hold_elapsed)
                status = f"PINCHED • hold {remaining:.1f}s to scroll"
        else:
            status = "Ready • Pinch to select • Hold pinch to scroll"
        screen.blit(_font_status.render(status, True, (255, 255, 255)), (30, 30))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    sys.exit()


# ==============================
# Entrypoint
# ==============================
if __name__ == "__main__":
    carousel_main()

