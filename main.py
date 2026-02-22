"""
Gesture Carousel – clean rewrite with anchor-based scrolling.

Scroll model:
  When you pinch, we record WHERE you grabbed (anchor) and what the scroll
  offset was at that moment (offset_at_grab).  Every frame while pinching,
  the scroll *target* is simply:
      target = offset_at_grab  +  (filtered_finger − anchor) * gain
  Then a smooth interpolation chases that target for display.

  This means MediaPipe landmark noise never accumulates — it only causes
  tiny wobble around the correct position, which the chase-smoothing absorbs.

Finger positions are cleaned with a One-Euro filter (adaptive low-pass):
  smooth at rest, responsive when moving.
"""

import cv2
import numpy as np
import math
import pygame
import sys
import time
import threading

from shared import (
    mp_hands, clamp, get_font,
    OneEuroFilter2D,
    is_pinching, thumb_pos,
    detect_three_finger_gesture,
    get_hand_center, calculate_finger_angle,
    lm_to_screen, draw_wheel,
)

# ==============================
# Carousel layout
# ==============================
CARD_COUNT = 7
CARD_W = 280
CARD_H = 280
CARD_GAP = 50
ROW_H = CARD_H + 80

CATEGORIES = [
    ["Mail", "Music", "Browser", "Messages", "Calendar", "Maps", "Camera"],
    ["Photos", "Notes", "Reminders", "Clock", "Weather", "Stocks", "News"],
    ["YouTube", "Netflix", "Twitch", "Spotify", "Podcasts", "Books", "Games"],
]
NUM_ROWS = len(CATEGORIES)

COLORS = {
    "Mail": (74, 144, 226), "Music": (252, 61, 86), "Browser": (35, 142, 250),
    "Messages": (76, 217, 100), "Calendar": (252, 61, 57), "Maps": (89, 199, 249),
    "Camera": (138, 138, 142), "Photos": (252, 203, 47), "Notes": (255, 214, 10),
    "Reminders": (255, 69, 58), "Clock": (30, 30, 30), "Weather": (99, 204, 250),
    "Stocks": (30, 30, 30), "News": (252, 61, 86), "YouTube": (255, 0, 0),
    "Netflix": (229, 9, 20), "Twitch": (145, 70, 255), "Spotify": (30, 215, 96),
    "Podcasts": (146, 72, 223), "Books": (255, 124, 45), "Games": (255, 45, 85),
}

W, H = 1280, 720
PINCH_THRESH = 0.08
SCROLL_GAIN = 4.0
MOVE_THRESH = 10        # px — distinguishes tap from scroll
CHASE_SPEED = 0.25      # how fast display chases target (0–1, lower = smoother)


# ==============================
# Drawing
# ==============================
def draw_icon(surf, name, x, y, scale=1.0, selected=False):
    w = int(CARD_W * scale)
    h = int(CARD_H * scale)
    br = max(12, int(50 * scale))
    r = pygame.Rect(x - w // 2, y - h // 2, w, h)
    c = tuple(min(255, int(COLORS.get(name, (100, 100, 100))[i] * 1.2))
              for i in range(3))
    pygame.draw.rect(surf, c, r, border_radius=br)
    if selected:
        s = pygame.Rect(r.x - int(6 * scale), r.y - int(6 * scale),
                        r.w + int(12 * scale), r.h + int(12 * scale))
        pygame.draw.rect(surf, (255, 255, 255), s,
                         width=max(2, int(8 * scale)), border_radius=br)
    isz = max(24, int(120 * scale))
    surf.blit(get_font(isz).render(name[0], True, (255, 255, 255, 180)),
              get_font(isz).render(name[0], True, (255, 255, 255, 180))
              .get_rect(center=(x, y - int(20 * scale))))
    tsz = max(12, int(36 * scale))
    txt = get_font(tsz).render(name, True, (255, 255, 255))
    surf.blit(txt, txt.get_rect(center=(x, y + int(60 * scale))))
    return r


def draw_row(surf, cx, cy, offset_x, row_idx, sel_card, sel_row, scale):
    names = CATEGORIES[row_idx]
    rects = []
    sw = int(CARD_W * scale)
    sg = int(CARD_GAP * scale)
    stride = sw + sg
    for i in range(CARD_COUNT):
        x = int(cx + i * stride + offset_x)
        if x + sw // 2 < -200 or x - sw // 2 > W + 200:
            continue   # off-screen cull
        sel = (sel_card == i and sel_row == row_idx)
        r = draw_icon(surf, names[i], x, int(cy), scale, sel)
        rects.append((r, i, row_idx))
    return rects


# ==============================
# Main
# ==============================
def main():
    pygame.init()

    from pygame._sdl2 import (Window as SDL2Win, Renderer as SDL2Ren,
                               Texture as SDL2Tex)
    win = SDL2Win("Gesture Carousel", size=(W, H))
    ren = SDL2Ren(win, vsync=True)
    screen = pygame.Surface((W, H))
    clock = pygame.time.Clock()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no camera"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3,
                           min_tracking_confidence=0.3, model_complexity=0)

    # --- state ---
    gui_scale = 1.0
    gui_scale_min, gui_scale_max = 0.6, 1.8
    gui_scale_sens = 0.32
    wheel_active = False
    wheel_angle = math.pi
    wheel_cx = wheel_cy = 0
    wheel_radius = 110
    last_fing_angle = None

    sel_card = sel_row = None          # selected card

    # Scroll state — anchor model
    target_x = 0.0                     # where we WANT the cards to be
    target_y = 0.0
    display_x = 0.0                    # where the cards ARE on screen (chases target)
    display_y = 0.0
    grab_offset_x = 0.0               # scroll offset when pinch started
    grab_offset_y = 0.0
    anchor_x = anchor_y = 0.0         # filtered finger pos when pinch started (px)
    is_scrolling = False               # did we pass the move threshold?
    pinch_prev = False

    # One-Euro filter for the pinch finger position (in pixels)
    pinch_filter = OneEuroFilter2D(min_cutoff=1.5, beta=0.01)

    # Separate One-Euro filter for the cursor dots (cosmetic)
    thumb_filt = OneEuroFilter2D(min_cutoff=1.0, beta=0.005)
    index_filt = OneEuroFilter2D(min_cutoff=1.0, beta=0.005)

    running = True

    # --- threaded camera + detection ---
    cam_surf = pygame.Surface((320, 240))
    latest = [None, None]  # [results, rgb]
    lock = threading.Lock()

    def det_loop():
        while running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005); continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            with lock:
                latest[0] = res; latest[1] = rgb
            time.sleep(0.001)

    threading.Thread(target=det_loop, daemon=True).start()

    # --- helper to build a simple state object for draw_wheel ---
    class WheelState:
        pass

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        with lock:
            results, rgb = latest

        if results is None or rgb is None:
            ren.clear(); ren.present(); clock.tick(0); continue

        now = time.monotonic()

        hand = None
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0].landmark

        if hand is None:
            pinch_filter.reset()
            thumb_filt.reset()
            index_filt.reset()
            wheel_active = False
            last_fing_angle = None

        pinch_now = is_pinching(hand, PINCH_THRESH) if hand else False

        # ---- Three-finger wheel (zoom) ----
        if hand and detect_three_finger_gesture(hand):
            if not wheel_active:
                hc = get_hand_center(hand)
                wheel_active = True
                wheel_cx = int(hc.x * W)
                wheel_cy = int(hc.y * H)
                last_fing_angle = None
            ang = calculate_finger_angle(hand)
            if last_fing_angle is not None:
                diff = ang - last_fing_angle
                if diff > math.pi: diff -= 2 * math.pi
                elif diff < -math.pi: diff += 2 * math.pi
                wheel_angle = (wheel_angle + diff * 2) % (2 * math.pi)
                gui_scale = clamp(gui_scale + diff * gui_scale_sens,
                                  gui_scale_min, gui_scale_max)
            last_fing_angle = ang
        else:
            wheel_active = False
            last_fing_angle = None

        # ---- Pinch scroll (anchor model) ----
        tap_pos = None
        if hand and not wheel_active:
            pos = thumb_pos(hand)
            if pos:
                raw_px = pos[0] * W
                raw_py = pos[1] * H
                filt_px, filt_py = pinch_filter(raw_px, raw_py, now)

                if pinch_now and not pinch_prev:
                    # --- grab ---
                    anchor_x, anchor_y = filt_px, filt_py
                    grab_offset_x = target_x
                    grab_offset_y = target_y
                    is_scrolling = False

                elif pinch_now and pinch_prev:
                    # --- drag ---
                    moved = math.hypot(filt_px - anchor_x, filt_py - anchor_y)
                    if moved > MOVE_THRESH:
                        is_scrolling = True

                    if is_scrolling:
                        # Scroll target = where it was + how far finger moved
                        target_x = grab_offset_x + (filt_px - anchor_x) * SCROLL_GAIN
                        target_y = grab_offset_y + (filt_py - anchor_y) * SCROLL_GAIN
                        # Clamp to valid range
                        stride_x = int((CARD_W + CARD_GAP) * gui_scale)
                        target_x = clamp(target_x, -(CARD_COUNT - 1) * stride_x, 0)
                        row_stride = int(ROW_H * gui_scale)
                        target_y = clamp(target_y, -(NUM_ROWS - 1) * row_stride, 0)

                elif not pinch_now and pinch_prev:
                    # --- release ---
                    if not is_scrolling:
                        tap_pos = (filt_px, filt_py)
                    is_scrolling = False
        else:
            is_scrolling = False

        pinch_prev = pinch_now

        # ---- Chase smoothing: display chases target ----
        display_x += (target_x - display_x) * CHASE_SPEED
        display_y += (target_y - display_y) * CHASE_SPEED
        # Snap when close to stop sub-pixel shimmer
        if abs(target_x - display_x) < 0.3:
            display_x = target_x
        if abs(target_y - display_y) < 0.3:
            display_y = target_y

        # ---- Draw ----
        screen.fill((20, 20, 30))
        cx = W // 2
        cy = H // 2
        all_rects = []
        row_stride = int(ROW_H * gui_scale)

        for ri in range(NUM_ROWS):
            ry = cy + ri * row_stride + round(display_y)
            all_rects += draw_row(screen, cx, ry, round(display_x),
                                  ri, sel_card, sel_row, gui_scale)

        # Resolve tap
        if tap_pos is not None:
            for rect, ci, ri in all_rects:
                if rect.collidepoint(tap_pos):
                    sel_card, sel_row = ci, ri
                    print(f"Selected: {CATEGORIES[ri][ci]}")
                    break

        # Wheel overlay
        ws = WheelState()
        ws.wheel_active = wheel_active
        ws.gui_scale = gui_scale
        ws.wheel_center_x = wheel_cx
        ws.wheel_center_y = wheel_cy
        ws.wheel_radius = wheel_radius
        ws.wheel_angle = wheel_angle
        draw_wheel(screen, ws)

        # Camera preview
        small = cv2.resize(rgb, (320, 240))
        pygame.surfarray.blit_array(cam_surf, np.transpose(small, (1, 0, 2)))
        screen.blit(cam_surf, (W - 330, 10))

        # Cursor dots
        if hand:
            tt = hand[4]
            it = hand[8]
            tx, ty = thumb_filt(tt.x * W, tt.y * H, now)
            ix, iy = index_filt(it.x * W, it.y * H, now)
            if not wheel_active and pinch_now:
                pygame.draw.line(screen, (255, 255, 255),
                                 (int(tx), int(ty)), (int(ix), int(iy)), 2)
            pygame.draw.circle(screen, (255, 255, 255), (int(tx), int(ty)), 12)
            pygame.draw.circle(screen, (200, 200, 200), (int(ix), int(iy)), 5)

        # GPU present
        tex = SDL2Tex.from_surface(ren, screen)
        ren.clear()
        tex.draw()
        ren.present()
        del tex

        clock.tick(0)

    cap.release()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
