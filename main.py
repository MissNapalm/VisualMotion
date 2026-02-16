"""
Gesture Carousel - main application loop.
"""

import math
import time

import pygame

from hand_tracker import HandTracker
from gestures import (
    is_pinching, pinch_position, is_three_finger,
    hand_center, finger_angle, lm_to_screen,
)
from state import (
    HandState, CARD_COUNT, CARD_WIDTH, CARD_HEIGHT, CARD_SPACING,
    ROW_BASE_SPACING, CATEGORIES, NUM_CATEGORIES, WINDOW_WIDTH, WINDOW_HEIGHT,
)
from renderer import clamp, draw_cards, draw_wheel


class App:
    """Top-level gesture carousel application."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Gesture Carousel")
        self.clock = pygame.time.Clock()
        self.state = HandState()
        self.tracker = HandTracker()
        self._font_status = pygame.font.Font(None, 48)
        self._tap = None
        self._double_tap = None

    def _process_wheel(self, hand):
        st = self.state
        if not is_three_finger(hand):
            st.wheel_active = False
            st.last_finger_angle = None
            return
        if not st.wheel_active:
            hc = hand_center(hand)
            st.wheel_active = True
            st.wheel_center_x = int(hc.x * WINDOW_WIDTH)
            st.wheel_center_y = int(hc.y * WINDOW_HEIGHT)
            st.last_finger_angle = None
        ang = finger_angle(hand)
        if st.last_finger_angle is not None:
            diff = ang - st.last_finger_angle
            if diff > math.pi:
                diff -= 2 * math.pi
            elif diff < -math.pi:
                diff += 2 * math.pi
            st.wheel_angle = (st.wheel_angle + diff * 2) % (2 * math.pi)
            st.gui_scale = clamp(
                st.gui_scale + diff * st.gui_scale_sensitivity,
                st.gui_scale_min, st.gui_scale_max,
            )
        st.last_finger_angle = ang

    def _process_pinch(self, hand, pinch_now):
        st = self.state
        if st.wheel_active:
            st.reset_pinch()
            return
        pos = pinch_position(hand)
        if pinch_now and not st.pinch_prev:
            if pos:
                px, py = pos[0] * WINDOW_WIDTH, pos[1] * WINDOW_HEIGHT
                st.pinch_start_pos = (px, py)
                st.last_pinch_x, st.last_pinch_y = px, py
                st.is_pinching = True
                st.pinch_hold_start = time.time()
                st.scroll_unlocked = False
        elif pinch_now and st.pinch_prev and pos:
            px, py = pos[0] * WINDOW_WIDTH, pos[1] * WINDOW_HEIGHT
            if st.last_pinch_x is not None:
                dx, dy = px - st.last_pinch_x, py - st.last_pinch_y
                if not st.scroll_unlocked:
                    if time.time() - st.pinch_hold_start >= st.pinch_hold_delay:
                        st.scroll_unlocked = True
                if st.scroll_unlocked and st.pinch_start_pos:
                    total = math.hypot(px - st.pinch_start_pos[0], py - st.pinch_start_pos[1])
                    if total > st.movement_threshold:
                        st.card_offset += dx * st.scroll_gain
                        st.category_offset += dy * st.scroll_gain
                        stride_x = int((CARD_WIDTH + CARD_SPACING) * st.gui_scale)
                        st.card_offset = clamp(st.card_offset, -(CARD_COUNT - 1) * stride_x, 0)
                        row_stride = int(ROW_BASE_SPACING * st.gui_scale)
                        st.category_offset = clamp(st.category_offset, -(NUM_CATEGORIES - 1) * row_stride, 0)
            st.last_pinch_x, st.last_pinch_y = px, py
        elif not pinch_now and st.pinch_prev:
            if st.pinch_start_pos and st.last_pinch_x is not None:
                total = math.hypot(
                    st.last_pinch_x - st.pinch_start_pos[0],
                    st.last_pinch_y - st.pinch_start_pos[1],
                )
                now = time.time()
                dt = now - st.last_pinch_time
                if total <= st.movement_threshold and not st.scroll_unlocked:
                    self._tap = (st.last_pinch_x, st.last_pinch_y)
                    if 0.05 < dt < st.double_pinch_window:
                        self._double_tap = (st.last_pinch_x, st.last_pinch_y)
                st.last_pinch_time = now
            st.reset_pinch()

    def _resolve_taps(self, all_rects):
        st = self.state
        if self._tap:
            tx, ty = self._tap
            for rect, ci, ca in all_rects:
                if rect.collidepoint(tx, ty):
                    name = CATEGORIES[ca][ci]
                    st.selected_card, st.selected_category = ci, ca
                    st.zoom_target = 1.0
                    print(f"Selected: {name} (card {ci}, category {ca})")
                    break
            self._tap = None
        if self._double_tap:
            dx, dy = self._double_tap
            for rect, ci, ca in all_rects:
                if rect.collidepoint(dx, dy):
                    print(f"Double pinch on {CATEGORIES[ca][ci]}")
                    break
            self._double_tap = None

    def _draw(self, hand, pinch_now):
        st = self.state
        screen = self.screen
        screen.fill((20, 20, 30))
        sm = st.scroll_smoothing
        st.smooth_card_offset += (st.card_offset - st.smooth_card_offset) * sm
        st.smooth_category_offset += (st.category_offset - st.smooth_category_offset) * sm
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        row_stride = int(ROW_BASE_SPACING * st.gui_scale)
        first = max(0, int(-st.smooth_category_offset / row_stride) - 1)
        last = min(NUM_CATEGORIES, int((-st.smooth_category_offset + WINDOW_HEIGHT) / row_stride) + 2)
        all_rects = []
        for cat in range(first, last):
            y = cy + cat * row_stride + st.smooth_category_offset
            all_rects += draw_cards(
                screen, cx, int(y), st.smooth_card_offset, cat,
                st.selected_card, st.selected_category, st.zoom_progress,
                WINDOW_WIDTH, st.gui_scale, CARD_WIDTH, CARD_HEIGHT, CARD_SPACING,
            )
        self._resolve_taps(all_rects)
        draw_wheel(screen, st, WINDOW_WIDTH, WINDOW_HEIGHT)
        if hand:
            (tx, ty), (ix, iy) = st.finger_smoother.update(
                lm_to_screen(hand[4], WINDOW_WIDTH, WINDOW_HEIGHT),
                lm_to_screen(hand[8], WINDOW_WIDTH, WINDOW_HEIGHT),
            )
            if not st.wheel_active and pinch_now:
                pygame.draw.line(screen, (255, 255, 255), (int(tx), int(ty)), (int(ix), int(iy)), 2)
            pygame.draw.circle(screen, (255, 255, 255), (int(tx), int(ty)), 8)
            pygame.draw.circle(screen, (255, 255, 255), (int(ix), int(iy)), 8)
        else:
            st.finger_smoother.reset()
        if st.wheel_active:
            status = f"WHEEL - GUI {st.gui_scale:.2f}x"
        elif st.is_pinching:
            if st.scroll_unlocked:
                status = "SCROLLING"
            else:
                remaining = max(0, st.pinch_hold_delay - (time.time() - st.pinch_hold_start))
                status = f"PINCHED - hold {remaining:.1f}s to scroll"
        else:
            status = "Ready - Pinch to select - Hold pinch to scroll"
        screen.blit(self._font_status.render(status, True, (255, 255, 255)), (30, 30))
        pygame.display.flip()

    def run(self):
        print("=" * 50)
        print("GESTURE CAROUSEL STARTED")
        print("Pinch to select | Hold pinch to scroll | Three-finger wheel to zoom")
        print("=" * 50)
        self.tracker.start()
        st = self.state
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._shutdown()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._shutdown()
                    return
            hand = self.tracker.latest()
            if hand is None:
                st.finger_smoother.reset()
                st.wheel_active = False
                st.last_finger_angle = None
                pinch_now = False
            else:
                pinch_now = is_pinching(hand, st.pinch_threshold)
                self._process_wheel(hand)
                self._process_pinch(hand, pinch_now)
            st.pinch_prev = pinch_now
            self._draw(hand, pinch_now)
            self.clock.tick(60)

    def _shutdown(self):
        self.tracker.stop()
        pygame.quit()


if __name__ == "__main__":
    App().run()
