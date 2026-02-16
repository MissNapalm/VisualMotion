"""
Browser Window — Embedded gesture-driven web browser placeholder.
Uses Pygame to render a browser-like UI with gesture navigation.
"""

import cv2
import numpy as np
import math
import pygame
import sys
import threading
import time

from shared import (
    mp_hands, clamp, get_font,
    get_pinch_distance, is_pinching_gesture, get_pinch_position,
    detect_three_finger_gesture, detect_ok_gesture,
    calculate_finger_angle, lm_to_screen,
)


class BrowserWindow:
    """Self-contained gesture-driven browser window."""

    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 800

    # Bookmarks / quick-launch tiles
    BOOKMARKS = [
        {"name": "Google",    "url": "google.com",       "color": (66, 133, 244)},
        {"name": "YouTube",   "url": "youtube.com",      "color": (255, 0, 0)},
        {"name": "GitHub",    "url": "github.com",       "color": (36, 41, 46)},
        {"name": "Wikipedia", "url": "wikipedia.org",    "color": (51, 51, 51)},
        {"name": "Reddit",    "url": "reddit.com",       "color": (255, 69, 0)},
        {"name": "Twitter",   "url": "x.com",            "color": (29, 161, 242)},
        {"name": "Stack OF",  "url": "stackoverflow.com","color": (244, 128, 36)},
        {"name": "MDN Docs",  "url": "developer.mozilla.org", "color": (30, 30, 30)},
    ]

    def __init__(self):
        self.scroll_y = 0.0
        self.smooth_scroll_y = 0.0
        self.scroll_gain = 4.0
        self._is_pinching = False
        self.last_pinch_x = None
        self.last_pinch_y = None
        self.pinch_start_pos = None
        self.movement_threshold = 10
        self.pinch_threshold = 0.08
        self.ok_prev = False
        self.ok_touch_threshold = 0.025
        self.wheel_active = False
        self.wheel_angle = 0.0
        self.last_finger_angle = None
        self.zoom_level = 1.0
        self.zoom_min = 0.5
        self.zoom_max = 2.0
        self.zoom_sense = 0.15
        self.selected_bookmark = None
        self.current_url = ""
        self.running = False

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_tab_bar(self, screen):
        W = self.WINDOW_WIDTH
        # tab bar background
        pygame.draw.rect(screen, (50, 54, 62), (0, 0, W, 40))
        # active tab
        pygame.draw.rect(screen, (35, 38, 46), (8, 4, 200, 36), border_radius=8)
        tab_font = pygame.font.SysFont('arial', 14)
        tab_label = self.current_url if self.current_url else "New Tab"
        screen.blit(tab_font.render(tab_label, True, (220, 220, 220)), (20, 14))
        # + button
        plus_font = pygame.font.SysFont('arial', 18, bold=True)
        screen.blit(plus_font.render("+", True, (160, 165, 175)), (218, 10))

    def _draw_address_bar(self, screen):
        W = self.WINDOW_WIDTH
        # address bar background
        pygame.draw.rect(screen, (35, 38, 46), (0, 40, W, 44))
        # nav buttons
        btn_font = pygame.font.SysFont('arial', 18)
        for i, sym in enumerate(["←", "→", "⟳"]):
            bx = 16 + i * 36
            screen.blit(btn_font.render(sym, True, (160, 165, 175)), (bx, 50))
        # URL bar
        pygame.draw.rect(screen, (50, 54, 62), (130, 48, W - 160, 28), border_radius=14)
        url_font = pygame.font.SysFont('arial', 14)
        url_text = self.current_url if self.current_url else "Search or enter URL"
        url_color = (200, 200, 200) if self.current_url else (120, 125, 135)
        screen.blit(url_font.render(url_text, True, url_color), (146, 54))

    def _draw_bookmarks_bar(self, screen):
        W = self.WINDOW_WIDTH
        pygame.draw.rect(screen, (42, 46, 54), (0, 84, W, 32))
        bk_font = pygame.font.SysFont('arial', 13)
        bx = 16
        for bm in self.BOOKMARKS[:6]:
            screen.blit(bk_font.render(bm["name"], True, (180, 185, 195)), (bx, 92))
            bx += bk_font.size(bm["name"])[0] + 24

    def _draw_new_tab_page(self, screen):
        """Draw a new-tab page with bookmark tiles."""
        W, H = self.WINDOW_WIDTH, self.WINDOW_HEIGHT
        content_y = 116

        # background
        pygame.draw.rect(screen, (30, 33, 40), (0, content_y, W, H - content_y))

        # title
        title_font = pygame.font.SysFont('arial', 28)
        title = title_font.render("New Tab", True, (200, 205, 215))
        screen.blit(title, title.get_rect(center=(W // 2, content_y + 60)))

        # search bar
        search_w = min(600, W - 100)
        search_x = (W - search_w) // 2
        search_y = content_y + 100
        pygame.draw.rect(screen, (50, 54, 62), (search_x, search_y, search_w, 44), border_radius=22)
        sf = pygame.font.SysFont('arial', 16)
        screen.blit(sf.render("🔍  Search the web…", True, (120, 125, 135)),
                     (search_x + 20, search_y + 12))

        # bookmark tiles
        tile_w, tile_h = 120, 100
        cols = min(len(self.BOOKMARKS), 4)
        gap = 24
        grid_w = cols * tile_w + (cols - 1) * gap
        start_x = (W - grid_w) // 2
        start_y = search_y + 80
        tile_rects = []
        for idx, bm in enumerate(self.BOOKMARKS):
            row = idx // cols
            col = idx % cols
            tx = start_x + col * (tile_w + gap)
            ty = start_y + row * (tile_h + gap) + int(self.smooth_scroll_y * 0.3)
            rect = pygame.Rect(tx, ty, tile_w, tile_h)
            tile_rects.append((rect, idx))

            # tile background
            hover = (self.selected_bookmark == idx)
            bg_color = (60, 64, 72) if not hover else (75, 80, 90)
            pygame.draw.rect(screen, bg_color, rect, border_radius=12)

            # icon circle
            cx_c = tx + tile_w // 2
            cy_c = ty + 36
            pygame.draw.circle(screen, bm["color"], (cx_c, cy_c), 22)
            letter_font = pygame.font.SysFont('arial', 20, bold=True)
            letter = letter_font.render(bm["name"][0], True, (255, 255, 255))
            screen.blit(letter, letter.get_rect(center=(cx_c, cy_c)))

            # label
            lbl_font = pygame.font.SysFont('arial', 12)
            lbl = lbl_font.render(bm["name"], True, (190, 195, 205))
            screen.blit(lbl, lbl.get_rect(center=(cx_c, ty + tile_h - 16)))

        return tile_rects

    def _draw_page_content(self, screen):
        """Draw placeholder content when a 'page' is loaded."""
        W, H = self.WINDOW_WIDTH, self.WINDOW_HEIGHT
        content_y = 116

        pygame.draw.rect(screen, (255, 255, 255), (0, content_y, W, H - content_y))

        # page header
        bm = self.BOOKMARKS[self.selected_bookmark] if self.selected_bookmark is not None else None
        header_color = bm["color"] if bm else (66, 133, 244)
        pygame.draw.rect(screen, header_color, (0, content_y, W, 60))
        hf = pygame.font.SysFont('arial', 24, bold=True)
        name = bm["name"] if bm else "Page"
        screen.blit(hf.render(name, True, (255, 255, 255)), (24, content_y + 16))

        # fake page body
        body_y = content_y + 80 + int(self.smooth_scroll_y)
        bf = pygame.font.SysFont('arial', 16)
        lines = [
            f"Welcome to {self.current_url}",
            "",
            "This is a simulated browser page rendered inside the gesture carousel.",
            "Use pinch-drag to scroll up and down through the content.",
            "Use the three-finger gesture to zoom in and out.",
            "Make the A-OK gesture to go back to the new tab page.",
            "",
            "─" * 60,
            "",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
            "Duis aute irure dolor in reprehenderit in voluptate velit esse.",
            "Excepteur sint occaecat cupidatat non proident, sunt in culpa.",
            "",
            "Curabitur pretium tincidunt lacus. Nulla gravida orci a odio.",
            "Nullam varius, turpis et commodo pharetra, est eros bibendum elit.",
            "Praesent dapibus, neque id cursus faucibus, tortor neque egestas augue.",
            "",
            "─" * 60,
            "",
            "Section 2: Features",
            "",
            "• Gesture-based scrolling with pinch drag",
            "• Three-finger zoom control",
            "• A-OK gesture to navigate back",
            "• Smooth animation and transitions",
            "",
            "Section 3: About",
            "",
            "This browser window is part of the VisualMotion gesture carousel.",
            "It demonstrates how hand tracking can replace traditional input.",
            "",
        ]
        for i, line in enumerate(lines):
            ly = body_y + i * 26
            if content_y + 60 < ly < H:
                color = (33, 37, 41) if line and not line.startswith("•") else (80, 90, 100)
                screen.blit(bf.render(line, True, color), (40, ly))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        pygame.init()
        W, H = self.WINDOW_WIDTH, self.WINDOW_HEIGHT
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Browser — Gesture Navigation")
        clock = pygame.time.Clock()

        print("=" * 50)
        print("GESTURE BROWSER STARTED")
        print("Pinch tiles to navigate • Pinch-drag to scroll • A-OK to go back / quit")
        print("=" * 50)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera!")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        hands = mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.5,
            min_tracking_confidence=0.5, model_complexity=0
        )

        self.running = True
        tile_rects = []
        pinch_prev = False

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False

            ret, frame = cap.read()
            if not ret:
                pygame.display.flip()
                clock.tick(60)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            right_hand = None
            if results.multi_hand_landmarks:
                right_hand = results.multi_hand_landmarks[0].landmark

            # --- Gestures ---
            # Three-finger wheel (zoom)
            if right_hand and detect_three_finger_gesture(right_hand):
                if not self.wheel_active:
                    self.wheel_active = True
                    self.last_finger_angle = None
                ang = calculate_finger_angle(right_hand)
                if self.last_finger_angle is not None:
                    diff = ang - self.last_finger_angle
                    if diff > math.pi:
                        diff -= 2 * math.pi
                    elif diff < -math.pi:
                        diff += 2 * math.pi
                    self.wheel_angle = (self.wheel_angle + diff * 2) % (2 * math.pi)
                    self.zoom_level = clamp(
                        self.zoom_level + diff * self.zoom_sense,
                        self.zoom_min, self.zoom_max
                    )
                self.last_finger_angle = ang
            else:
                self.wheel_active = False
                self.last_finger_angle = None

            # A-OK: go back to new tab, or quit if already there
            ok_now = detect_ok_gesture(right_hand, self.ok_touch_threshold) if right_hand else False
            if ok_now and not self.ok_prev:
                if self.current_url:
                    print(f"A-OK — navigating back from {self.current_url}")
                    self.current_url = ""
                    self.selected_bookmark = None
                    self.scroll_y = 0.0
                    self.smooth_scroll_y = 0.0
                else:
                    print("A-OK — closing browser.")
                    self.running = False
            self.ok_prev = ok_now

            # Pinch
            pinch_now = is_pinching_gesture(right_hand, self.pinch_threshold) if right_hand else False
            pos = get_pinch_position(right_hand) if right_hand else None

            if right_hand and not self.wheel_active:
                if pinch_now and pos:
                    px = pos[0] * W
                    py = pos[1] * H

                    if not self._is_pinching:
                        # pinch just started
                        self.pinch_start_pos = (px, py)
                        self._is_pinching = True
                        self.last_pinch_y = py
                    else:
                        # continuing pinch — scroll
                        if self.last_pinch_y is not None and self.pinch_start_pos:
                            total_move = math.hypot(
                                px - self.pinch_start_pos[0],
                                py - self.pinch_start_pos[1]
                            )
                            if total_move > self.movement_threshold:
                                dy = (py - self.last_pinch_y) * self.scroll_gain
                                self.scroll_y += dy
                                self.scroll_y = clamp(self.scroll_y, -3000, 0)
                        self.last_pinch_y = py
                else:
                    # pinch released — check for tap on bookmark tile
                    if self._is_pinching and self.pinch_start_pos and self.last_pinch_y is not None:
                        total_move = math.hypot(
                            self.last_pinch_y - self.pinch_start_pos[1],
                            (pos[0] * W if pos else self.pinch_start_pos[0]) - self.pinch_start_pos[0]
                        )
                        if total_move <= self.movement_threshold and not self.current_url:
                            # check tile hit
                            tap_x, tap_y = self.pinch_start_pos
                            for rect, idx in tile_rects:
                                if rect.collidepoint(tap_x, tap_y):
                                    bm = self.BOOKMARKS[idx]
                                    self.selected_bookmark = idx
                                    self.current_url = bm["url"]
                                    self.scroll_y = 0.0
                                    self.smooth_scroll_y = 0.0
                                    print(f"✓ Navigated to {bm['name']} ({bm['url']})")
                                    break
                    self._is_pinching = False
                    self.last_pinch_y = None
                    self.pinch_start_pos = None
            else:
                self._is_pinching = False
                self.last_pinch_y = None
                self.pinch_start_pos = None

            pinch_prev = pinch_now

            self.smooth_scroll_y += (self.scroll_y - self.smooth_scroll_y) * 0.35

            # --- Draw ---
            screen.fill((30, 33, 40))
            self._draw_tab_bar(screen)
            self._draw_address_bar(screen)
            self._draw_bookmarks_bar(screen)

            if self.current_url:
                self._draw_page_content(screen)
            else:
                tile_rects = self._draw_new_tab_page(screen)

            pygame.display.flip()
            clock.tick(60)

        cap.release()
        pygame.quit()
        sys.exit()


# Allow running standalone: python browser_window.py
if __name__ == "__main__":
    BrowserWindow().run()
