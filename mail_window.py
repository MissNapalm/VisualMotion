"""
Mail Window — Embedded email inbox with gesture-based scrolling and zooming.
"""

import cv2
import numpy as np
import math
import pygame
import sys
import random
import datetime

from shared import (
    mp_hands, clamp, get_font,
    get_pinch_distance, is_pinching_gesture, get_pinch_position,
    detect_three_finger_gesture, detect_ok_gesture,
    calculate_finger_angle, lm_to_screen,
)


class MailWindow:
    """Self-contained gesture-driven email inbox window."""

    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800

    def __init__(self):
        self.emails = self._make_fake_emails(80)
        self.scroll_y = 0.0
        self.smooth_scroll_y = 0.0
        self.scroll_gain = 4.0
        self._is_pinching = False
        self.last_pinch_y = None
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
        self.page_height = 3000
        self.running = False

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    @staticmethod
    def _make_fake_emails(n=60):
        SENDERS = [
            "Lena from Summation", "Nadia (Infra)", "Core Team Updates",
            "Billing Bot", "Product Announce", "Priya @ ML", "Marketing Ops",
            "Ahmed • Security", "Bruno (Design)", "Support", "CI Logs", "The Ops Room"
        ]
        SUBJECTS = [
            "Standup notes & priorities", "Your invoice is ready",
            "Quarterly planning: draft agenda", "Welcome to the beta",
            "Alert: action required on deployment", "Design review — nav cleanup",
            "Incident postmortem draft", "Metrics weekly recap",
            "Invitation: user research sessions", "Release train R42 checklist",
            "Reminder: security training", "Infra cost dashboard — July"
        ]
        SNIPPETS = [
            "Sharing quick notes from the sync today. We agreed to cut scope…",
            "Hi there! Your invoice for September is now available. You can…",
            "Here's a rough draft for next quarter's planning doc. Please…",
            "You're in! Start by exploring the quick start guide and sample…",
            "We detected a failing canary in us-west-2. Rollback is prepared…",
            "Attaching comps with the nav collapsed and with tabs. Feedback…",
            "Root cause was a misconfigured retry policy. We'll add guards…",
            "Top-line usage is up 12%. Retention is flat; activation dipped…",
            "We'd love to schedule 30 minutes to discuss your current tooling…",
            "This week: 16 merges, 2 hotfixes, 0 regressions. Please read…",
            "A friendly nudge that your security review is due Friday. It…",
            "Spend is trending 8% down week over week due to improved cache…"
        ]
        LABELS = ["Work", "Docs", "Billing", "Follow-up", "Personal", "Newsletters", "Release"]
        now = datetime.datetime.now()
        emails = []
        for _ in range(n):
            sender = random.choice(SENDERS)
            subject = random.choice(SUBJECTS)
            snippet = random.choice(SNIPPETS)
            unread = random.random() < 0.45
            starred = random.random() < 0.20
            lbls = random.sample(LABELS, k=random.randint(0, 2))
            dt = now - datetime.timedelta(minutes=random.randint(5, 60 * 24 * 14))
            if (now - dt).days == 0:
                time_label = dt.strftime("%I:%M %p").lstrip("0")
            else:
                time_label = dt.strftime("%b %d")
            emails.append({
                "sender": sender,
                "subject": subject,
                "snippet": snippet,
                "time": time_label,
                "unread": unread,
                "starred": starred,
                "labels": lbls
            })
        return emails

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_inbox_list(self, surface, x, y, w, h):
        pygame.draw.rect(surface, (255, 255, 255), (x, y, w, h))
        row_h = 72
        start_i = max(0, int((-self.smooth_scroll_y) // row_h) - 2)
        end_i = min(len(self.emails), start_i + h // row_h + 4)
        for i in range(start_i, end_i):
            ry = y + int(self.smooth_scroll_y) + i * row_h
            rect = pygame.Rect(x, ry, w, row_h)
            if self.emails[i]["unread"]:
                pygame.draw.rect(surface, (244, 248, 255), rect)
            else:
                pygame.draw.rect(surface, (255, 255, 255), rect)
            pygame.draw.line(surface, (233, 237, 241),
                             (rect.x, rect.bottom), (rect.right, rect.bottom), 1)
            # sender
            f_sender = pygame.font.SysFont('arial', 16, bold=self.emails[i]["unread"])
            surface.blit(f_sender.render(self.emails[i]["sender"], True, (33, 37, 41)),
                         (rect.x + 70, rect.y + 10))
            # subject + snippet
            subj_font = pygame.font.SysFont('arial', 16, bold=self.emails[i]["unread"])
            snip_font = pygame.font.SysFont('arial', 16)
            subj = self.emails[i]["subject"]
            snip = self.emails[i]["snippet"]
            subj_img = subj_font.render(subj, True, (33, 37, 41))
            sep_img = snip_font.render(" — ", True, (107, 114, 128))
            snp_img = snip_font.render(snip, True, (107, 114, 128))
            mx = rect.x + 70 + 200 + 10
            my = rect.y + 10
            maxw = w - (mx - x) - 90
            sj = subj
            sj_img = subj_img
            while sj_img.get_width() > maxw and len(sj) > 1:
                sj = sj[:-2] + "…"
                sj_img = subj_font.render(sj, True, (33, 37, 41))
            surface.blit(sj_img, (mx, my))
            sx = mx + sj_img.get_width()
            if sx + sep_img.get_width() < x + w - 90:
                surface.blit(sep_img, (sx, my))
                sx += sep_img.get_width()
            remain = (x + w - 90) - sx
            if remain > 0:
                sn = snip
                sn_img = snip_font.render(sn, True, (107, 114, 128))
                while sn_img.get_width() > remain and len(sn) > 1:
                    sn = sn[:-2] + "…"
                    sn_img = snip_font.render(sn, True, (107, 114, 128))
                surface.blit(sn_img, (sx, my))
            # time
            timg = pygame.font.SysFont('arial', 14).render(
                self.emails[i]["time"], True, (107, 114, 128))
            surface.blit(timg, (x + w - 12 - timg.get_width(), my + 2))
        page_bottom = y + self.smooth_scroll_y + len(self.emails) * row_h
        return page_bottom - y

    def _draw_chrome(self, screen):
        """Draw top bar, sidebar, and status bar."""
        W, H = self.WINDOW_WIDTH, self.WINDOW_HEIGHT
        # top bar
        pygame.draw.rect(screen, (255, 255, 255), (0, 0, W, 72))
        pygame.draw.rect(screen, (245, 247, 250), (12, 12, 220, 48), border_radius=12)
        pygame.draw.rect(screen, (245, 247, 250), (244, 20, W - 244 - 20, 32), border_radius=16)

        content_y = 80
        content_h = H - content_y

        # sidebar
        pygame.draw.rect(screen, (248, 249, 250), (0, content_y, 220, content_h))
        pygame.draw.rect(screen, (215, 227, 252), (16, content_y + 16, 188, 44), border_radius=22)
        sidebar_font = pygame.font.SysFont('arial', 18, bold=True)
        screen.blit(sidebar_font.render("Compose", True, (30, 64, 175)), (58, content_y + 27))
        pygame.draw.line(screen, (230, 232, 235), (220, content_y), (220, H), 1)

        # status bar
        font = pygame.font.SysFont('arial', 16, bold=True)
        max_scroll = max(0, self.page_height - (H - 80))
        status = f"Zoom {self.zoom_level:.2f}x • Scroll {int(-self.smooth_scroll_y)}/{max_scroll}"
        screen.blit(font.render(status, True, (90, 98, 110)), (16, H - 28))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        pygame.init()
        W, H = self.WINDOW_WIDTH, self.WINDOW_HEIGHT
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Mail — Embedded Inbox (Fallback)")
        clock = pygame.time.Clock()

        print("=" * 50)
        print("EMBEDDED INBOX (FALLBACK)")
        print("Pinch-drag to scroll • Three-finger rotate to zoom • A-OK to quit")
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

            # A-OK to quit
            ok_now = detect_ok_gesture(right_hand, self.ok_touch_threshold) if right_hand else False
            if ok_now and not self.ok_prev:
                print("A-OK — closing inbox fallback.")
                self.running = False
            self.ok_prev = ok_now

            # Pinch scroll
            if right_hand and not self.wheel_active:
                pinch_now = is_pinching_gesture(right_hand, self.pinch_threshold)
                pos = get_pinch_position(right_hand)
                if pinch_now and pos:
                    py = pos[1] * H
                    if self._is_pinching and self.last_pinch_y is not None:
                        dy = (py - self.last_pinch_y) * self.scroll_gain
                        self.scroll_y += dy
                        max_scroll_y = max(0, self.page_height - (H - 80))
                        self.scroll_y = clamp(self.scroll_y, -max_scroll_y, 0)
                    self.last_pinch_y = py
                    self._is_pinching = True
                else:
                    self._is_pinching = False
                    self.last_pinch_y = None
            else:
                self._is_pinching = False
                self.last_pinch_y = None

            self.smooth_scroll_y += (self.scroll_y - self.smooth_scroll_y) * 0.35

            # --- Draw ---
            screen.fill((240, 242, 245))
            self._draw_chrome(screen)

            content_x = 220
            content_y = 80
            content_w = W - content_x
            content_h = H - content_y

            s = self.zoom_level
            if abs(s - 1.0) > 0.001:
                zw = int(content_w * s)
                zh = int(content_h * s)
                zoom_surf = pygame.Surface((zw, zh), pygame.SRCALPHA).convert_alpha()
                zoom_surf.fill((255, 255, 255, 255))
                # temporarily scale scroll for zoomed surface
                saved = self.smooth_scroll_y
                self.smooth_scroll_y *= s
                ph = self._draw_inbox_list(zoom_surf, 0, 0, zw, zh)
                self.smooth_scroll_y = saved
                scaled = pygame.transform.smoothscale(
                    zoom_surf, (content_w, content_h)
                ).convert_alpha()
                screen.blit(scaled, (content_x, content_y))
                self.page_height = int(ph / s)
            else:
                ph = self._draw_inbox_list(screen, content_x, content_y, content_w, content_h)
                self.page_height = ph

            pygame.display.flip()
            clock.tick(60)

        cap.release()
        pygame.quit()
        sys.exit()


# Allow running standalone: python mail_window.py
if __name__ == "__main__":
    MailWindow().run()
