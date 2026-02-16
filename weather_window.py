"""
Fake weather app — big, cartoony, fills the whole 1280×720 window.
No scrolling required: everything fits on one screen.
"""
import math
import pygame

# ── Fake data ───────────────────────────────────────────────────────
CITY = "San Francisco"
TEMP = 68
DESC = "Partly Cloudy"
HI, LO = 72, 55

HOURLY = [
    ("Now", 68), ("1 PM", 70), ("2 PM", 71), ("3 PM", 72),
    ("4 PM", 70), ("5 PM", 67), ("6 PM", 64), ("7 PM", 61),
]

DAILY = [
    ("Mon", 72, 55), ("Tue", 69, 54), ("Wed", 65, 52),
    ("Thu", 63, 50), ("Fri", 67, 53), ("Sat", 71, 56), ("Sun", 73, 57),
]

# ── Colours ─────────────────────────────────────────────────────────
_SKY_TOP = (60, 160, 240)
_SKY_BOT = (25, 80, 180)
_WHITE = (255, 255, 255)
_LTBLUE = (180, 220, 255)
_YELLOW = (255, 220, 80)
_ORANGE = (255, 160, 50)
_GLASS = (255, 255, 255, 50)
_GLASS_B = (255, 255, 255, 90)
_BAR_BG = (60, 110, 190)
_BAR_FG = (255, 200, 60)

# ── Font cache ──────────────────────────────────────────────────────
_fc: dict[int, pygame.font.Font] = {}


def _f(size: int) -> pygame.font.Font:
    size = max(10, size)
    if size not in _fc:
        _fc[size] = pygame.font.Font(None, size)
    return _fc[size]


# ── Helpers ─────────────────────────────────────────────────────────
def _grad(surf, rect):
    for y in range(rect.height):
        t = y / max(1, rect.height)
        c = tuple(int(_SKY_TOP[i] + (_SKY_BOT[i] - _SKY_TOP[i]) * t) for i in range(3))
        pygame.draw.line(surf, c, (rect.x, rect.y + y), (rect.x + rect.width, rect.y + y))


def _glass(surf, rect, rad=18):
    s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(s, _GLASS, s.get_rect(), border_radius=rad)
    pygame.draw.rect(s, _GLASS_B, s.get_rect(), width=2, border_radius=rad)
    surf.blit(s, rect.topleft)


def _draw_sun(surf, cx, cy, r):
    """Cartoony sun with rays."""
    for i in range(12):
        a = math.radians(i * 30)
        x1 = cx + int((r + 6) * math.cos(a))
        y1 = cy + int((r + 6) * math.sin(a))
        x2 = cx + int((r + 22) * math.cos(a))
        y2 = cy + int((r + 22) * math.sin(a))
        pygame.draw.line(surf, _YELLOW, (x1, y1), (x2, y2), 4)
    pygame.draw.circle(surf, _YELLOW, (cx, cy), r)
    pygame.draw.circle(surf, (255, 240, 150), (cx - r // 4, cy - r // 4), r // 3)


def _draw_cloud(surf, cx, cy, w):
    """Cartoony cloud blob."""
    col = (240, 245, 255)
    r = w // 4
    pygame.draw.circle(surf, col, (cx - r, cy), r)
    pygame.draw.circle(surf, col, (cx + r, cy), r)
    pygame.draw.circle(surf, col, (cx, cy - r // 2), int(r * 1.2))
    pygame.draw.rect(surf, col, (cx - r - r, cy, r * 4, r))


# ── WeatherWindow ──────────────────────────────────────────────────

class WeatherWindow:
    """Full-screen cartoony weather overlay (1280x720). Scales with gui_scale."""

    def __init__(self, screen_w: int, screen_h: int):
        self.visible = False
        self._sw = screen_w
        self._sh = screen_h

    def open(self):
        self.visible = True

    def close(self):
        self.visible = False

    def hit_test(self, px, py, gui_scale):
        r = self._rect(gui_scale)
        return r.collidepoint(int(px), int(py))

    def _rect(self, s):
        w = min(int(1200 * s), self._sw - 20)
        h = min(int(680 * s), self._sh - 20)
        return pygame.Rect((self._sw - w) // 2, (self._sh - h) // 2, w, h)

    # ────────────────────────────────────────────────────────────────
    def draw(self, surface: pygame.Surface, gui_scale: float = 1.0):
        if not self.visible:
            return
        s = gui_scale
        win = self._rect(s)
        ox, oy = win.x, win.y
        W, H = win.width, win.height

        # Clip so nothing spills outside the window
        prev_clip = surface.get_clip()
        surface.set_clip(win)

        # ── sky gradient background ──
        _grad(surface, win)

        # ── cartoony sun + cloud ──
        sun_x = ox + W - int(140 * s)
        sun_y = oy + int(100 * s)
        _draw_sun(surface, sun_x, sun_y, int(40 * s))
        _draw_cloud(surface, sun_x - int(60 * s), sun_y + int(20 * s), int(70 * s))

        # -- TOP: city, big temp, description --
        city_lbl = _f(int(44 * s)).render(CITY, True, _WHITE)
        surface.blit(city_lbl, city_lbl.get_rect(centerx=ox + W // 2, top=oy + int(16 * s)))

        temp_lbl = _f(int(110 * s)).render(str(TEMP) + chr(176), True, _WHITE)
        surface.blit(temp_lbl, temp_lbl.get_rect(centerx=ox + W // 2, top=oy + int(50 * s)))

        desc_lbl = _f(int(32 * s)).render(DESC, True, _LTBLUE)
        surface.blit(desc_lbl, desc_lbl.get_rect(centerx=ox + W // 2, top=oy + int(150 * s)))

        hilo_lbl = _f(int(28 * s)).render(
            "H: " + str(HI) + chr(176) + "   L: " + str(LO) + chr(176), True, _LTBLUE)
        surface.blit(hilo_lbl, hilo_lbl.get_rect(centerx=ox + W // 2, top=oy + int(180 * s)))

        # -- HOURLY ROW --
        hr_y = oy + int(210 * s)
        hr_h = int(100 * s)
        hr_rect = pygame.Rect(ox + int(20 * s), hr_y, W - int(40 * s), hr_h)
        _glass(surface, hr_rect, max(8, int(14 * s)))

        col_w = hr_rect.width // len(HOURLY)
        for i, (label, t) in enumerate(HOURLY):
            cx = hr_rect.x + col_w * i + col_w // 2
            lbl = _f(int(22 * s)).render(label, True, _LTBLUE)
            surface.blit(lbl, lbl.get_rect(centerx=cx, top=hr_y + int(8 * s)))
            icon_y = hr_y + int(36 * s)
            if t >= 70:
                pygame.draw.circle(surface, _YELLOW, (cx, icon_y), int(10 * s))
            else:
                pygame.draw.circle(surface, (200, 215, 240), (cx, icon_y), int(10 * s))
                pygame.draw.circle(surface, (200, 215, 240), (cx + int(7 * s), icon_y), int(8 * s))
            tv = _f(int(24 * s)).render(str(t) + chr(176), True, _WHITE)
            surface.blit(tv, tv.get_rect(centerx=cx, top=hr_y + int(62 * s)))

        # -- 7-DAY FORECAST --
        dy_y = hr_y + hr_h + int(12 * s)
        row_h = int(30 * s)
        dy_h = int(30 * s) + row_h * len(DAILY)
        dy_rect = pygame.Rect(ox + int(20 * s), dy_y, W - int(40 * s), dy_h)
        _glass(surface, dy_rect, max(8, int(14 * s)))

        hdr = _f(int(20 * s)).render("7-DAY FORECAST", True, _LTBLUE)
        surface.blit(hdr, (dy_rect.x + int(14 * s), dy_y + int(6 * s)))

        for i, (day, hi, lo) in enumerate(DAILY):
            ry = dy_y + int(28 * s) + i * row_h
            if i > 0:
                sep = pygame.Surface((dy_rect.width - int(28 * s), 1), pygame.SRCALPHA)
                sep.fill((255, 255, 255, 40))
                surface.blit(sep, (dy_rect.x + int(14 * s), ry - int(2 * s)))
            d_lbl = _f(int(24 * s)).render(day, True, _WHITE)
            surface.blit(d_lbl, (dy_rect.x + int(16 * s), ry))
            lo_lbl = _f(int(22 * s)).render(str(lo) + chr(176), True, _LTBLUE)
            surface.blit(lo_lbl, (dy_rect.x + int(130 * s), ry + int(2 * s)))
            bar_x = dy_rect.x + int(190 * s)
            bar_w = int(480 * s)
            bar_h = max(4, int(10 * s))
            bar_y = ry + int(8 * s)
            pygame.draw.rect(surface, _BAR_BG, (bar_x, bar_y, bar_w, bar_h),
                             border_radius=max(2, int(5 * s)))
            lo_f = (lo - 48) / 30
            hi_f = (hi - 48) / 30
            fx = bar_x + int(lo_f * bar_w)
            fw = max(6, int((hi_f - lo_f) * bar_w))
            pygame.draw.rect(surface, _BAR_FG, (fx, bar_y, fw, bar_h),
                             border_radius=max(2, int(5 * s)))
            hi_lbl = _f(int(22 * s)).render(str(hi) + chr(176), True, _WHITE)
            surface.blit(hi_lbl, (bar_x + bar_w + int(10 * s), ry + int(2 * s)))

        # ── close hint ──
        hint = _f(int(22 * s)).render("pinch outside to close", True, _LTBLUE)
        surface.blit(hint, hint.get_rect(centerx=ox + W // 2, bottom=win.bottom - int(8 * s)))

        # ── Restore clip and draw border on top ──
        surface.set_clip(prev_clip)
        pygame.draw.rect(surface, _WHITE, win, width=3, border_radius=max(8, int(22 * s)))
