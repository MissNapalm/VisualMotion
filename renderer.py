"""
Renderer — all Pygame drawing: app icons, card rows, zoom wheel, hand HUD, status bar.
"""

import math
import pygame

from state import APP_COLORS, CATEGORIES, CARD_COUNT


# ==============================
# Font cache
# ==============================
_font_cache: dict[int, pygame.font.Font] = {}


def get_font(size: int) -> pygame.font.Font:
    if size not in _font_cache:
        _font_cache[size] = pygame.font.Font(None, size)
    return _font_cache[size]


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ==============================
# App icon
# ==============================
def draw_app_icon(surface, app_name, x, y, base_w, base_h,
                  is_selected=False, zoom_scale=1.0, gui_scale=1.0):
    w = int(base_w * gui_scale)
    h = int(base_h * gui_scale)
    if is_selected:
        w = int(w * zoom_scale)
        h = int(h * zoom_scale)
    br = max(12, int(50 * gui_scale))
    rect = pygame.Rect(x - w // 2, y - h // 2, w, h)
    base_color = APP_COLORS.get(app_name, (100, 100, 100))
    color = tuple(min(255, int(base_color[i] * 1.2)) for i in range(3))
    pygame.draw.rect(surface, color, rect, border_radius=br)
    if is_selected:
        sel = pygame.Rect(rect.x - int(6 * gui_scale), rect.y - int(6 * gui_scale),
                          rect.width + int(12 * gui_scale), rect.height + int(12 * gui_scale))
        pygame.draw.rect(surface, (255, 255, 255), sel,
                         width=max(2, int(8 * gui_scale)), border_radius=br)
    ratio = w / max(1, int(base_w * gui_scale))
    icon_img = get_font(max(24, int(120 * ratio))).render(app_name[0], True, (255, 255, 255, 180))
    surface.blit(icon_img, icon_img.get_rect(center=(x, y - int(20 * gui_scale))))
    text_img = get_font(max(12, int(36 * ratio))).render(app_name, True, (255, 255, 255))
    surface.blit(text_img, text_img.get_rect(center=(x, y + int(60 * gui_scale))))
    return rect


# ==============================
# Card row
# ==============================
def draw_cards(surface, center_x, center_y, card_offset, category_idx,
               selected_card, selected_category, zoom_progress,
               window_width, gui_scale, base_w, base_h, base_spacing):
    names = CATEGORIES[category_idx]
    rects = []
    stride = int(base_w * gui_scale) + int(base_spacing * gui_scale)
    first = max(0, int((-card_offset - window_width // 2) / stride) - 1)
    last = min(CARD_COUNT, int((-card_offset + window_width // 2) / stride) + 2)
    # non-selected first
    for i in range(first, last):
        x, y = int(center_x + i * stride + card_offset), int(center_y)
        if not (selected_card == i and selected_category == category_idx):
            rects.append((draw_app_icon(surface, names[i], x, y, base_w, base_h,
                                        False, 1.0, gui_scale), i, category_idx))
    # selected on top
    for i in range(first, last):
        x, y = int(center_x + i * stride + card_offset), int(center_y)
        if selected_card == i and selected_category == category_idx:
            rects.append((draw_app_icon(surface, names[i], x, y, base_w, base_h,
                                        True, 1.0 + zoom_progress * 0.3, gui_scale), i, category_idx))
    return rects


# ==============================
# Zoom wheel overlay
# ==============================
def draw_wheel(surface, state, window_width, window_height):
    if not state.wheel_active:
        return
    s = state.gui_scale
    cx, cy = state.wheel_center_x, state.wheel_center_y
    r = int(state.wheel_radius * s)
    white = (255, 255, 255)
    margin = r + int(80 * s)
    buf = pygame.Surface((margin * 2, margin * 2), pygame.SRCALPHA)
    m = margin
    for i in range(5):
        rr = r + int(15 * s) + i * int(10 * s)
        pygame.draw.circle(buf, (*white, 100 - i * 20), (m, m), rr, max(1, int(2 * s)))
    surface.blit(buf, (cx - margin, cy - margin))

    pygame.draw.circle(surface, white, (cx, cy), r, max(1, int(4 * s)))
    pygame.draw.circle(surface, white, (cx, cy), r - int(20 * s), max(1, int(2 * s)))

    segs = 48
    prog = int((state.wheel_angle / (2 * math.pi)) * segs) % segs
    ir = r - int(10 * s)
    for i in range(prog):
        sa = math.radians(i * 360 / segs) - math.pi / 2
        ea = math.radians((i + 1) * 360 / segs) - math.pi / 2
        pygame.draw.line(surface, white,
                         (cx + int(ir * math.cos(sa)), cy + int(ir * math.sin(sa))),
                         (cx + int(ir * math.cos(ea)), cy + int(ir * math.sin(ea))),
                         max(1, int(6 * s)))

    pl = r - int(30 * s)
    px = cx + int(pl * math.cos(state.wheel_angle))
    py = cy + int(pl * math.sin(state.wheel_angle))
    pygame.draw.line(surface, white, (cx, cy), (px, py), max(1, int(3 * s)))
    pygame.draw.circle(surface, white, (px, py), max(2, int(6 * s)))
    pygame.draw.circle(surface, white, (cx, cy), max(2, int(8 * s)))

    font = get_font(max(18, int(40 * s)))
    t = font.render(f"GUI {state.gui_scale:.2f}x", True, white)
    tr = t.get_rect(center=(cx, cy + r + int(44 * s)))
    bg = pygame.Rect(tr.x - int(10 * s), tr.y - int(5 * s),
                     tr.width + int(20 * s), tr.height + int(10 * s))
    pygame.draw.rect(surface, (20, 20, 20), bg)
    pygame.draw.rect(surface, white, bg, max(1, int(2 * s)))
    surface.blit(t, tr)
