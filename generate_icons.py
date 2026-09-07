"""Generate simple PNG icons for the PWA without PIL.

FIX: draws a clearly-visible tapered leaf (light green on deep green) with a
midrib vein and stem, plus TRUE transparency for the non-maskable icons'
rounded corners. The old icons were a solid-looking green square because the
leaf and background shades were nearly identical and corners were white.
"""
import struct
import zlib

def make_png(width, height, pixels_rgba):
    """Create an 8-bit RGBA PNG from raw RGBA pixel data."""
    def chunk(chunk_type, data):
        c = struct.pack(">I", len(data)) + chunk_type + data
        c += struct.pack(">I", zlib.crc32(chunk_type + data) & 0xffffffff)
        return c

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 6 = RGBA

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 (None)
        raw += pixels_rgba[y * width * 4:(y + 1) * width * 4]

    idat = zlib.compress(bytes(raw))
    return sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', b'')


def draw_leaf_icon(size, maskable):
    """Draw a deep-green rounded square with a light tapered leaf + vein.

    - Non-maskable: rounded corners are TRANSPARENT.
    - Maskable: full-bleed background, leaf kept within the ~80% safe zone.
    """
    bg = (20, 83, 45)          # #14532d deep green
    leaf = (74, 222, 128)      # #4ade80 light green (strong contrast)
    vein = (22, 101, 52)       # #166534 darker midrib
    stem = (200, 240, 210)     # pale stem

    radius = size // 6
    cx = size * 0.5
    cy = size * 0.5
    lrx = size * (0.34 if not maskable else 0.30)
    lry = size * (0.13 if not maskable else 0.11)
    ang = 0.7853981634  # 45 deg
    cos_a, sin_a = 0.70710678, 0.70710678

    vein_hw = lry * 0.18    # half-width of midrib
    stem_len = size * 0.14
    stem_w = size * 0.022

    pixels = bytearray()
    for y in range(size):
        for x in range(size):
            # --- background alpha (transparent rounded corners) ---
            if maskable:
                in_bg = True
            else:
                if x < radius and y < radius:
                    in_bg = (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2
                elif x >= size - radius and y < radius:
                    in_bg = (x - (size - radius)) ** 2 + (y - radius) ** 2 <= radius ** 2
                elif x < radius and y >= size - radius:
                    in_bg = (x - radius) ** 2 + (y - (size - radius)) ** 2 <= radius ** 2
                elif x >= size - radius and y >= size - radius:
                    in_bg = (x - (size - radius)) ** 2 + (y - (size - radius)) ** 2 <= radius ** 2
                else:
                    in_bg = True
            if not in_bg:
                pixels += bytes((0, 0, 0, 0))
                continue

            # --- rotate to leaf coordinates (long axis along +u) ---
            dx, dy = x - cx, y - cy
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a

            # tapered leaf: narrows to a point at its far end (u > 0)
            nr = (u / lrx) ** 2 + (v / lry) ** 2
            taper = 1.0 - 0.55 * max(0.0, u / lrx)
            in_leaf = nr <= 1.0 and abs(v) <= lry * taper

            # stem: short diagonal from leaf base toward bottom-left
            s0x, s0y = cx - lrx * 0.55 * cos_a, cy - lrx * 0.55 * sin_a
            s1x, s1y = s0x - stem_len * cos_a, s0y + stem_len * sin_a
            # point-to-segment distance
            sx, sy = x - s0x, y - s0y
            ex, ey = s1x - s0x, s1y - s0y
            seg = (ex * ex + ey * ey) or 1.0
            t = max(0.0, min(1.0, (sx * ex + sy * ey) / seg))
            px, py = s0x + t * ex, s0y + t * ey
            in_stem = ((x - px) ** 2 + (y - py) ** 2) <= stem_w * stem_w

            if in_stem:
                pixels += bytes(stem + (255,))
            elif in_leaf:
                if abs(v) <= vein_hw and u <= lrx * 0.72:
                    pixels += bytes(vein + (255,))
                else:
                    pixels += bytes(leaf + (255,))
            else:
                pixels += bytes(bg + (255,))

    return make_png(size, size, bytes(pixels))


# --- Generate icons ---
specs = [
    ('icon-192.png', 192, False),
    ('icon-512.png', 512, False),
    ('icon-192-maskable.png', 192, True),
    ('icon-512-maskable.png', 512, True),
]
for name, size, maskable in specs:
    with open(name, 'wb') as f:
        f.write(draw_leaf_icon(size, maskable))
    print('Created', name)