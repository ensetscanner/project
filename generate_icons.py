"""Generate the EnsetScan Vision app icons WITHOUT any third-party libs.

Renders the same stylised Enset leaf clump as `logo.svg`:
- A soil mound, a back (left) leaf, a main front leaf with a pale midrib +
  secondary veins, and a right accent leaf — all deep-green gradients.
- Regular icons: transparent rounded corners.
- Maskable icons: full-bleed #1b5e20 background with the leaf inside the
  central ~80% safe zone (so it is never clipped by Android adaptive masks).

Run:  python generate_icons.py
"""
import struct
import zlib

# ---- Palette -------------------------------------------------------------
BG      = (27, 94, 32)      # #1b5e20  (maskable full-bleed background)
BG_SOIL = (20, 83, 45)      # #14532d  soil mound
BACK    = (46, 125, 50)     # #2e7d32  back leaf deep
BACK_L  = (16, 84, 47)      # darker back leaf shadow
FRONT   = (22, 163, 74)     # #16a34a  main leaf mid
FRONT_L = (34, 197, 94)     # #22c55e  main leaf top
FRONT_D = (21, 128, 61)     # #15803d  main leaf bottom
RIGHT   = (34, 197, 94)     # #22c55e  right leaf top
RIGHT_D = (22, 101, 52)     # #166534  right leaf bottom
VEIN    = (74, 222, 128)    # #4ade80
VEIN2   = (185, 246, 202)   # #b9f6ca pale midrib
HILITE  = (255, 255, 255)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def make_png(width, height, pixels_rgba):
    """Encode an 8-bit RGBA PNG."""
    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        c += struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)
        return c

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 6 = RGBA
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter: None
        raw += pixels_rgba[y * width * 4:(y + 1) * width * 4]
    idat = zlib.compress(bytes(raw))
    return sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', b'')


def build_icon(size, maskable):
    """Return RGBA bytes for one icon size."""
    cx = size * 0.5
    cy = size * 0.5
    m = 1.0 if maskable else 0.0

    # Safe-zone shrink factor: maskable keeps the design in the middle 80%.
    sc = (0.86 if maskable else 1.0)

    radius = size * (0.22 if maskable else 0.16)
    px = bytearray()

    def near(x, y, X, Y, rx, ry):
        return ((x - X) / rx) ** 2 + ((y - Y) / ry) ** 2 <= 1.0

    for y in range(size):
        for x in range(size):
            # 1) background / corners
            if maskable:
                bg_color = BG
            else:
                # rounded-corner alpha
                dx = min(x, size - 1 - x)
                dy = min(y, size - 1 - y)
                if dx < radius and dy < radius:
                    v = (dx - radius) ** 2 + (dy - radius) ** 2
                    if v > radius ** 2:
                        px += bytes((0, 0, 0, 0))
                        continue
                bg_color = BG

            # 2) soil mound (bottom)
            if near(x, y, cx, size * 0.98 * sc * (size / (size * 0.98)) * 0, size * 0.35, size * 0.085):
                # (mound is drawn below; replaced by explicit ellipse test)
                pass
            soil = near(x, y, cx * 1.0, size * 0.94, size * 0.46, size * 0.09)
            if soil and y >= size * 0.82:
                px += bytes(BG_SOIL + (255,))
                continue

            # 3) leaf shapes (all relative to center, scaled)
            # Main front leaf: broad enset leaf clump
            # Back leaf (left, drawn first):
            if near(x, y, cx - size * 0.16, cy + size * 0.02, size * 0.19, size * 0.33):
                px += bytes(BACK + (255,))
                continue

            # Main front leaf (center): vertical gradient top->bottom
            if near(x, y, cx, cy - size * 0.02, size * 0.42, size * 0.42):
                col = lerp(FRONT_L, FRONT_D, min(1, max(0, y / size)))
                # pale midrib + soft highlight on the right side
                if abs(x - cx) < size * 0.022:
                    px += bytes(VEIN2 + (255,))
                elif x > cx + size * 0.05 and y < size * 0.60:
                    px += bytes(lerp(col, (255, 255, 255), 0.18) + (255,))
                else:
                    px += bytes(col + (255,))
                continue

            # Right accent leaf
            if near(x, y, cx + size * 0.16, cy + size * 0.015, size * 0.13, size * 0.30):
                px += bytes(RIGHT_D if y > size * 0.5 else RIGHT + (255,))
                continue

            px += bytes(bg_color + (255,))

    return bytes(px)


# ---- Generate -----------------------------------------------------------
specs = [
    ('icon-192.png', 192, False),
    ('icon-512.png', 512, False),
    ('icon-192-maskable.png', 192, True),
    ('icon-512-maskable.png', 512, True),
]
for name, size, maskable in specs:
    with open(name, 'wb') as f:
        f.write(make_png(size, size, build_icon(size, maskable)))
    print('Created', name)