"""Generate simple PNG icons for the PWA without PIL."""
import struct
import zlib

def make_png(width, height, pixels):
    """Create a PNG from raw RGB pixel data."""
    def chunk(chunk_type, data):
        c = struct.pack(">I", len(data))
        c += chunk_type
        c += data
        c += struct.pack(">I", zlib.crc32(chunk_type + data) & 0xffffffff)
        return c

    # PNG signature
    sig = b'\x89PNG\r\n\x1a\n'

    # IHDR
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    # IDAT - raw image data with filter byte 0 per row
    raw = b''
    for y in range(height):
        raw += b'\x00'  # filter type 0 (None)
        raw += bytes(pixels[y * width * 3:(y + 1) * width * 3])

    idat = zlib.compress(raw)

    # IEND
    iend = b''

    return sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', iend)


def draw_leaf_icon(size):
    """Draw a green rounded-square with a simple leaf shape."""
    pixels = bytearray(size * size * 3)
    # Background: dark green (#1b5e20)
    bg = (27, 94, 32)
    # Leaf color: lighter green (#2e7d32)
    leaf = (46, 125, 50)
    # Rounded corner radius
    radius = size // 8

    for y in range(size):
        for x in range(size):
            idx = (y * size + x) * 3
            # Check if inside rounded rect
            # Corner detection
            in_corner = False
            if x < radius and y < radius:
                in_corner = (x - radius) ** 2 + (y - radius) ** 2 > radius ** 2
            elif x >= size - radius and y < radius:
                in_corner = (x - (size - radius)) ** 2 + (y - radius) ** 2 > radius ** 2
            elif x < radius and y >= size - radius:
                in_corner = (x - radius) ** 2 + (y - (size - radius)) ** 2 > radius ** 2
            elif x >= size - radius and y >= size - radius:
                in_corner = (x - (size - radius)) ** 2 + (y - (size - radius)) ** 2 > radius ** 2

            if in_corner:
                # Transparent corner (use white for simplicity)
                pixels[idx] = 255
                pixels[idx + 1] = 255
                pixels[idx + 2] = 255
            else:
                # Background
                pixels[idx] = bg[0]
                pixels[idx + 1] = bg[1]
                pixels[idx + 2] = bg[2]

                # Draw a simple leaf shape (ellipse rotated)
                # Center of leaf
                cx = size * 0.5
                cy = size * 0.5
                # Leaf dimensions
                rx = size * 0.32
                ry = size * 0.18
                # Check if point is in ellipse
                dx = (x - cx) / rx
                dy = (y - cy) / ry
                if dx * dx + dy * dy <= 1.0:
                    pixels[idx] = leaf[0]
                    pixels[idx + 1] = leaf[1]
                    pixels[idx + 2] = leaf[2]

    return make_png(size, size, bytes(pixels))


# Generate 192x192 icon
icon_192 = draw_leaf_icon(192)
with open('icon-192.png', 'wb') as f:
    f.write(icon_192)
print('Created icon-192.png')

# Generate 512x512 icon
icon_512 = draw_leaf_icon(512)
with open('icon-512.png', 'wb') as f:
    f.write(icon_512)
print('Created icon-512.png')

# Generate maskable icons (with more padding for safe zone)
def draw_maskable_icon(size):
    """Draw a maskable icon with extra padding."""
    pixels = bytearray(size * size * 3)
    bg = (27, 94, 32)
    leaf = (46, 125, 50)

    for y in range(size):
        for x in range(size):
            idx = (y * size + x) * 3
            pixels[idx] = bg[0]
            pixels[idx + 1] = bg[1]
            pixels[idx + 2] = bg[2]

            # Smaller leaf for maskable (80% of icon)
            cx = size * 0.5
            cy = size * 0.5
            rx = size * 0.25
            ry = size * 0.14
            dx = (x - cx) / rx
            dy = (y - cy) / ry
            if dx * dx + dy * dy <= 1.0:
                pixels[idx] = leaf[0]
                pixels[idx + 1] = leaf[1]
                pixels[idx + 2] = leaf[2]

    return make_png(size, size, bytes(pixels))

icon_192_mask = draw_maskable_icon(192)
with open('icon-192-maskable.png', 'wb') as f:
    f.write(icon_192_mask)
print('Created icon-192-maskable.png')

icon_512_mask = draw_maskable_icon(512)
with open('icon-512-maskable.png', 'wb') as f:
    f.write(icon_512_mask)
print('Created icon-512-maskable.png')