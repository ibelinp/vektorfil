"""Draw the VektorFIL icon: a filter response (flat passband, steep skirt, a notch) in
white on a blue rounded square.  Writes assets/vektorfil.ico (all Windows sizes) and
assets/vektorfil.png.

    python tools/make_icon.py
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
S = 1024                                     # draw large, scale down for each size


def curve_points():
    """Elliptic-lowpass-like magnitude response, mapped into the icon square."""
    pts = []
    x0, x1, y_top, y_bot = 0.15, 0.86, 0.33, 0.80
    for i in range(400):
        t = i / 399
        if t < 0.46:                         # passband with a gentle ripple
            y = 0.012 * math.sin(t / 0.46 * 2.5 * math.pi)
        elif t < 0.66:                       # steep skirt down into the notch
            u = (t - 0.46) / 0.20
            y = u ** 1.6
        else:                                # stopband lobe back up after the notch
            u = (t - 0.66) / 0.34
            y = 1.0 - 0.42 * math.sin(min(u, 1) * math.pi * 0.9) ** 0.8
        pts.append(((x0 + (x1 - x0) * t) * S, (y_top + (y_bot - y_top) * y) * S))
    return pts


def draw():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(S * 0.20)
    d.rounded_rectangle([int(S * 0.04)] * 2 + [int(S * 0.96)] * 2, radius=r, fill=(28, 72, 176, 255))
    # faint grid
    for k in range(1, 4):
        y = S * (0.25 + 0.17 * k)
        d.line([(S * 0.14, y), (S * 0.86, y)], fill=(255, 255, 255, 55), width=int(S * 0.006))
    # stroke the curve with a round brush (wide polylines get ragged joints)
    pts = curve_points()
    rad = S * 0.028
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.hypot(xb - xa, yb - ya) / 2))
        for k in range(n):
            x, y = xa + (xb - xa) * k / n, ya + (yb - ya) * k / n
            d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(255, 255, 255, 255))
    return img


def main():
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    img = draw()
    img.resize((256, 256), Image.LANCZOS).save(out / "vektorfil.png")
    img.save(out / "vektorfil.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                                          (128, 128), (256, 256)])
    print("wrote", out / "vektorfil.ico", "and", out / "vektorfil.png")


if __name__ == "__main__":
    main()
