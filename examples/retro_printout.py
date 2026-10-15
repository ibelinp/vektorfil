"""A retro, terminal-style ASCII printout of a VektorFIL design (and a PNG of it).

The design is the classic elliptic lowpass example of the 1990s synthesis manuals:
7th order, 0.1 dB ripple, 1 kHz, stopband from 1150 Hz, 600 ohm.

    python examples/retro_printout.py      -> prints the listing, writes retro_printout.png
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vektorfil.filters import FilterSpec, design  # noqa: E402


def eng(x, unit):
    for k, p in ((1, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p")):
        if abs(x) >= k:
            return f"{x / k:11.6f} {p}{unit}"
    return f"{x:11.4e} {unit}"


def listing(res):
    lad = res.ladder
    out = []
    n = 1
    out.append(f"{n:3d} |---R---|  {lad.r_source:11.6f} ohm")
    for b in lad.branches:
        n += 1
        if b.pos == "shunt" and b.kind == "C":
            out.append(f"{n:3d} |   C      {eng(b.C, 'F')}")
            out.append("    |")
        elif b.pos == "series" and b.kind == "L":
            out.append(f"{n:3d} |---L---|  {eng(b.L, 'H')}")
        elif b.pos == "series" and b.kind == "LC_par":
            f0 = 1 / (2 * math.pi * math.sqrt(b.L * b.C))
            out.append("    |  ,-',")
            out.append(f"{n:3d} |  L C     {eng(b.L, 'H')}   res.freq")
            out.append(f"    |  `-,'    {eng(b.C, 'F')}   {f0 / 1e3:9.6f} kHz")
        else:
            out.append(f"{n:3d} |  {b.pos} {b.kind}  L={b.L}  C={b.C}")
    n += 1
    out.append(f"{n:3d} |---R---|  {lad.r_load:11.6f} ohm")
    return out


def main():
    spec = FilterSpec(ftype="Lowpass", shape="Elliptic", subtype="Shunt element first", order=7, fc=1000,
                      r_source=600, ripple_db=0.1, elliptic_spec="Fs", fs=1150)
    res = design(spec)
    # same resonator order as the classic printout: notches from high to low along the ladder
    import mpmath as mp
    from vektorfil.synth import approx
    from vektorfil.synth.ladder import synthesize
    from vektorfil.synth.network import Ladder
    proto = approx.elliptic(7, 0.1, ws=1.15)
    seq = sorted(proto.zeros, reverse=True) + [mp.inf] * proto.n_inf
    res.ladder = Ladder.from_prototype(synthesize(proto, first="shunt", sequence=seq), r0=600, fc=1000)
    head = ["VEKTORFIL 1.0.0                  LADDER  LISTING",
            "ELLIPTIC LOWPASS  N=7   AP=0.10 DB   FP=1000 HZ   FS=1150 HZ",
            f"AMIN={res.info['amin_db']:.6f} DB     TERMINATIONS 600 OHM", ""]
    zeros = "  ".join(f"{z:.4f}" for z in res.info["zeros"])
    tail = ["", f"TRANSMISSION ZEROS (HZ): {zeros}", "*** END OF LISTING ***"]
    lines = head + listing(res) + tail
    text = "\n".join(lines)
    print(text)

    try:   # green-phosphor terminal image for posting
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    font = None
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            font = ImageFont.truetype(name, 22)
            break
        except OSError:
            pass
    font = font or ImageFont.load_default()
    w = max(len(l) for l in lines) * 13 + 80
    h = len(lines) * 28 + 80
    img = Image.new("RGB", (w, h), (8, 14, 8))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        y = 40 + i * 28
        d.text((40, y), line, font=font, fill=(60, 255, 120))
        # faint scanlines
        d.line([(0, y + 26), (w, y + 26)], fill=(4, 22, 8))
    out = Path(__file__).with_suffix(".png")
    img.save(out)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
