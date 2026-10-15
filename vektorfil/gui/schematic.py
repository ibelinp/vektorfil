"""Schematic view of a ladder, drawn in the style of a circuit editor.

Layout is computed from measured text, so labels never overlap:
  * every element gets a designator (L1, L2, ..., C1, C2, ... in order of appearance)
    and a value line ("L=26.95 nH"), the value in an accent colour;
  * series elements carry their label above, shunt elements to the right
    (a shunt tank: L label left, C label right);
  * ports are drawn as chevrons (1) / (2) with the termination resistance;
  * when the ladder does not fit, the view keeps a readable minimum scale and grows
    wider, so the surrounding scroll area shows a horizontal scroll bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from vektorfil.synth.network import Ladder

MIN_SCALE = 0.85      # never draw smaller than this (scroll instead)
MAX_SCALE = 1.5
SYM = 46              # length of a two-terminal symbol body (coil / capacitor + leads)
GAP = 8               # gap between a symbol and its label


def eng(x, unit):
    """Engineering notation with 4 significant digits: 26.95 nH, 172.6 fF."""
    if x is None:
        return ""
    for k, p in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "u"),
                 (1e-9, "n"), (1e-12, "p"), (1e-15, "f")):
        if abs(x) >= k * 0.99995:
            return f"{x / k:.4g} {p}{unit}"
    return f"{x:.3g} {unit}"


@dataclass
class _Cell:
    """One column of the layout: a port or a ladder branch, with its measured width."""
    kind: str                  # 'port_in', 'port_out', or a branch
    width: float
    branch: object = None
    names: tuple = (None, None)
    extra: dict = field(default_factory=dict)


class SchematicView(QWidget):
    """Widget that draws a Ladder as a schematic (see the module docstring)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ladder: Ladder | None = None
        self.message = ""
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(250)
        self._layout = None
        self.port_names = ("1", "2")       # labels of the input and output ports

    # ------------------------------------------------------------ public
    def set_ladder(self, ladder: Ladder | None, message: str = "", ports=("1", "2")):
        """Show a new ladder (or a message when ladder is None); ports: the two port labels."""
        self.ladder = ladder
        self.port_names = tuple(ports)
        self.message = message
        self._layout = None
        self._relayout()
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update()

    # ------------------------------------------------------------ fonts / colours
    def _fonts(self):
        """Fonts for designators and values."""
        f_name = QFont(self.font())
        f_name.setPointSizeF(max(9.0, f_name.pointSizeF()))
        f_val = QFont(f_name)
        return f_name, f_val

    def _colors(self):
        """Wire/text colour, value colour, dimmed colour and node colour for the current theme."""
        pal = self.palette()
        fg = pal.windowText().color()
        dark = pal.window().color().lightness() < 128
        value = QColor("#7fb2ff") if dark else QColor("#1f45b5")
        node = QColor("#3aa655") if not dark else QColor("#6fd38b")
        dim = QColor(fg)
        dim.setAlpha(170)
        return fg, value, dim, node

    # ------------------------------------------------------------ layout
    def _label_w(self, lines):
        """Width in pixels of the widest of the given label lines."""
        f_name, f_val = self._fonts()
        fm = QFontMetricsF(f_name)
        return max((fm.horizontalAdvance(t) for t in lines if t), default=0.0)

    def _relayout(self):
        """Measure every label, size every column and the whole drawing."""
        lad = self.ladder
        if lad is None:
            self.setMinimumWidth(0)
            return
        fm = QFontMetricsF(self._fonts()[0])
        self.lh = fm.height()
        names = lad.designators()
        cells = [_Cell("port_in", 60 + self._label_w([f"{lad.r_source:.4g} Ω"]) / 2)]
        for b, (ln, cn) in zip(lad.branches, names):
            lab_l = [ln, self._val("L", b.L)] if ln else []
            if b.kind == "TEM":
                lab_l = [ln, self._tl_val(b)]
            lab_c = [cn, self._val("C", b.C)] if cn else []
            if b.pos == "series":
                if b.kind in ("L", "C"):
                    w = max(SYM + 18, self._label_w(lab_l or lab_c) + 14)
                elif b.kind == "LC_par":
                    w = max(SYM + 40, self._label_w(lab_l + lab_c) + 14)
                else:  # series L then series C in the same arm
                    wl = max(SYM + 10, self._label_w(lab_l) + 10)
                    wc = max(SYM + 10, self._label_w(lab_c) + 10)
                    w = wl + wc
                    cells.append(_Cell(b.kind, w, b, (ln, cn), {"wl": wl}))
                    continue
            else:
                if b.kind in ("LC_par", "TEM"):
                    wl, wc = self._label_w(lab_l), self._label_w(lab_c)
                    w = wl + GAP + 16 + 16 + GAP + wc + 18
                    cells.append(_Cell(b.kind, w, b, (ln, cn), {"wl": wl}))
                    continue
                w = 14 + GAP + self._label_w(lab_l + lab_c) + 22
            cells.append(_Cell(b.kind, w, b, (ln, cn)))
        cells.append(_Cell("port_out", 60 + self._label_w([self._term_text(lad.r_load)]) / 2))
        self._layout = cells
        lh = self.lh
        self.rail_y = 18 + 2 * lh + 34          # room for tank bridges and their labels
        self.gnd_y = self.rail_y + max(150.0, 4 * lh + 70)
        self.total_w = sum(c.width for c in cells) + 20
        self.total_h = self.gnd_y + 26
        self.setMinimumWidth(int(self.total_w * MIN_SCALE) + 2)

    @staticmethod
    def _term_text(r):
        """Termination label: '50 Ω', or 'open' / 'short' for a singly terminated end."""
        if r == float("inf"):
            return "open"
        if r == 0:
            return "short"
        return f"{r:.4g} Ω"

    @staticmethod
    def _tl_val(b):
        """TEM line label: 'Z=7.89 Ω, 45°'."""
        return f"Z={b.z0:.4g} Ω, {b.theta:.3g}°"

    @staticmethod
    def _val(sym, v):
        if v is None:
            return ""
        return f"{sym}={eng(v, 'H' if sym == 'L' else 'F')}"

    # ------------------------------------------------------------ symbols
    def _inductor_h(self, p, x0, x1, y, n=4):
        """Horizontal coil from x0 to x1 at height y (with leads)."""
        body = min(SYM - 10, x1 - x0 - 8)
        lead = (x1 - x0 - body) / 2
        p.drawLine(QPointF(x0, y), QPointF(x0 + lead, y))
        p.drawLine(QPointF(x1 - lead, y), QPointF(x1, y))
        r = body / n / 2
        for i in range(n):
            p.drawArc(QRectF(x0 + lead + i * 2 * r, y - r * 1.15, 2 * r, 2 * r * 1.15), 0, 180 * 16)

    def _inductor_v(self, p, x, y0, y1, n=4):
        """Vertical coil from y0 to y1 at x (with leads)."""
        body = min(SYM - 10, y1 - y0 - 8)
        lead = (y1 - y0 - body) / 2
        p.drawLine(QPointF(x, y0), QPointF(x, y0 + lead))
        p.drawLine(QPointF(x, y1 - lead), QPointF(x, y1))
        r = body / n / 2
        for i in range(n):
            p.drawArc(QRectF(x - r * 1.15, y0 + lead + i * 2 * r, 2 * r * 1.15, 2 * r), -90 * 16, 180 * 16)

    def _capacitor_h(self, p, x0, x1, y):
        """Horizontal capacitor between x0 and x1 at height y."""
        m = (x0 + x1) / 2
        p.drawLine(QPointF(x0, y), QPointF(m - 4, y))
        p.drawLine(QPointF(m + 4, y), QPointF(x1, y))
        p.drawLine(QPointF(m - 4, y - 12), QPointF(m - 4, y + 12))
        p.drawLine(QPointF(m + 4, y - 12), QPointF(m + 4, y + 12))

    def _capacitor_v(self, p, x, y0, y1):
        """Vertical capacitor between y0 and y1 at x."""
        m = (y0 + y1) / 2
        p.drawLine(QPointF(x, y0), QPointF(x, m - 4))
        p.drawLine(QPointF(x, m + 4), QPointF(x, y1))
        p.drawLine(QPointF(x - 12, m - 4), QPointF(x + 12, m - 4))
        p.drawLine(QPointF(x - 12, m + 4), QPointF(x + 12, m + 4))

    def _tline_v(self, p, x, y0, y1):
        """Vertical transmission-line (coaxial) symbol from y0 to y1 at x: a body with an
        inner conductor, the outer conductor drawn as a long thin rectangle."""
        h = min(SYM, y1 - y0 - 8)
        lead = (y1 - y0 - h) / 2
        p.drawLine(QPointF(x, y0), QPointF(x, y0 + lead))
        p.drawLine(QPointF(x, y1 - lead), QPointF(x, y1))
        p.drawRect(QRectF(x - 6, y0 + lead, 12, h))
        p.drawLine(QPointF(x, y0 + lead), QPointF(x, y0 + lead + h))

    def _ground(self, p, x, y):
        """Ground symbol at (x, y)."""
        p.drawLine(QPointF(x - 11, y), QPointF(x + 11, y))
        p.drawLine(QPointF(x - 7, y + 4), QPointF(x + 7, y + 4))
        p.drawLine(QPointF(x - 3, y + 8), QPointF(x + 3, y + 8))

    def _dot(self, p, x, y):
        """Junction dot at (x, y)."""
        p.save()
        p.setBrush(p.pen().color())
        p.drawEllipse(QPointF(x, y), 2.6, 2.6)
        p.restore()

    def _text(self, p, x, y, name, value, align):
        """Two-line label: designator (text colour) over value (accent colour).
        align: 'center' (x is the centre), 'left' (x is the left edge), 'right'."""
        fg, valc, dim, _ = self._colors()
        f_name, f_val = self._fonts()
        fm = QFontMetricsF(f_name)
        for i, (t, col, fnt) in enumerate(((name, fg, f_name), (value, valc, f_val))):
            if not t:
                continue
            w = fm.horizontalAdvance(t)
            xx = x - w / 2 if align == "center" else (x if align == "left" else x - w)
            p.setPen(col)
            p.setFont(fnt)
            p.drawText(QPointF(xx, y + fm.ascent() + i * fm.height()), t)

    # ------------------------------------------------------------ paint
    def paintEvent(self, _):
        """Draw ports, branches and labels, scaled to fit (never below MIN_SCALE)."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), self.palette().base())
        fg, valc, dim, node = self._colors()
        lad = self.ladder
        if lad is None or self._layout is None:
            p.setPen(dim)
            p.drawText(self.rect(), Qt.AlignCenter, self.message or "No design")
            return
        scale = min(MAX_SCALE, self.width() / self.total_w, self.height() / self.total_h)
        scale = max(scale, MIN_SCALE)
        p.translate(max(4.0, (self.width() - self.total_w * scale) / 2),
                    max(2.0, (self.height() - self.total_h * scale) / 2))
        p.scale(scale, scale)

        wire = QPen(fg, 1.5)
        wire.setCapStyle(Qt.RoundCap)
        lh = self.lh
        Y, G = self.rail_y, self.gnd_y
        x = 10.0
        for c in self._layout:
            p.setPen(wire)
            x0, x1 = x, x + c.width
            b = c.branch
            ln, cn = c.names
            if c.kind in ("port_in", "port_out"):
                r = lad.r_source if c.kind == "port_in" else lad.r_load
                if c.kind == "port_in":
                    xc = x0 + 22
                    p.drawLine(QPointF(xc, Y - 12), QPointF(xc + 12, Y))
                    p.drawLine(QPointF(xc + 12, Y), QPointF(xc, Y + 12))
                    p.drawLine(QPointF(xc + 12, Y), QPointF(x1, Y))
                    self._text(p, xc + 4, Y + 18, f"({self.port_names[0]})", f"{r:.4g} Ω", "center")
                else:
                    xc = x1 - 34
                    p.drawLine(QPointF(x0, Y), QPointF(xc + 12, Y))
                    p.drawLine(QPointF(xc, Y - 12), QPointF(xc + 12, Y))
                    p.drawLine(QPointF(xc + 12, Y), QPointF(xc, Y + 12))
                    self._text(p, xc + 8, Y + 18, f"({self.port_names[1]})", self._term_text(r), "center")
            elif b.pos == "series":
                if b.kind == "L":
                    self._inductor_h(p, x0, x1, Y)
                    self._text(p, (x0 + x1) / 2, Y - 14 - 2 * lh, ln, self._val("L", b.L), "center")
                elif b.kind == "C":
                    self._capacitor_h(p, x0, x1, Y)
                    self._text(p, (x0 + x1) / 2, Y - 16 - 2 * lh, cn, self._val("C", b.C), "center")
                elif b.kind == "LC_par":
                    a, e = x0 + 12, x1 - 12
                    up = Y - 30
                    p.drawLine(QPointF(x0, Y), QPointF(a, Y))
                    p.drawLine(QPointF(e, Y), QPointF(x1, Y))
                    p.drawLine(QPointF(a, Y), QPointF(a, up))
                    p.drawLine(QPointF(e, Y), QPointF(e, up))
                    self._inductor_h(p, a, e, Y)
                    self._capacitor_h(p, a, e, up)
                    self._dot(p, a, Y)
                    self._dot(p, e, Y)
                    self._text(p, (x0 + x1) / 2, up - 14 - 2 * lh, cn, self._val("C", b.C), "center")
                    self._text(p, (x0 + x1) / 2, Y + 10, ln, self._val("L", b.L), "center")
                else:  # LC_ser in the series arm: L then C
                    xm = x0 + c.extra["wl"]
                    self._inductor_h(p, x0, xm, Y)
                    self._capacitor_h(p, xm, x1, Y)
                    self._text(p, (x0 + xm) / 2, Y - 14 - 2 * lh, ln, self._val("L", b.L), "center")
                    self._text(p, (xm + x1) / 2, Y - 16 - 2 * lh, cn, self._val("C", b.C), "center")
            else:
                p.drawLine(QPointF(x0, Y), QPointF(x1, Y))
                top, bot = Y, G - 12
                if b.kind in ("LC_par", "TEM"):
                    xs = x0 + c.extra["wl"] + GAP + 16 + 4
                    self._dot(p, xs, Y)
                    p.drawLine(QPointF(xs, Y), QPointF(xs, top + 18))
                    p.drawLine(QPointF(xs - 16, top + 18), QPointF(xs + 16, top + 18))
                    p.drawLine(QPointF(xs - 16, bot - 18), QPointF(xs + 16, bot - 18))
                    p.drawLine(QPointF(xs, bot - 18), QPointF(xs, bot))
                    if b.kind == "TEM":
                        self._tline_v(p, xs - 16, top + 18, bot - 18)
                    else:
                        self._inductor_v(p, xs - 16, top + 18, bot - 18)
                    self._capacitor_v(p, xs + 16, top + 18, bot - 18)
                    ym = (top + bot) / 2 - lh
                    self._text(p, xs - 16 - GAP - 4, ym, ln,
                               self._tl_val(b) if b.kind == "TEM" else self._val("L", b.L), "right")
                    self._text(p, xs + 16 + GAP + 6, ym, cn, self._val("C", b.C), "left")
                else:
                    xs = x0 + 14
                    self._dot(p, xs, Y)
                    tx = xs + 12 + GAP
                    if b.kind == "C":
                        self._capacitor_v(p, xs, top, bot)
                        self._text(p, tx, (top + bot) / 2 - lh, cn, self._val("C", b.C), "left")
                    elif b.kind == "L":
                        self._inductor_v(p, xs, top, bot)
                        self._text(p, tx, (top + bot) / 2 - lh, ln, self._val("L", b.L), "left")
                    else:  # LC_ser to ground: L above C
                        ym = (top + bot) / 2
                        self._inductor_v(p, xs, top, ym)
                        p.setPen(wire)
                        self._capacitor_v(p, xs, ym, bot)
                        self._text(p, tx, (top + ym) / 2 - lh, ln, self._val("L", b.L), "left")
                        self._text(p, tx, (ym + bot) / 2 - lh, cn, self._val("C", b.C), "left")
                p.setPen(wire)
                self._ground(p, xs, G - 12)
            x = x1
