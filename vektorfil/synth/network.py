"""Ladder networks: scaling, lowpass->highpass transform and ABCD analysis (float)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .ladder import Branch, LadderResult


@dataclass
class Ladder:
    """A denormalised ladder network (values in H and F) with its terminations.
    Provides scaling, lowpass->highpass, ABCD/S-parameter analysis and an element table.
    """
    branches: list               # list[Branch] with float values (H, F)
    r_source: float = 50.0
    r_load: float = 50.0
    title: str = ""
    notes: dict = field(default_factory=dict)
    q_l: float | None = None     # inductor Q (constant over frequency); None = lossless
    q_c: float | None = None     # capacitor Q

    # ----------------------------------------------------------- construction
    @classmethod
    def from_prototype(cls, res: LadderResult, r0=50.0, fc=1.0, title=""):
        """Denormalise a 1-ohm, 1-rad/s lowpass prototype to r0 ohms and fc Hz."""
        wc = 2 * math.pi * fc
        out = []
        for b in res.branches:
            L = float(b.L) * r0 / wc if b.L is not None else None
            C = float(b.C) / (r0 * wc) if b.C is not None else None
            out.append(Branch(b.pos, b.kind, L=L, C=C))
        return cls(out, r_source=float(res.r_source) * r0, r_load=float(res.r_load) * r0, title=title)

    def to_highpass(self, fc_lp_proto_scaled=None):
        """Lowpass -> highpass at the same cutoff: L <-> C with L' = 1/(wc^2 C) etc.
        Assumes this ladder is a lowpass denormalised to cutoff fc (wc)."""
        wc2 = (2 * math.pi * self.notes["fc"]) ** 2
        out = []
        for b in self.branches:
            if b.kind == 'L':
                out.append(Branch(b.pos, 'C', C=1 / (wc2 * b.L)))
            elif b.kind == 'C':
                out.append(Branch(b.pos, 'L', L=1 / (wc2 * b.C)))
            else:  # resonators keep their type; L and C swap roles
                out.append(Branch(b.pos, b.kind, L=1 / (wc2 * b.C), C=1 / (wc2 * b.L)))
        return Ladder(out, self.r_source, self.r_load, self.title + " (HP)", dict(self.notes), self.q_l, self.q_c)

    # ----------------------------------------------------------- analysis
    def abcd(self, f):
        """ABCD matrices at frequencies f (Hz): returns array (n, 2, 2)."""
        f = np.atleast_1d(np.asarray(f, dtype=float))
        s = 1j * 2 * np.pi * f
        w = np.abs(2 * np.pi * f)
        M = np.zeros((len(f), 2, 2), complex)
        M[:, 0, 0] = M[:, 1, 1] = 1
        ql, qc = self.q_l, self.q_c

        def zl(L):  # inductor impedance with series loss R = wL/Q
            return s * L + (w * L / ql if ql else 0)

        def yc(C):  # capacitor admittance with shunt loss G = wC/Q
            return s * C + (w * C / qc if qc else 0)

        for b in self.branches:
            if b.kind == 'L':
                x = zl(b.L)
            elif b.kind == 'C':
                x = yc(b.C)
            elif b.kind == 'LC_par':          # parallel L||C
                x = 1 / (1 / zl(b.L) + yc(b.C)) if b.pos == 'series' else (1 / zl(b.L) + yc(b.C))
            elif b.kind == 'LC_ser':          # series L+C
                x = (zl(b.L) + 1 / yc(b.C)) if b.pos == 'series' else 1 / (zl(b.L) + 1 / yc(b.C))
            elif b.kind == 'TEM':             # shorted line to ground || loading capacitor
                th = np.radians(b.theta) * f / b.f_ref
                y_line = 1 / (1j * b.z0 * np.tan(th) + 1e-300)
                x = y_line + (yc(b.C) if b.C else 0)
            else:
                raise ValueError(b.kind)
            # x is an impedance for series branches and an admittance for shunt branches
            if b.pos == 'series' and b.kind == 'C':
                x = 1 / x
            if b.pos == 'shunt' and b.kind == 'L':
                x = 1 / x
            T = np.zeros_like(M)
            T[:, 0, 0] = T[:, 1, 1] = 1
            if b.pos == 'series':
                T[:, 0, 1] = x
            else:
                T[:, 1, 0] = x
            M = M @ T
        return M

    @property
    def single_terminated(self):
        """'open' or 'short' for a singly terminated ladder (load infinite or zero), else None."""
        if math.isinf(self.r_load):
            return "open"
        if self.r_load == 0:
            return "short"
        return None

    def sparams(self, f):
        """S11 and S21 at frequencies f (Hz), referenced to r_source and r_load.

        Singly terminated ladders (r_load = inf or 0) report as 'S21' the normalised
        transfer into the unterminated end: V2/E with the end open, Rs*I2/E with it
        shorted (1 at DC for a lowpass, like S21 of a matched doubly terminated filter);
        S11 is the reflection at the terminated input."""
        M = self.abcd(f)
        A, B, C, D = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
        rs, rl = self.r_source, self.r_load
        st = self.single_terminated
        if st == "open":
            zin = A / C
            s21 = 1 / (A + C * rs)
        elif st == "short":
            zin = B / D
            s21 = rs / (B + D * rs)
        else:
            den = A * rl + B + C * rs * rl + D * rs
            s21 = 2 * np.sqrt(rs * rl) / den
            s11 = (A * rl + B - C * rs * rl - D * rs) / den
            return s11, s21
        s11 = (zin - rs) / (zin + rs)
        return s11, s21

    def s_matrix(self, f, z0=50.0):
        """Full 2-port S-parameters (S11, S21, S12, S22) referenced to z0 at both ports,
        as a network analyser would measure the bare filter (terminations not included)."""
        M = self.abcd(f)
        A, B, C, D = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
        den = A + B / z0 + C * z0 + D
        s11 = (A + B / z0 - C * z0 - D) / den
        s21 = 2 / den
        s12 = 2 * (A * D - B * C) / den        # = S21 for a reciprocal (passive) ladder
        s22 = (-A + B / z0 - C * z0 + D) / den
        return s11, s21, s12, s22

    def touchstone(self, f, z0=50.0, comment=""):
        """Touchstone .s2p text in dB/angle format: '# Hz S DB R <z0>'.
        Column order per line: f, S11, S21, S12, S22 (each as dB and angle in degrees)."""
        f = np.asarray(f, float)
        sp = self.s_matrix(f, z0)
        lines = [f"! {self.title}", "! Created by VektorFIL"]
        lines += [f"! {c}" for c in comment.splitlines() if c]
        lines.append(f"# Hz S DB R {z0:g}")
        for i, fi in enumerate(f):
            vals = []
            for s in sp:
                v = s[i]
                vals += [20 * np.log10(abs(v) + 1e-300), np.degrees(np.angle(v))]
            lines.append(f"{fi:.10g} " + " ".join(f"{x:.6f}" for x in vals))
        return "\n".join(lines) + "\n"

    def response(self, f):
        """dict with S11/S21 dB, phase (deg) and group delay (s)."""
        f = np.asarray(f, float)
        s11, s21 = self.sparams(f)
        # at a transmission zero S21 passes through zero and its phase flips by 180 deg;
        # that flip is not delay, so unwrap with a 180 deg period (the true phase of a
        # lossless ladder never jumps by 180 deg between neighbouring points otherwise)
        ph = np.unwrap(np.angle(s21), period=np.pi)
        gd = -np.gradient(ph, 2 * np.pi * f) if len(f) > 1 else np.full(len(f), np.nan)
        return {"f": f, "s11_db": 20 * np.log10(np.abs(s11) + 1e-300),
                "s21_db": 20 * np.log10(np.abs(s21) + 1e-300),
                "phase_deg": np.degrees(ph), "gd": gd}

    def designators(self):
        """Per branch: (L name or None, C name or None).

        Inductors and capacitors are numbered separately in order along the ladder
        (L1, L2, ... / C1, C2, ...); a TEM line resonator takes the first slot as TL1,
        TL2, ...  The schematic, the element table and the netlist export all use this,
        so the names always agree."""
        nl = nc = nt = 0
        out = []
        for b in self.branches:
            ln = cn = None
            if b.kind == "TEM":
                nt += 1
                ln = f"TL{nt}"
            elif b.L is not None:
                nl += 1
                ln = f"L{nl}"
            if b.C is not None:
                nc += 1
                cn = f"C{nc}"
            out.append((ln, cn))
        return out

    def table(self):
        """Plain-text element list, one line per component, e.g.
        'L2       26.95 nH   series  (series resonator L2+C3, f0 = 100 MHz)'."""
        kind_txt = {"LC_par": "parallel resonator", "LC_ser": "series resonator"}
        rows = [f"Source  {self.r_source:.6g} ohm"]
        for b, (ln, cn) in zip(self.branches, self.designators()):
            note = ""
            if b.kind == "TEM":
                rows.append(f"{ln:<6s}  Z0={b.z0:.5g} ohm, {b.theta:.4g} deg at {_eng(b.f_ref, 'Hz')}"
                            f"   shunt   (shorted line, loaded by {cn})")
                rows.append(f"{cn:<6s}  {_eng(b.C, 'F'):>12s}   shunt   (loading capacitor of {ln})")
                continue
            if b.L is not None and b.C is not None:
                f0 = 1 / (2 * math.pi * math.sqrt(b.L * b.C))
                note = f"  ({kind_txt[b.kind]} {ln}+{cn}, f0 = {_eng(f0, 'Hz')})"
            if b.L is not None:
                rows.append(f"{ln:<6s}  {_eng(b.L, 'H'):>12s}   {b.pos:<6s}{note}")
            if b.C is not None:
                rows.append(f"{cn:<6s}  {_eng(b.C, 'F'):>12s}   {b.pos:<6s}{note}")
        st = self.single_terminated
        rows.append(f"Load    {st} (singly terminated)" if st else f"Load    {self.r_load:.6g} ohm")
        return "\n".join(rows)


def _eng(x, unit):
    """Engineering notation with 5 significant digits for the element table."""
    if x == 0:
        return f"0 {unit}"
    pre = [(1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k'), (1, ''), (1e-3, 'm'), (1e-6, 'u'),
           (1e-9, 'n'), (1e-12, 'p'), (1e-15, 'f')]
    for k, p in pre:
        if abs(x) >= k:
            return f"{x / k:.5g} {p}{unit}"
    return f"{x:.4g} {unit}"
