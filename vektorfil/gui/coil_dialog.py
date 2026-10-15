"""Tools -> Air-core coil: turns of wire for an inductor of the current design.

Pick an inductor of the design (or type a value), the form diameter, the wire gauge
and the spacing; the turns, winding length and wire length update as you type.  The
calculation is vektorfil.analysis.coil (Wheeler's 1982 single-layer formula).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel,
                               QLineEdit)

from vektorfil.analysis.coil import awg_mm, design_coil

AWG = range(10, 33)


class CoilDialog(QDialog):
    """Live air-core coil calculator."""
    def __init__(self, parent, inductors=()):
        """inductors: (designator, henry) pairs of the current design."""
        super().__init__(parent)
        self.setWindowTitle("Air-core coil")
        lay = QFormLayout(self)
        self.c_pick = QComboBox()
        self.c_pick.addItem("Typed value", None)
        for name, value in inductors:
            from vektorfil.gui.schematic import eng
            self.c_pick.addItem(f"{name}   {eng(value, 'H')}", value)
        self.e_l = QLineEdit("100")
        self.e_form = QDoubleSpinBox()
        self.e_form.setRange(0.5, 200.0)
        self.e_form.setDecimals(2)
        self.e_form.setValue(5.0)
        self.e_form.setSuffix(" mm")
        self.c_awg = QComboBox()
        for g in AWG:
            self.c_awg.addItem(f"AWG {g}  ({awg_mm(g):.3f} mm)", g)
        self.c_awg.setCurrentIndex(list(AWG).index(22))
        self.e_pitch = QDoubleSpinBox()
        self.e_pitch.setRange(0.0, 50.0)
        self.e_pitch.setDecimals(3)
        self.e_pitch.setSuffix(" mm")
        self.e_pitch.setSpecialValueText("close wound")
        lay.addRow("Inductor", self.c_pick)
        lay.addRow("Inductance (nH)", self.e_l)
        lay.addRow("Form diameter", self.e_form)
        lay.addRow("Wire", self.c_awg)
        lay.addRow("Turn spacing", self.e_pitch)
        self.out = QLabel()
        self.out.setTextFormat(Qt.RichText)
        self.out.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addRow(self.out)
        note = QLabel("Single-layer air core, Wheeler's formula (about 0.1 % for the current sheet). "
                      "Leads, wire thickness effects and self-capacitance are not included: trim by "
                      "spreading or squeezing the turns.")
        note.setWordWrap(True)
        lay.addRow(note)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self.c_pick.currentIndexChanged.connect(self._picked)
        self.e_l.textChanged.connect(self.recalc)
        self.e_form.valueChanged.connect(self.recalc)
        self.c_awg.currentIndexChanged.connect(self.recalc)
        self.e_pitch.valueChanged.connect(self.recalc)
        if inductors:
            self.c_pick.setCurrentIndex(1)
        self.recalc()

    def _picked(self):
        v = self.c_pick.currentData()
        if v is not None:
            self.e_l.setText(f"{v * 1e9:.5g}")

    def recalc(self):
        """Update the turns and lengths from the current inputs."""
        from vektorfil.gui.filter_window import parse_value
        try:
            lt = parse_value(self.e_l.text(), 1e-9)
            if not lt:
                raise ValueError("enter an inductance")
            wire = awg_mm(self.c_awg.currentData()) * 1e-3
            pitch = self.e_pitch.value() * 1e-3
            d = design_coil(lt, self.e_form.value() * 1e-3, wire, pitch or None)
        except Exception as e:
            self.out.setText(f"<i>{e}</i>")
            return
        self.out.setText(
            f"Turns: <b>{d.turns:.2f}</b> &nbsp;(<b>{d.turns_whole}</b> turns give {d.l_whole * 1e9:.4g} nH)<br>"
            f"Winding length: <b>{d.length * 1e3:.2f} mm</b> &nbsp; pitch {d.pitch * 1e3:.3f} mm<br>"
            f"Coil diameter to wire centre: {d.mean_diameter * 1e3:.2f} mm<br>"
            f"Wire in the winding: {d.wire_length * 1e3:.0f} mm (plus leads)")
