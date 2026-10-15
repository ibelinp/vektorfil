"""File -> Export report: a one-page (or longer) PDF of the current design.

Contents: title and date, the specification summary, the response plot, the schematic
and the element list.  Images are taken from the window's own widgets, so the report
shows exactly what is on screen (same plot range, markers, theme).
"""
from __future__ import annotations

import datetime

from PySide6.QtCore import QMarginsF, QRectF, Qt
from PySide6.QtGui import (QFont, QFontDatabase, QFontMetricsF, QPageLayout, QPageSize, QPainter, QPdfWriter,
                           QTextDocument)

from vektorfil import __version__


def _plain(html: str) -> str:
    """Rich text (the summary) as plain text."""
    doc = QTextDocument()
    doc.setHtml(html)
    return doc.toPlainText()


def write_report(path, title, summary_html, images, table_text):
    """images: QPixmaps (plot, schematic) in order; table_text: the element list."""
    pdf = QPdfWriter(str(path))
    pdf.setPageSize(QPageSize(QPageSize.A4))
    pdf.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Millimeter)
    pdf.setResolution(300)
    pdf.setTitle(title)
    pdf.setCreator(f"VektorFIL {__version__}")
    p = QPainter(pdf)
    page = QRectF(0, 0, pdf.width(), pdf.height())
    y = 0.0

    def need(h):
        nonlocal y
        if y + h > page.height() and y > 0:
            pdf.newPage()
            y = 0.0

    def text(s, size, bold=False, mono=False):
        nonlocal y
        # the system's own UI or fixed-width font, so the report looks right on every platform
        f = QFont(QFontDatabase.systemFont(QFontDatabase.FixedFont if mono else QFontDatabase.GeneralFont))
        f.setPointSizeF(size)
        f.setBold(bold)
        p.setFont(f)
        fm = QFontMetricsF(f, pdf)
        for line in s.splitlines() or [""]:
            need(fm.height())
            p.drawText(QRectF(0, y, page.width(), fm.height()), Qt.AlignLeft | Qt.AlignVCenter, line)
            y += fm.height()

    text(title, 16, bold=True)
    text(f"VektorFIL {__version__} — {datetime.datetime.now():%Y-%m-%d %H:%M}", 8)
    y += 30
    text(_plain(summary_html), 9)
    y += 40
    for pm in images:
        if pm is None or pm.isNull():
            continue
        w = page.width()
        h = w * pm.height() / pm.width()
        if h > page.height() * 0.6:
            h = page.height() * 0.6
            w = h * pm.width() / pm.height()
        need(h)
        p.drawPixmap(QRectF(0, y, w, h), pm, QRectF(pm.rect()))
        y += h + 40
    text(table_text, 8, mono=True)
    p.end()
