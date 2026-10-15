# VektorFIL

by Pieter Ibelings

An educational RF filter design tool.  The goal: a simple, interactive workflow -- pick
a filter, type values, see the schematic and response update live -- on top of a real
network-synthesis engine rather than tables.

VektorFIL was created from scratch by Pieter Ibelings with the help of Claude (Anthropic's
AI assistant).  Everything is written from published theory (see *References*); no code or
data from other programs is used.  Worked examples from the public-domain S/FILSYN manuals
serve only as reference results to test against.

## Download

Get the program from the [latest release](https://github.com/ibelinp/vektorfil/releases/latest).
No Python needed. `1.1.0` below stands for the version you download.

### Windows

Run **VektorFIL-1.1.0-setup.exe** (installs for the current user, no administrator rights
needed), or unzip **VektorFIL-1.1.0-windows.zip** anywhere and run `VektorFIL.exe`.
If Windows SmartScreen warns, click **More info -> Run anyway**.

### macOS (Apple silicon: M1 or newer)

1. Download **VektorFIL-1.1.0-macos-arm64.zip** and double-click it to unzip.
2. Drag **VektorFIL.app** into your **Applications** folder.
3. Open it. The first time, macOS blocks it because the app is not notarized by Apple
   (that requires a paid developer account). Click **Done**, then open
   **System Settings -> Privacy & Security**, scroll down and click **Open Anyway** next to
   the VektorFIL message, and confirm. From then on it opens normally.

If macOS says the app "is damaged", open Terminal and run
`xattr -dr com.apple.quarantine /Applications/VektorFIL.app`, then open it again.
Intel Macs: run from source (below).

### Linux (x86-64: Ubuntu 22.04, Debian 12, Fedora 36 or newer)

```bash
tar xzf VektorFIL-1.1.0-linux-x86_64.tar.gz
VektorFIL/VektorFIL
```

If Qt reports a missing "xcb" plugin, install `libxcb-cursor0` (Ubuntu/Debian) or
`xcb-util-cursor` (Fedora).

The macOS and Linux builds are made and self-tested by GitHub Actions
(`.github/workflows/build.yml`, `tools/build_unix.py`); every push also runs the tests on
all three systems.

## Running from source

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m vektorfil                        # or double-click "VektorFIL Filter.bat"
.venv/Scripts/python -m pytest tests                    # regression tests
.venv/Scripts/python -m vektorfil --selftest            # quick check without a window
```

## Making a release

1. **Version**: set `__version__` in `vektorfil/__init__.py` (e.g. `1.2.0`), run the tests,
   commit and push.  Every push runs the tests on Windows, macOS and Linux on GitHub
   (Actions tab), and builds the macOS and Linux versions as a check.
2. **Windows build** on a Windows PC:

   ```bash
   .venv/Scripts/python -m pip install -r tools/requirements-build.txt
   .venv/Scripts/python tools/build_windows.py
   ```

   This packages VektorFIL with PyInstaller (a few minutes), runs the self-test on the result,
   and writes the program folder, a portable zip and an Inno Setup installer to
   `%USERPROFILE%/VektorFIL-build` (kept outside synced folders on purpose; set
   `VEKTORFIL_BUILD` to change it).  The installer needs Inno Setup 6 (`ISCC.exe`), either
   installed normally or placed in `_ref/tools/innosetup`.  `tools/collect_licenses.py`
   refreshes `THIRD_PARTY_LICENSES.txt` and runs as part of the build.  A native-code build
   with Nuitka is available with `--nuitka`.
3. **Tag**: `git tag v1.2.0` and `git push origin v1.2.0`.
4. **Release** on GitHub (Releases -> Draft a new release): choose the tag, write the notes,
   attach the Windows `-setup.exe` and `-windows.zip`, and publish.
5. **macOS and Linux** files are then built, self-tested and attached to the release by
   GitHub Actions within about 15 minutes (`.github/workflows/build.yml`, which runs
   `tools/build_unix.py`).  For a release published before the workflow ran, use
   Actions -> Build -> Run workflow with the release tag.

`tools/requirements-build.txt` pins the library versions used for the releases, so all three
systems are built from the same versions.  The macOS app is signed ad hoc only (no paid Apple
developer account), hence the one-time "Open Anyway" for users.

## What it does

| Response | Approximations | Circuits |
|---|---|---|
| Lowpass / Highpass | Butterworth, Chebyshev, Bessel, Elliptic (odd; even type B/C), singly terminated Butterworth/Chebyshev/Elliptic | Shunt element first, series element first |
| Bandpass, all-pole | Butterworth, Chebyshev, Bessel | Shunt/series element first (full transform); shunt resonators with capacitive or inductive coupling; series resonators with shunt-C coupling; tubular; tumbular; coaxial resonators with capacitive coupling; quasi-symmetric (Carassa) |
| Bandpass, elliptic | Elliptic | Shunt/series element first (full transform); zig-zag with fewest inductors (Saal–Ulbrich, even orders) |
| Bandstop | all | Shunt element first, series element first |

Singly terminated filters (source resistance on one side, the other end open or shorted)
are the building blocks of diplexers and multiplexers.

Options: edge attenuation, elliptic by Amin, Fs or a stepped stopband mask, end coils for a
wider tubular range, component Q, a live order estimate.  Extras: standard component values
(E24/E48/E96 with automatic snap-and-retune), exact band edges for coupled-resonator
filters, plot markers, Smith chart, step and impulse response, Monte Carlo tolerance
analysis, an air-core coil calculator, design files, PDF reports, SPICE and Touchstone
(.s2p) export.

## Code map

```
vektorfil/
  synth/                 the synthesis engine (pure math, no GUI)
    poly.py              high-precision (mpmath, 50 digits) polynomial helpers
    approx.py            lowpass approximations: E(s), F(s), P(s) for Butterworth,
                         Chebyshev, Bessel, elliptic, and equiripple with arbitrary zeros
    placer.py            PLACER: minimax placement of transmission zeros against a
                         stopband mask; even-order elliptic types B and C
    ladder.py            Darlington ladder synthesis by zero shifting (lowpass)
    bandpass.py          direct bandpass synthesis: bandpass E/F/P (geometric or
                         equiripple with any DC/infinity zero split) and a search over
                         extraction sequences (finds e.g. the zig-zag filter)
    bandxform.py         exact element-wise lowpass -> bandpass / bandstop transforms
    coupled.py           narrow-band coupled-resonator bandpass (top C, top L, shunt C,
                         tubular, top C with TEM line resonators) from external Q and
                         coupling coefficients
    symmetric.py         symmetry-preserving bandpass (inverter design + refinement)
    network.py           Ladder: element list, scaling, ABCD analysis, S-parameters
  filters/               the interactive design layer
    design.py            FilterSpec -> design() -> DesignResult (ladder + info); circuit
                         names; stepped stopband masks; order estimate
    stdvalues.py         E-series values, snapping with re-tuning
    diplexer.py          lowpass/highpass diplexer from singly terminated channels
    project.py           design files (.vfil, JSON)
  analysis/              tools that work on a finished design (plain numpy)
    montecarlo.py        component-tolerance spread of the response
    timedomain.py        step and impulse response by inverse FFT
    coil.py              single-layer air-core coils (Wheeler 1982)
  gui/
    filter_window.py     main window (PySide6 / Qt), live recompute in a worker thread,
                         markers, design files, menus, About box, --selftest
    diplexer_window.py   diplexer designer (Tools -> Diplexer...)
    schematic.py         schematic drawing with measured, non-overlapping labels
    smith.py             Smith chart of S11
    montecarlo_tab.py    the Monte Carlo tab
    coil_dialog.py       Tools -> Air-core coil
    report.py            File -> Export report (PDF)
    plotstyle.py         shared plot colours and styling
  app_paths.py           where logs (AppData) and bundled resources live
tests/                   pytest regression tests (closed-form values, published
                         S/FILSYN examples, simulated responses)
examples/                small scripts (plots)
tools/                   release tooling: build_windows.py (Windows installer and zip),
                         build_unix.py (macOS app / Linux archive), requirements-build.txt
                         (pinned versions), collect_licenses.py, make_icon.py
.github/workflows/       build.yml: tests on three systems, macOS and Linux release builds
installer/               Inno Setup script for the Windows installer
assets/                  program icon
_ref/                    reference material: S/FILSYN downloads, extracted manual text,
                         DESIGN_NOTES.md (summary of the S/FILSYN manuals), tools
```

### How a design is made

1. **Approximation** (`approx.py`, `placer.py`, `bandpass.equiripple`): build the
   polynomials of the transfer function, E(s) (Hurwitz), F(s) (reflection) and P(s)
   (transmission), with E·E* = F·F* + P·P* (Feldtkeller).  Equiripple passbands with
   arbitrary transmission zeros use Cameron's generalized-Chebyshev recursion (lowpass) or
   a weighted Remez exchange in ω² (bandpass).
2. **Synthesis** (`ladder.py`, `bandpass.py`): the input impedance Z = (E+F)/(E−F) is
   taken apart one transmission zero at a time — full removal of poles at DC or infinity,
   *zero shifting* by partial removal, then extraction of the resonator.  All of this runs
   at 50 significant digits; S/FILSYN used 128-bit arithmetic for the same reason.
3. **Realization** (`design.py`, `bandxform.py`, `coupled.py`): scale to the requested
   impedance and frequency, transform, or build coupled-resonator forms.
4. **Analysis** (`network.py`): ABCD-matrix chain → S11, S21, group delay, with optional
   component Q.

Every design is checked by simulating the finished ladder against the transfer function
it was built from; the tests do the same.

## References

* G. L. Matthaei, L. Young, E. M. T. Jones, *Microwave Filters, Impedance-Matching
  Networks, and Coupling Structures* (prototype values, coupled resonators).
* A. I. Zverev, *Handbook of Filter Synthesis* (synthesis, tables).
* R. J. Cameron, "General coupling matrix synthesis methods for Chebyshev filtering
  functions", IEEE MTT, 1999 (generalized Chebyshev recursion).
* R. Saal, E. Ulbrich, "On the design of filters by synthesis", IRE Trans. CT, 1958
  (zero shifting, minimum-inductor bandpass).
* F. Carassa, "Band-pass filters having quasi-symmetrical attenuation and group-delay
  characteristics", Alta Frequenza, 1961 (zero ratio for symmetric bandpass).
* S. B. Cohn, "Direct-coupled-resonator filters", Proc. IRE, 1957.
* G. Szentirmai, S/FILSYN manuals (workflow, PLACER, examples) — see `_ref/sfilsyn`.
* R. Rhea, *HF Filter Design and Computer Simulation*.

## Contributions

VektorFIL is provided as-is.  Issues, feature requests and pull requests are not accepted.
You are welcome to fork it and change it in any way the GPLv3 allows.

## License

Copyright (C) 2026 Pieter Ibelings

VektorFIL is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.  It is distributed in the hope that it
will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the [LICENSE](LICENSE) file.

Third-party libraries keep their own licenses: PySide6 / Qt (LGPLv3), numpy, scipy and
mpmath (BSD), pyqtgraph (MIT).

