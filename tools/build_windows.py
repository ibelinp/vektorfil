"""Build the Windows release of VektorFIL: a program folder, a portable zip and an installer.

    .venv\\Scripts\\python tools\\build_windows.py                 (PyInstaller, a few minutes)
    .venv\\Scripts\\python tools\\build_windows.py --nuitka        (Nuitka native compile, slow)
    .venv\\Scripts\\python tools\\build_windows.py --installer-only

Steps
  1. refresh THIRD_PARTY_LICENSES.txt (and the icon if missing);
  2. build the program folder <build>/VektorFIL/ -- a folder, not a one-file exe, so the Qt
     libraries stay separate, replaceable DLLs as the LGPL requires;
  3. run the built program's --selftest (result in %LOCALAPPDATA%/VektorFIL/selftest.txt);
  4. write <build>/VektorFIL-<version>-windows.zip (portable version);
  5. package the folder with Inno Setup into <build>/VektorFIL-<version>-setup.exe
     (ISCC.exe is looked up in _ref/tools/innosetup or the standard install locations).

<build> is %USERPROFILE%/VektorFIL-build unless VEKTORFIL_BUILD is set; it is kept outside
synced folders (Dropbox, OneDrive) on purpose.

PyInstaller (pip install pyinstaller) is the default: it bundles the official, tested
builds of numpy/scipy/Qt.  The Nuitka route (pip install nuitka ordered-set zstandard)
compiles everything to C with the Zig compiler; with Python 3.13 / Nuitka 4.3 its build
crashed at run time, so it is kept only as an option.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("VEKTORFIL_BUILD", Path.home() / "VektorFIL-build"))
sys.path.insert(0, str(ROOT))
from vektorfil import __version__  # noqa: E402

PY = sys.executable
DIST = BUILD / "VektorFIL"
V4 = ".".join((__version__.split(".") + ["0", "0", "0"])[:4])
EXCLUDE = ["matplotlib", "PIL", "pytest", "IPython", "tkinter", "pandas", "h5py", "cupy",
           "pyqtgraph.examples", "pyqtgraph.opengl", "OpenGL"]


def run(cmd, **kw):
    print("\n>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


# --------------------------------------------------------------------------- PyInstaller

def _version_file(path):
    """Windows version resource for the exe (shown in Explorer -> Properties)."""
    t = tuple(int(x) for x in V4.split("."))
    path.write_text(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={t}, prodvers={t}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Pieter Ibelings'),
      StringStruct('FileDescription', 'VektorFIL - L-C filter and diplexer synthesis'),
      StringStruct('FileVersion', '{__version__}'),
      StringStruct('InternalName', 'VektorFIL'),
      StringStruct('LegalCopyright', 'Copyright (C) 2026 Pieter Ibelings - GNU GPL v3'),
      StringStruct('OriginalFilename', 'VektorFIL.exe'),
      StringStruct('ProductName', 'VektorFIL'),
      StringStruct('ProductVersion', '{__version__}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""", encoding="utf-8")


def pyinstaller():
    work = BUILD / "pyi-work"
    work.mkdir(parents=True, exist_ok=True)
    vfile = work / "version_info.txt"
    _version_file(vfile)
    sep = os.pathsep            # ';' on Windows
    data = [(ROOT / "docs" / "index.html", "docs"), (ROOT / "docs" / "img", "docs/img"),
            (ROOT / "assets", "assets"), (ROOT / "LICENSE", "."),
            (ROOT / "THIRD_PARTY_LICENSES.txt", ".")]
    cmd = [PY, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed",
           "--name", "VektorFIL", "--icon", ROOT / "assets" / "vektorfil.ico",
           "--version-file", vfile,
           "--distpath", BUILD, "--workpath", work, "--specpath", work,
           "--collect-submodules", "pyqtgraph", "--collect-data", "pyqtgraph"]
    for src, dst in data:
        cmd += ["--add-data", f"{src}{sep}{dst}"]
    for mod in EXCLUDE:
        cmd += ["--exclude-module", mod]
    cmd.append(ROOT / "VektorFIL.py")
    run(cmd)


# --------------------------------------------------------------------------- Nuitka (optional)

def nuitka():
    run([PY, "-m", "nuitka",
         "--standalone", "--zig", "--assume-yes-for-downloads",
         "--enable-plugin=pyside6", "--include-package=pyqtgraph",
         "--nofollow-import-to=" + ",".join(EXCLUDE),
         "--windows-console-mode=attach",
         f"--windows-icon-from-ico={ROOT / 'assets' / 'vektorfil.ico'}",
         "--company-name=Pieter Ibelings", "--product-name=VektorFIL",
         f"--file-version={V4}", f"--product-version={V4}",
         "--file-description=VektorFIL - L-C filter and diplexer synthesis",
         "--copyright=Copyright (C) 2026 Pieter Ibelings - GNU GPL v3",
         f"--include-data-dir={ROOT / 'docs' / 'img'}=docs/img",
         f"--include-data-files={ROOT / 'docs' / 'index.html'}=docs/index.html",
         f"--include-data-dir={ROOT / 'assets'}=assets",
         f"--include-data-files={ROOT / 'LICENSE'}=LICENSE",
         f"--include-data-files={ROOT / 'THIRD_PARTY_LICENSES.txt'}=THIRD_PARTY_LICENSES.txt",
         f"--output-dir={BUILD}", "--output-filename=VektorFIL.exe", "--remove-output",
         ROOT / "VektorFIL.py"])
    nd = BUILD / "VektorFIL.dist"
    if DIST.exists():
        shutil.rmtree(DIST)
    nd.rename(DIST)


# --------------------------------------------------------------------------- checks / packaging

def selftest():
    exe = DIST / "VektorFIL.exe"
    report = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VektorFIL" / "selftest.txt"
    if report.exists():
        report.unlink()
    print("\n>> self-test of", exe, flush=True)
    r = subprocess.run([str(exe), "--selftest"], capture_output=True, text=True, timeout=900)
    text = report.read_text(encoding="utf-8") if report.exists() else (r.stdout + r.stderr)
    print(text)
    if r.returncode != 0 or "SELFTEST PASSED" not in text:
        raise SystemExit(f"self-test FAILED (exit code {r.returncode})")


def find_iscc():
    cands = [ROOT / "_ref" / "tools" / "innosetup" / "ISCC.exe",
             Path(os.environ.get("ProgramFiles(x86)", "")) / "Inno Setup 6" / "ISCC.exe",
             Path(os.environ.get("ProgramFiles", "")) / "Inno Setup 6" / "ISCC.exe",
             Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe"]
    for c in cands:
        if c.exists():
            return c
    found = shutil.which("ISCC")
    return Path(found) if found else None


def installer():
    iscc = find_iscc()
    if iscc is None:
        print("Inno Setup (ISCC.exe) not found: skipping the installer.")
        return
    run([iscc, f"/DMyAppVersion={__version__}", f"/DSourceDir={DIST}", f"/DOutputDir={BUILD}",
         f"/DRootDir={ROOT}", ROOT / "installer" / "vektorfil.iss"])


def portable_zip():
    base = BUILD / f"VektorFIL-{__version__}-windows"
    shutil.make_archive(str(base), "zip", root_dir=BUILD, base_dir=DIST.name)
    print("wrote", f"{base}.zip")


def main():
    if "--installer-only" not in sys.argv:
        run([PY, ROOT / "tools" / "collect_licenses.py"])
        if not (ROOT / "assets" / "vektorfil.ico").exists():
            run([PY, ROOT / "tools" / "make_icon.py"])
        if "--nuitka" in sys.argv:
            nuitka()
        else:
            pyinstaller()
        selftest()
        portable_zip()
    installer()
    print("\nDone. Output in", BUILD)


if __name__ == "__main__":
    main()
