"""Build the macOS or Linux release of VektorFIL (run on that system; GitHub Actions does
this for every release, see .github/workflows/build.yml).

    python tools/build_unix.py

Steps
  1. package the program with PyInstaller (a folder build, like the Windows release);
     on macOS an application bundle VektorFIL.app with the version in its Info.plist;
  2. run the built program's --selftest off-screen;
  3. write the download archive to <build>/:
        macOS  VektorFIL-<version>-macos-arm64.zip      (VektorFIL.app)
        Linux  VektorFIL-<version>-linux-x86_64.tar.gz  (folder VektorFIL/ with the program)

<build> is ./dist-release unless VEKTORFIL_BUILD is set.  The builds are not signed by
Apple (that needs a paid developer account); macOS users open the app the first time
with right-click -> Open.
"""
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("VEKTORFIL_BUILD", ROOT / "dist-release"))
sys.path.insert(0, str(ROOT))
from vektorfil import __version__, app_paths  # noqa: E402

PY = sys.executable
MAC = sys.platform == "darwin"
ARCH = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(
    platform.machine().lower(), platform.machine().lower())
EXCLUDE = ["matplotlib", "PIL", "pytest", "IPython", "tkinter", "pandas", "h5py", "cupy",
           "pyqtgraph.examples", "pyqtgraph.opengl", "OpenGL"]


def annotate(title, text):
    """On GitHub Actions, show text as an error annotation on the run page (readable
    without logging in); elsewhere just print it."""
    if os.environ.get("GITHUB_ACTIONS"):
        tail = "\n".join(text.strip().splitlines()[-40:])
        print(f"::error title={title}::" + tail.replace("%", "%25").replace("\r", "").replace("\n", "%0A"),
              flush=True)
    else:
        print(title, text, sep="\n")


def run(cmd, **kw):
    print("\n>>", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True, **kw)
    print(r.stdout[-20000:], r.stderr[-20000:], sep="\n", flush=True)
    if r.returncode != 0:
        annotate(f"{Path(str(cmd[0])).name} failed ({r.returncode})", r.stderr or r.stdout)
        raise SystemExit(r.returncode)


def pyinstaller():
    """The program folder (Linux) or VektorFIL.app (macOS) in BUILD."""
    work = BUILD / "pyi-work"
    work.mkdir(parents=True, exist_ok=True)
    data = [(ROOT / "docs" / "index.html", "docs"), (ROOT / "docs" / "img", "docs/img"),
            (ROOT / "assets", "assets"), (ROOT / "LICENSE", "."),
            (ROOT / "THIRD_PARTY_LICENSES.txt", ".")]
    cmd = [PY, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--name", "VektorFIL",
           "--distpath", BUILD, "--workpath", work, "--specpath", work,
           "--collect-submodules", "pyqtgraph", "--collect-data", "pyqtgraph"]
    if MAC:   # PyInstaller turns the PNG into the .icns that macOS needs (with Pillow)
        cmd += ["--icon", ROOT / "assets" / "vektorfil.png",
                "--osx-bundle-identifier", "io.github.ibelinp.vektorfil"]
    for src, dst in data:
        cmd += ["--add-data", f"{src}{os.pathsep}{dst}"]
    for mod in EXCLUDE:
        cmd += ["--exclude-module", mod]
    cmd.append(ROOT / "VektorFIL.py")
    run(cmd)
    if MAC:
        _mac_version()


def _mac_version():
    """Put the version into the bundle's Info.plist, then re-sign the bundle (ad hoc):
    editing the plist invalidates the signature PyInstaller made, and Apple-silicon Macs
    refuse to start unsigned code."""
    app = BUILD / "VektorFIL.app"
    plist = app / "Contents" / "Info.plist"
    info = plistlib.loads(plist.read_bytes())
    info.update(CFBundleShortVersionString=__version__, CFBundleVersion=__version__,
                CFBundleName="VektorFIL", CFBundleDisplayName="VektorFIL",
                NSHighResolutionCapable=True,
                NSHumanReadableCopyright="Copyright (C) 2026 Pieter Ibelings - GNU GPL v3")
    plist.write_bytes(plistlib.dumps(info))
    run(["codesign", "--force", "--deep", "--sign", "-", app])


def executable() -> Path:
    if MAC:
        return BUILD / "VektorFIL.app" / "Contents" / "MacOS" / "VektorFIL"
    return BUILD / "VektorFIL" / "VektorFIL"


def selftest():
    """Run the built program's --selftest without a screen."""
    report = app_paths.data_dir() / "selftest.txt"
    if report.exists():
        report.unlink()
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    exe = executable()
    print("\n>> self-test of", exe, flush=True)
    r = subprocess.run([str(exe), "--selftest"], capture_output=True, text=True, timeout=900, env=env)
    text = report.read_text(encoding="utf-8") if report.exists() else (r.stdout + r.stderr)
    print(text)
    if r.returncode != 0 or "SELFTEST PASSED" not in text:
        print(r.stdout, r.stderr)
        annotate(f"self-test failed (exit code {r.returncode})", "\n".join((text, r.stdout, r.stderr)))
        raise SystemExit(f"self-test FAILED (exit code {r.returncode})")


def archive() -> Path:
    """The download file."""
    if MAC:
        out = BUILD / f"VektorFIL-{__version__}-macos-{ARCH}.zip"
        # ditto keeps the bundle's symlinks and permissions (a plain zip would break it)
        run(["ditto", "-c", "-k", "--keepParent", BUILD / "VektorFIL.app", out])
    else:
        out = BUILD / f"VektorFIL-{__version__}-linux-{ARCH}.tar.gz"
        with tarfile.open(out, "w:gz") as tf:
            tf.add(BUILD / "VektorFIL", arcname="VektorFIL")
    print("wrote", out)
    return out


def main():
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    pyinstaller()
    selftest()
    out = archive()
    # tell the workflow which file to publish
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a", encoding="utf-8") as fh:
            fh.write(f"file={out}\n")


if __name__ == "__main__":
    main()
