@echo off
rem Start the VektorFIL lumped filter designer
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m vektorfil.gui.filter_window
