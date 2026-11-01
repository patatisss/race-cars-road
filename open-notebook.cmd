@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m jupyterlab --ServerApp.root_dir="%CD%" experiment.ipynb
