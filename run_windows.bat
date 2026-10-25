@echo off
cd /d "%~dp0"

if not exist best.pt (
    echo ERROR: best.pt is missing from this folder.
    pause
    exit /b 1
)

if not exist .venv (
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Could not create a virtual environment.
        echo Install Python 3 and try again.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
if errorlevel 1 goto error

python -m pip install -r requirements.txt
if errorlevel 1 goto error

set "MODEL_PATH=%CD%\best.pt"
python app.py
pause
exit /b 0

:error
echo Dependency installation failed.
pause
exit /b 1
