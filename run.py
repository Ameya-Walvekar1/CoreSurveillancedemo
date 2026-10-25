import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
MODEL = ROOT / "best.pt"


def fail(message):
    print(f"\nERROR: {message}")
    raise SystemExit(1)


def main():
    # Validate the Python version used to bootstrap the environment.
    if sys.version_info < (3, 10):
        fail("Python 3.10 or newer is required.")

    if not REQUIREMENTS.is_file():
        fail("requirements.txt was not found.")

    if not MODEL.is_file():
        fail("best.pt was not found in the project folder.")

    # Locate the virtual environment's Python executable.
    if os.name == "nt":
        venv_python = VENV / "Scripts" / "python.exe"
    else:
        venv_python = VENV / "bin" / "python"

    # Create a fresh virtual environment when necessary.
    if not venv_python.is_file():
        print("Creating Python virtual environment...")
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV)],
            check=True,
            cwd=ROOT,
        )

    # Install/update dependencies using the environment's Python.
    print("Installing/checking project dependencies...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS),
        ],
        check=True,
        cwd=ROOT,
    )

    # Make the bundled trained checkpoint independent of local paths.
    env = os.environ.copy()
    env["MODEL_PATH"] = str(MODEL)

    print("\nStarting Traffic Surveillance...")
    print("When Gradio starts, open the local URL printed below.")
    print("Press Ctrl+C to stop the application.\n")

    subprocess.run(
        [str(venv_python), str(ROOT / "app.py")],
        cwd=ROOT,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        fail(
            f"A setup or application command failed with exit code "
            f"{exc.returncode}. Check the error above."
        )
    except KeyboardInterrupt:
        print("\nApplication stopped.")
