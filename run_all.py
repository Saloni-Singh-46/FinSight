"""FinSight Unified Launcher - Starts FastAPI Backend + Streamlit UI concurrently."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
import urllib.request

ROOT_DIR = Path(__file__).parent.resolve()
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
VENV_UVICORN = ROOT_DIR / ".venv" / "Scripts" / "uvicorn.exe"
VENV_STREAMLIT = ROOT_DIR / ".venv" / "Scripts" / "streamlit.exe"

# Fallbacks if running outside venv directly
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
UVICORN_EXE = str(VENV_UVICORN) if VENV_UVICORN.exists() else "uvicorn"
STREAMLIT_EXE = str(VENV_STREAMLIT) if VENV_STREAMLIT.exists() else "streamlit"


def is_port_open(port: int) -> bool:
    """Returns True when the port is free on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def find_free_port(start_port: int, max_attempts: int = 20) -> int:
    """Chooses the next available local port starting from start_port."""
    port = start_port
    for _ in range(max_attempts):
        if is_port_open(port):
            return port
        port += 1
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts}.")


def check_api_health(url: str, timeout_secs: int = 15) -> bool:
    """Polls FastAPI health endpoint until it is ready."""
    start_time = time.time()
    while time.time() - start_time < timeout_secs:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    print("=" * 70)
    print("  🚀 FINSIGHT: MULTI-AGENT AI FINANCIAL RESEARCH ANALYST")
    print("=" * 70)
    print(f"[*] Workspace Root: {ROOT_DIR}")
    print(f"[*] Python Executable: {PYTHON_EXE}")

    # Set PYTHONPATH
    env = os.environ.copy()
    src_dir = str(ROOT_DIR / "src")
    env["PYTHONPATH"] = (
        f"{src_dir};{env.get('PYTHONPATH', '')}"
        if os.name == "nt"
        else f"{src_dir}:{env.get('PYTHONPATH', '')}"
    )

    api_port = find_free_port(8000)
    ui_port = find_free_port(8501)
    processes = []

    try:
        # 1. Start FastAPI Backend
        api_url = f"http://127.0.0.1:{api_port}/health"
        print(f"\n[1/2] Starting FastAPI Backend on {api_url} ...")
        api_cmd = [
            PYTHON_EXE,
            "-m",
            "uvicorn",
            "finsight.api.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
            "--log-level",
            "info",
        ]
        api_process = subprocess.Popen(
            api_cmd,
            cwd=str(ROOT_DIR),
            env=env,
        )
        processes.append(("FastAPI Backend", api_process))

        print("[*] Waiting for FastAPI backend to initialize...")
        if check_api_health(api_url):
            print(f"    ✅ FastAPI Backend is UP and Healthy at http://127.0.0.1:{api_port}")
            print(f"    📚 OpenAPI Swagger Docs: http://127.0.0.1:{api_port}/docs")
        else:
            print("    ⚠️ FastAPI backend startup took longer than expected; continuing...")

        # 2. Start Streamlit UI Dashboard
        print(f"\n[2/2] Starting Streamlit UI Dashboard on http://localhost:{ui_port} ...")
        st_cmd = [
            STREAMLIT_EXE,
            "run",
            str(ROOT_DIR / "ui" / "app.py"),
            f"--server.port={ui_port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--theme.base=dark",
        ]
        ui_process = subprocess.Popen(
            st_cmd,
            cwd=str(ROOT_DIR),
            env=env,
        )
        processes.append(("Streamlit UI", ui_process))

        print("\n" + "=" * 70)
        print("  ✨ FINSIGHT IS RUNNING LIVE!")
        print("=" * 70)
        print(f"  📊 Web Dashboard:  http://localhost:{ui_port}")
        print(f"  ⚡ REST / SSE API:  http://127.0.0.1:{api_port}")
        print(f"  📖 API Docs:       http://127.0.0.1:{api_port}/docs")
        print("=" * 70)
        print("  Press Ctrl+C to terminate all services safely.\n")

        # Keep running and wait for child processes
        ui_process.wait()

    except KeyboardInterrupt:
        print("\n[*] Stopping FinSight services gracefully...")
    finally:
        for name, proc in processes:
            if proc.poll() is None:
                print(f"[*] Terminating {name} (PID: {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("[*] All services stopped cleanly. Goodbye!")


if __name__ == "__main__":
    main()
