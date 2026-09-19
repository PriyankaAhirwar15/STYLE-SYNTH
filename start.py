#!/usr/bin/env python3
"""
STYLE-SYNTH Launcher
Reliable startup script that ensures the system always works
"""
import os
import sys
import subprocess
import time
import signal
import atexit
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'numpy',
        'groq', 'requests', 'torch'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"[*] Some packages might be missing: {', '.join(missing)}")
        print("   Continuing anyway - they may be installed...")
        return True

    print("[+] Core dependencies verified")
    return True

def start_api_server(port=8002):
    """Start the FastAPI server"""
    print(f"[*] Starting FastAPI backend on port {port}...")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--reload",
        "--log-level", "info"
    ]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"[-] Failed to start API server: {e}")
        return None

def start_streamlit_app(port=8501, api_port=8002):
    """Start the Streamlit frontend"""
    print(f"[*] Starting Streamlit Studio on port {port}...")

    env = os.environ.copy()
    env["STREAMLIT_API_PORT"] = str(api_port)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "frontend/app.py",
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"[-] Failed to start Streamlit: {e}")
        return None

def wait_for_server(url, timeout=30):
    """Wait for server to be ready"""
    import requests
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)

    return False

def cleanup_processes(processes):
    """Clean up running processes"""
    for name, process in processes.items():
        if process and process.poll() is None:
            print(f"[*] Stopping {name}...")
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                process.kill()

def main():
    print("STYLE-SYNTH 2.0 - Studio Launcher")
    print("=" * 45)

    if not check_dependencies():
        return 1

    API_PORT = 8002
    STREAMLIT_PORT = 8501

    processes = {}
    running = True

    def signal_handler(signum, frame):
        nonlocal running
        print("\n[*] Shutdown requested...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(lambda: cleanup_processes(processes))

    try:
        # Start API server
        api_process = start_api_server(API_PORT)
        if not api_process:
            return 1
        processes["API Server"] = api_process

        # Wait for API to be ready
        api_url = f"http://127.0.0.1:{API_PORT}/health"
        print(f"[*] Waiting for API server at {api_url}...")
        if not wait_for_server(api_url, 20):
            print("[-] API server failed to respond, but continuing...")
        else:
            print("[+] API server ready!")

        # Start Streamlit
        streamlit_process = start_streamlit_app(STREAMLIT_PORT, API_PORT)
        if not streamlit_process:
            return 1
        processes["Streamlit"] = streamlit_process

        # Wait for Streamlit to be ready
        streamlit_url = f"http://127.0.0.1:{STREAMLIT_PORT}/healthz"
        print(f"[*] Waiting for Streamlit at http://127.0.0.1:{STREAMLIT_PORT}...")
        if not wait_for_server(streamlit_url, 15):
            print("[*] Streamlit is launching...")

        print("\nSTYLE-SYNTH 2.0 IS RUNNING!")
        print(f"-> Web Interface: http://127.0.0.1:{STREAMLIT_PORT}")
        print(f"-> API Endpoint:  http://127.0.0.1:{API_PORT}")
        print("\n8 GAN Styles: anime, cartoon_3d, cyberpunk, oil_painting, sketch, watercolor, pop_art, gothic_ink")
        print("Press Ctrl+C to stop.\n")

        while running:
            for name, process in list(processes.items()):
                if process.poll() is not None:
                    print(f"[-] {name} stopped unexpectedly (exit code: {process.returncode})")
                    running = False
                    break

            if running:
                time.sleep(2)

    except KeyboardInterrupt:
        print("\n[*] Shutdown requested by user")
    except Exception as e:
        print(f"[-] Unexpected error: {e}")
        return 1
    finally:
        cleanup_processes(processes)
        print("[+] Cleanup complete")

    return 0

if __name__ == "__main__":
    sys.exit(main())