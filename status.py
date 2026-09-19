#!/usr/bin/env python3
"""
STYLE-SYNTH Status Checker
Run anytime to verify active services
"""
import sys
import os

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import requests
import time

def check_service(name, url, timeout=4):
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        response_time = time.time() - start_time

        if response.status_code == 200:
            print(f"[+] {name}: UP ({response_time:.2f}s)")
            return True
        else:
            print(f"[-] {name}: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[-] {name}: DOWN ({str(e)})")
        return False

def main():
    print("STYLE-SYNTH Status Check")
    print("=" * 35)

    services = [
        ("API Server", "http://127.0.0.1:8002/health"),
        ("Web Interface", "http://127.0.0.1:8501/healthz"),
    ]

    all_up = True
    for name, url in services:
        if not check_service(name, url):
            all_up = False

    print()
    if all_up:
        print("All services are running! STYLE-SYNTH is ready.")
        print("\nWeb Interface: http://127.0.0.1:8501")
        print("API Endpoint:  http://127.0.0.1:8002")
        return 0
    else:
        print("Some services are offline. Run 'python start.py' to launch.")
        return 1

if __name__ == "__main__":
    sys.exit(main())