#!/usr/bin/env python3
"""
STYLE-SYNTH Status Checker
Run anytime to verify the system is working
"""
import requests
import sys
import time

def check_service(name, url, timeout=5):
    """Check if a service is responding"""
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        response_time = time.time() - start_time

        if response.status_code == 200:
            print(f"✅ {name}: UP ({response_time:.2f}s)")
            return True
        else:
            print(f"❌ {name}: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {name}: DOWN ({str(e)})")
        return False

def main():
    """Check all STYLE-SYNTH services"""
    print("🎭 STYLE-SYNTH Status Check")
    print("=" * 30)

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
        print("🎉 All services are running! STYLE-SYNTH is ready.")
        print("\n🌐 Web Interface: http://127.0.0.1:8501")
        print("🔌 API Endpoint: http://127.0.0.1:8002")
        print("\n📤 Upload face photos and try different styles!")
        return 0
    else:
        print("⚠️ Some services are down.")
        print("\n💡 To start STYLE-SYNTH, run:")
        print("   python start.py")
        print("   # or on Windows: start.bat")
        return 1

if __name__ == "__main__":
    sys.exit(main())