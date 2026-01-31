# run_frontend.py
import os
import sys
import subprocess
import time
import webbrowser

def check_backend():
    """Check if backend is running"""
    try:
        import requests
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
    except:
        print("❌ Backend is not running")
        return False

def start_backend():
    """Start backend server"""
    print("🚀 Starting backend server...")
    
    # Run backend in a separate process
    backend_proc = subprocess.Popen(
        [sys.executable, "run_backend.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for backend to start
    for i in range(30):  # Wait up to 30 seconds
        time.sleep(1)
        if check_backend():
            return backend_proc
        print(f"  Waiting for backend... ({i+1}/30)")
    
    print("❌ Backend failed to start")
    backend_proc.terminate()
    return None

def main():
    """Main entry point"""
    print("=" * 60)
    print("🤖 Agentic RAG System - Frontend Launcher")
    print("=" * 60)
    
    # Check if backend is running
    if not check_backend():
        print("\nBackend is not running. Do you want to:")
        print("1. Start backend automatically")
        print("2. Start backend manually and continue")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            backend_proc = start_backend()
            if not backend_proc:
                sys.exit(1)
        elif choice == "2":
            input("Start backend manually (run_backend.py), then press Enter to continue...")
            if not check_backend():
                print("❌ Backend still not running")
                sys.exit(1)
        else:
            sys.exit(0)
    
    # Start Streamlit frontend
    print("\n🚀 Starting Streamlit frontend...")
    print("🌐 Frontend will run on: http://localhost:8501")
    print("\nPress Ctrl+C to stop the frontend")
    print("-" * 60)
    
    try:
        # Open browser after delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Run Streamlit
        streamlit_cmd = [
            "streamlit", "run", 
            "frontend/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--theme.base", "light",
            "--browser.serverAddress", "localhost"
        ]
        
        subprocess.run(streamlit_cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 Frontend stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()