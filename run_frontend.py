# run_frontend.py
import os
import sys
import subprocess
import time
import webbrowser
import threading

def check_backend():
    """Check if backend is running - IMPROVED VERSION"""
    try:
        import requests
        response = requests.get("http://127.0.0.1:5000/health", timeout=3)
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
    except requests.exceptions.ConnectionError:
        print("❌ Backend connection failed")
    except Exception as e:
        print(f"❌ Backend check error: {e}")
    return False

def start_backend():
    """Start backend server - SIMPLIFIED VERSION"""
    print("🚀 Starting backend server...")
    
    try:
        # Run backend directly
        backend_proc = subprocess.Popen(
            [sys.executable, "-m", "backend.app"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print("Backend process started. Waiting for it to be ready...")
        
        # Wait for backend to start with better checking
        max_wait = 60  # Increased to 60 seconds
        for i in range(max_wait):
            time.sleep(1)
            
            # Check if process is still alive
            if backend_proc.poll() is not None:
                # Process died
                stdout, stderr = backend_proc.communicate()
                print(f"Backend process died. Exit code: {backend_proc.returncode}")
                if stderr:
                    print(f"Stderr: {stderr[:500]}")
                return None
            
            # Check if backend is responding
            if check_backend():
                print(f"✅ Backend started successfully after {i+1} seconds")
                return backend_proc
            
            if (i + 1) % 5 == 0:
                print(f"  Still waiting... ({i+1}/{max_wait})")
        
        print("❌ Backend failed to start within timeout")
        backend_proc.terminate()
        return None
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def main():
    """Main entry point"""
    print("=" * 60)
    print("🤖 Agentic RAG System - Frontend Launcher")
    print("=" * 60)
    
    # First, try direct check
    if check_backend():
        print("✅ Backend already running")
    else:
        print("\nBackend is not running. Do you want to:")
        print("1. Start backend automatically")
        print("2. Start backend manually and continue")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            backend_proc = start_backend()
            if not backend_proc:
                print("\nFailed to start backend. You can:")
                print("1. Start backend manually in another terminal:")
                print("   cd backend")
                print("   python app.py")
                print("2. Then run this script again and choose option 2")
                sys.exit(1)
        elif choice == "2":
            input("Start backend manually (open another terminal and run 'python run_backend.py'), then press Enter to continue...")
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
            time.sleep(5)  # Increased delay
            webbrowser.open("http://localhost:8501")
            print("✅ Browser opened to http://localhost:8501")
        
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Run Streamlit with explicit config
        streamlit_cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "frontend/app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.serverAddress", "localhost",
            "--theme.base", "light"
        ]
        
        print(f"Running: {' '.join(streamlit_cmd)}")
        subprocess.run(streamlit_cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 Frontend stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start frontend: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()