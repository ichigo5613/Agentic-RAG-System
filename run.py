# run.py in project root
import os
import sys
import subprocess

def main():
    """Run the Agentic RAG System"""
    print("=" * 60)
    print("🤖 Starting Agentic RAG System")
    print("=" * 60)
    
    # Start backend
    print("\n🚀 Starting backend on http://127.0.0.1:5000")
    backend_cmd = [sys.executable, "-m", "backend.app"]
    
    try:
        backend_process = subprocess.Popen(
            backend_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait a moment for backend to start
        import time
        time.sleep(3)
        
        print("✅ Backend started!")
        print("\n📋 To start frontend, open a new terminal and run:")
        print("   cd frontend")
        print("   streamlit run app.py")
        print("\n🌐 Frontend will run on: http://localhost:8501")
        print("\nPress Ctrl+C to stop the backend")
        print("-" * 60)
        
        # Keep backend running
        backend_process.wait()
        
    except KeyboardInterrupt:
        print("\n\n👋 Stopping system...")
        backend_process.terminate()
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()