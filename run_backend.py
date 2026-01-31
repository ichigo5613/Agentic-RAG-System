# run_backend.py
import os
import sys
import subprocess
import time
import webbrowser

def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
    except:
        print("❌ Ollama is not running")
        return False

def start_ollama():
    """Start Ollama service"""
    print("🚀 Starting Ollama...")
    
    # Try to start Ollama (platform dependent)
    import platform
    system = platform.system()
    
    try:
        if system == "Windows":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif system == "Darwin":  # macOS
            subprocess.Popen(["open", "-a", "Ollama"])
        else:  # Linux
            subprocess.Popen(["ollama", "serve"])
        
        # Wait for Ollama to start
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            if check_ollama():
                return True
            print(f"  Waiting for Ollama... ({i+1}/30)")
        
        print("❌ Ollama failed to start")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start Ollama: {e}")
        return False

def check_requirements():
    """Check if all requirements are installed - FIXED VERSION"""
    print("🔍 Checking requirements...")
    
    # FIXED: Updated import names to match actual package names
    required_packages = [
        ("flask", "flask"),
        ("langchain", "langchain"),
        ("langchain_community", "langchain-community"),
        ("langchain_chroma", "langchain-chroma"),
        ("sentence_transformers", "sentence-transformers"),
        ("pypdf", "pypdf"),
        ("dotenv", "python-dotenv"),
        ("requests", "requests"),
        ("chromadb", "chromadb")
    ]
    
    missing = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            missing.append(package_name)
            print(f"  ❌ {package_name}")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All requirements satisfied")
    return True

def main():
    """Main entry point"""
    print("=" * 60)
    print("🤖 Agentic RAG System - Startup Script")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check Ollama
    if not check_ollama():
        print("\nOllama is required. Do you want to:")
        print("1. Try to start Ollama automatically")
        print("2. Start Ollama manually and continue")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            if not start_ollama():
                sys.exit(1)
        elif choice == "2":
            input("Start Ollama manually, then press Enter to continue...")
            if not check_ollama():
                print("❌ Ollama still not running")
                sys.exit(1)
        else:
            sys.exit(0)
    
    # Check if models are available
    print("\n🔍 Checking Ollama models...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        models = response.json().get("models", [])
        
        # Check for phi3:mini
        has_phi3 = any("phi3:mini" in model.get("name", "") for model in models)
        
        if not models:
            print("❌ No models found in Ollama")
            print("\nPull a model with: ollama pull phi3:mini")
            sys.exit(1)
        elif not has_phi3:
            print("⚠️  phi3:mini not found. Available models:")
            for model in models:
                print(f"  - {model.get('name')}")
            print("\nYou can use available models by changing OLLAMA_MODEL in .env")
        else:
            print("✅ phi3:mini model found")
    except:
        print("⚠️  Could not check Ollama models")
    
    # Start backend
    print("\n🚀 Starting Agentic RAG System backend...")
    print("📁 Backend will run on: http://localhost:5000")
    print("📊 Health check: http://localhost:5000/health")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Add the backend directory to Python path
        backend_path = os.path.join(os.path.dirname(__file__), "backend")
        sys.path.insert(0, backend_path)
        
        # Import and run the Flask app
        from app import app
        
        # Open browser after delay
        def open_browser():
            time.sleep(2)
            webbrowser.open("http://localhost:5000")
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Run the app
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()