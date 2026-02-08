"""
Automatic Launcher for Multimodal Emotion Recognition System
Starts both backend (LLM fallback mode) and frontend with one command
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import os

def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🎭 MULTIMODAL EMOTION RECOGNITION SYSTEM 🎭           ║
    ║                                                              ║
    ║     Attention-Based Fusion • Real-Time • Production Ready   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("\n✨ Starting emotion recognition system...\n")


def check_prerequisites():
    """Check if prerequisites are met"""
    print("📋 Checking prerequisites...")
    
    # Check if virtual environment exists
    venv_path = Path("venv/Scripts/python.exe")
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run:")
        print("   python -m venv venv")
        print("   .\\venv\\Scripts\\Activate.ps1")
        print("   pip install -r requirements.txt")
        return False
    print("✅ Virtual environment found")
    
    # Check if frontend exists
    frontend_path = Path("frontend/package.json")
    if not frontend_path.exists():
        print("❌ Frontend not found. Please ensure frontend directory exists.")
        return False
    print("✅ Frontend directory found")
    
    # Check if backend exists
    backend_path = Path("backend/main.py")
    if not backend_path.exists():
        print("❌ Backend not found. Please ensure backend directory exists.")
        return False
    print("✅ Backend directory found")
    
    # Check if node_modules exists
    node_modules = Path("frontend/node_modules")
    if not node_modules.exists():
        print("⚠️  Frontend dependencies not installed. Run 'npm install' in frontend folder.")
        print("   Continuing anyway...")
    else:
        print("✅ Frontend dependencies found")
    
    print()
    return True


def start_backend(use_fallback=True):
    """Start FastAPI backend server"""
    mode = "LLM Fallback (Reliable)" if use_fallback else "Neural Model"
    print(f"🔧 Starting backend in {mode} mode...")
    print("   Backend URL: http://localhost:8000")
    
    # Set environment variable for LLM fallback
    env = os.environ.copy()
    if use_fallback:
        env["USE_LLM_FALLBACK"] = "true"
    
    # Start backend using virtual environment Python
    venv_python = Path("venv/Scripts/python.exe")
    backend_process = subprocess.Popen(
        [str(venv_python), "backend/main.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    print("   ⏳ Waiting for backend to start...")
    time.sleep(3)  # Give backend time to start
    
    if backend_process.poll() is None:
        print("   ✅ Backend started successfully!\n")
        return backend_process
    else:
        print("   ❌ Backend failed to start")
        print("   Check the terminal output for errors")
        return None


def start_frontend():
    """Start Next.js frontend server"""
    print("🎨 Starting frontend...")
    print("   Frontend URL: http://localhost:3000")
    
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="frontend",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        shell=True
    )
    
    print("   ⏳ Waiting for frontend to start...")
    time.sleep(5)  # Give frontend more time to compile
    
    if frontend_process.poll() is None:
        print("   ✅ Frontend started successfully!\n")
        return frontend_process
    else:
        print("   ❌ Frontend failed to start")
        print("   Check if 'npm install' was run in frontend folder")
        return None


def open_browser():
    """Open browser after a delay"""
    print("🌐 Opening browser in 3 seconds...")
    time.sleep(3)
    
    try:
        webbrowser.open("http://localhost:3000")
        print("   ✅ Browser opened\n")
    except Exception as e:
        print(f"   ⚠️  Could not open browser automatically: {e}")
        print("   Please manually open: http://localhost:3000\n")


def print_instructions():
    """Print usage instructions"""
    instructions = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                     🎉 SYSTEM READY! 🎉                      ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📍 Access Points:
       • Frontend UI:  http://localhost:3000
       • Backend API:  http://localhost:8000
       • API Docs:     http://localhost:8000/docs
    
    🎮 How to Use:
       1. Type text or record audio
       2. Click "Analyze Emotion"
       3. View results and attention heatmap
       4. Click "Continue to Action Page" for wellness support
    
    🔄 Mode Switching:
       Run in another terminal: python toggle_mode.py
    
    🛑 To Stop:
       Press Ctrl+C (may need to press twice)
    
    📚 Documentation:
       • QUICK_START.md - All commands
       • DEMO_PRESENTATION_SCRIPT.md - Demo guide
       • CODE_ARCHITECTURE_EXPLAINED.md - Technical docs
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    print(instructions)


def main():
    """Main launcher function"""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix the issues above.\n")
        sys.exit(1)
    
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend (with LLM fallback by default for reliability)
        backend_process = start_backend(use_fallback=True)
        if not backend_process:
            print("\n❌ Failed to start backend. Exiting.\n")
            sys.exit(1)
        
        # Start frontend
        frontend_process = start_frontend()
        if not frontend_process:
            print("\n❌ Failed to start frontend. Stopping backend.\n")
            if backend_process:
                backend_process.terminate()
            sys.exit(1)
        
        # Open browser
        open_browser()
        
        # Print instructions
        print_instructions()
        
        # Keep processes running
        print("✨ System is running. Press Ctrl+C to stop both servers.\n")
        
        # Monitor processes
        while True:
            time.sleep(1)
            
            # Check if backend crashed
            if backend_process.poll() is not None:
                print("\n❌ Backend process stopped unexpectedly!")
                break
            
            # Check if frontend crashed
            if frontend_process.poll() is not None:
                print("\n❌ Frontend process stopped unexpectedly!")
                break
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...\n")
    
    finally:
        # Cleanup: terminate both processes
        if backend_process and backend_process.poll() is None:
            print("   Stopping backend...")
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
                print("   ✅ Backend stopped")
            except subprocess.TimeoutExpired:
                backend_process.kill()
                print("   ⚠️  Backend force-killed")
        
        if frontend_process and frontend_process.poll() is None:
            print("   Stopping frontend...")
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
                print("   ✅ Frontend stopped")
            except subprocess.TimeoutExpired:
                frontend_process.kill()
                print("   ⚠️  Frontend force-killed")
        
        print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    main()
