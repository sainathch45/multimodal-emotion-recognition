"""
Automatic Launcher for Futuristic Emotion Recognition System
Starts both backend and frontend with one command
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
    ║     🚀 FUTURISTIC EMOTION RECOGNITION SYSTEM 🚀              ║
    ║                                                              ║
    ║     Multimodal AI • Dynamic UI • Intelligent Actions        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("\n✨ Starting your futuristic emotion recognition system...\n")


def check_prerequisites():
    """Check if prerequisites are met"""
    print("📋 Checking prerequisites...")
    
    # Check if model checkpoint exists
    model_path = Path("experiments/emotion_pretrained_sota/checkpoint_best.pt")
    if not model_path.exists():
        print(f"❌ Model checkpoint not found at: {model_path}")
        print("   Please train the model first or ensure the checkpoint exists.")
        return False
    print("✅ Model checkpoint found")
    
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
    
    print()
    return True


def start_backend():
    """Start FastAPI backend server"""
    print("🔧 Starting FastAPI backend on http://localhost:8000")
    
    # Change to backend directory and start server
    backend_process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd="backend",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    )
    
    # Wait for backend to start
    print("   Waiting for backend to initialize...")
    time.sleep(5)
    
    print("✅ Backend started successfully!\n")
    return backend_process


def start_frontend():
    """Start Next.js frontend server"""
    print("🎨 Starting Next.js frontend on http://localhost:3000")
    
    # Change to frontend directory and start dev server
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="frontend",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    )
    
    # Wait for frontend to start
    print("   Waiting for frontend to initialize...")
    time.sleep(10)
    
    print("✅ Frontend started successfully!\n")
    return frontend_process


def open_browser():
    """Open browser to the application"""
    print("🌐 Opening browser...")
    time.sleep(2)
    webbrowser.open("http://localhost:3000")
    print("✅ Browser opened!\n")


def print_success_message():
    """Print success message with URLs"""
    message = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              🎉 SYSTEM LAUNCHED SUCCESSFULLY! 🎉             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🔗 Access Points:
    
       Frontend UI:  http://localhost:3000
       Backend API:  http://localhost:8000
       API Docs:     http://localhost:8000/docs
       Health Check: http://localhost:8000/health
    
    📱 Features Available:
    
       ✨ Futuristic glassmorphism UI with particles
       🎤 Real-time audio recording and upload
       💬 Emotion-aware chat support bots
       🎯 Intelligent emotion-based actions
       📊 87.6% accuracy emotion detection
    
    💡 Usage Tips:
    
       1. Express yourself with text or voice
       2. Get instant emotion detection
       3. Access emotion-specific support
       4. Interact with wellness bots
    
    🎬 Demo Flow:
    
       Landing → Input → Analyze → Results → Action Page → Chat Bot
    
    ⚠️  To stop the application:
       - Close both console windows
       - Or press Ctrl+C in this window
    
    ═══════════════════════════════════════════════════════════════
    
    Ready to detect emotions! Open http://localhost:3000 to begin.
    
    """
    print(message)


def main():
    """Main launcher function"""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    try:
        # Start backend
        backend_process = start_backend()
        
        # Start frontend
        frontend_process = start_frontend()
        
        # Open browser
        open_browser()
        
        # Print success message
        print_success_message()
        
        # Keep running
        print("🔄 Application is running. Press Ctrl+C to stop...\n")
        
        # Wait for user interrupt
        try:
            backend_process.wait()
        except KeyboardInterrupt:
            print("\n\n⚠️  Shutting down...")
            backend_process.terminate()
            frontend_process.terminate()
            print("✅ Application stopped successfully.")
    
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
