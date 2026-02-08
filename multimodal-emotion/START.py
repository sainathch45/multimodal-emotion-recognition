"""
🚀 ONE-CLICK STARTUP SCRIPT
Starts both backend and frontend servers automatically
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import os

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🎭 EMOTION RECOGNITION AI - STARTUP SCRIPT 🎭           ║
║                                                              ║
║     Starting Backend + Frontend Servers...                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

def main():
    print_banner()
    
    # Get project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if venv exists
    venv_python = project_root / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print("❌ Virtual environment not found!")
        print("   Run: python -m venv venv")
        print("   Then: .\\venv\\Scripts\\activate")
        print("   Then: pip install -r backend/requirements.txt")
        sys.exit(1)
    
    # Check if node_modules exists
    frontend_path = project_root / "frontend"
    node_modules = frontend_path / "node_modules"
    if not node_modules.exists():
        print("❌ Frontend dependencies not installed!")
        print("   Run: cd frontend && npm install")
        sys.exit(1)
    
    try:
        # Start Backend
        print("\n🔧 Starting Backend Server (Port 8000)...")
        backend_cmd = [str(venv_python), "backend/main.py"]
        backend_process = subprocess.Popen(
            backend_cmd,
            cwd=project_root,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        print("✅ Backend server started!")
        
        # Wait for backend to initialize
        time.sleep(3)
        
        # Start Frontend
        print("\n🎨 Starting Frontend Server (Port 3000)...")
        frontend_cmd = ["npm", "run", "dev"]
        frontend_process = subprocess.Popen(
            frontend_cmd,
            cwd=frontend_path,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            shell=True
        )
        print("✅ Frontend server started!")
        
        # Wait for frontend to start
        time.sleep(5)
        
        # Open browser
        print("\n🌐 Opening browser...")
        webbrowser.open("http://localhost:3000")
        
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ✅ ALL SYSTEMS RUNNING!                                 ║
║                                                              ║
║     📍 Frontend: http://localhost:3000                      ║
║     📍 Backend:  http://localhost:8000                      ║
║                                                              ║
║     Press Ctrl+C in EITHER console to stop both servers    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Keep script running
        print("\n⏳ Servers are running. Close the console windows to stop.")
        input("\nPress Enter to stop all servers and exit...\n")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Cleanup
        try:
            backend_process.terminate()
            frontend_process.terminate()
        except:
            pass
        print("✅ Servers stopped. Goodbye!")

if __name__ == "__main__":
    main()
