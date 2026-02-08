"""
Quick Setup Script for Futuristic Emotion Recognition System
Installs all dependencies in one go
"""

import subprocess
import sys
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def run_command(cmd, cwd=None, shell=True):
    """Run command and handle errors"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=shell,
            check=True,
            capture_output=True,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    """Main setup function"""
    print_header("🚀 Futuristic Emotion Recognition - Quick Setup")
    
    # 1. Install backend dependencies
    print_header("📦 Installing Backend Dependencies")
    print("Installing: fastapi, uvicorn, python-multipart, pydantic...")
    
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", 
         "fastapi", "uvicorn[standard]", "python-multipart", "pydantic"],
        shell=False
    )
    
    if success:
        print("✅ Backend dependencies installed successfully!")
    else:
        print(f"❌ Backend installation failed: {output}")
        return False
    
    # 2. Check if frontend exists
    frontend_path = Path("frontend")
    if not frontend_path.exists():
        print("❌ Frontend directory not found!")
        return False
    
    # 3. Install frontend dependencies
    print_header("🎨 Installing Frontend Dependencies")
    print("Installing: Next.js, React, Tailwind CSS, Framer Motion...")
    print("This may take 2-3 minutes...\n")
    
    success, output = run_command(
        "npm install",
        cwd="frontend"
    )
    
    if success:
        print("\n✅ Frontend dependencies installed successfully!")
    else:
        print(f"\n❌ Frontend installation failed: {output}")
        print("\nTry manually:")
        print("  cd frontend")
        print("  npm install")
        return False
    
    # 4. Success message
    print_header("🎉 Setup Complete!")
    print("""
All dependencies installed successfully!

🚀 To launch the application, run:

    python launch_futuristic_app.py

Or manually:

    Terminal 1: cd backend && python main.py
    Terminal 2: cd frontend && npm run dev

Then open: http://localhost:3000

📖 Documentation:
    - FUTURISTIC_APP_GUIDE.md (setup & usage)
    - PRESENTATION_GUIDE.md (demo script)
    - TRANSFORMATION_COMPLETE.md (what's new)

Ready to detect emotions! ✨
    """)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n⚠️  Setup encountered errors. Please check the output above.")
            input("Press Enter to exit...")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
