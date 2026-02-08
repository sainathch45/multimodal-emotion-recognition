"""
Quick Toggle Script - Switch between Model and LLM Fallback
Use this during demo if needed!
"""

import requests
import sys

def toggle_mode():
    """Toggle the backend mode"""
    try:
        print("🔄 Toggling backend mode...")
        response = requests.post("http://localhost:8000/toggle-mode", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"   New mode: {data['mode']}")
            print(f"   Message: {data['message']}")
            return True
        else:
            print(f"\n❌ Failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Backend not running!")
        print("   Start backend first: python backend/main.py")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

def check_status():
    """Check current backend status"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("\n📊 Current Status:")
            print(f"   Mode: {data.get('mode', 'Unknown')}")
            print(f"   Status: {data.get('status', 'Unknown')}")
            print(f"   Device: {data.get('device', 'Unknown')}")
            return True
        return False
    except:
        print("\n❌ Backend not responding")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🎛️  BACKEND MODE TOGGLE")
    print("=" * 60)
    
    # Check current status
    if not check_status():
        sys.exit(1)
    
    # Ask user
    print("\n" + "=" * 60)
    choice = input("\nToggle mode? (y/n): ").strip().lower()
    
    if choice == 'y':
        toggle_mode()
        print("\n" + "=" * 60)
        print("Checking new status...")
        check_status()
    else:
        print("\nNo changes made.")
    
    print("\n" + "=" * 60)
    print("💡 TIP: You can run this script anytime during demo!")
    print("=" * 60)
