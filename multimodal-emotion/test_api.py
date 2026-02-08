"""
Test the backend API with the fixed LLM fallback
"""

import requests
import time

# Wait for server to be ready
time.sleep(10)

API_URL = "http://localhost:8000"

test_cases = [
    ("I am really glad about this situation", "Happiness"),
    ("I'm so incredibly happy and excited about this", "Happiness"),
    ("I'm feeling really down and hopeless", "Sadness"),
    ("I'm absolutely furious about this", "Anger"),
]

print("\n" + "="*80)
print("TESTING BACKEND API")
print("="*80)

for i, (text, expected) in enumerate(test_cases, 1):
    print(f"\n{i}. Testing: \"{text}\"")
    print(f"   Expected: {expected}")
    
    try:
        response = requests.post(
            f"{API_URL}/predict_text",
            json={"text": text},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Predicted: {result['emotion']} ({result['confidence']:.1%})")
            print(f"   Model Used: {result.get('model_used', 'unknown')}")
            
            if result['emotion'] == expected:
                print(f"   🎉 CORRECT!")
            else:
                print(f"   ❌ WRONG!")
        else:
            print(f"   ❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")

print("\n" + "="*80)
