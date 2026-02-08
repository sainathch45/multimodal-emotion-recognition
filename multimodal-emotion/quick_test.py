"""Quick API test"""
import requests

tests = [
    "I'm so incredibly happy and excited about this",
    "I'm feeling really down and hopeless",  
    "I'm absolutely furious about this"
]

for text in tests:
    response = requests.post(
        "http://localhost:8000/predict",
        data={"text": text, "auto_transcribe": False}
    )
    if response.ok:
        result = response.json()
        print(f"✓ {text[:30]}... → {result['emotion']} ({result['confidence']:.1%}) [{result.get('model_used', 'unknown')}]")
    else:
        print(f"✗ Error: {response.status_code}")
