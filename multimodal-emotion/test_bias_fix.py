"""
Quick test to verify bias detection and LLM fallback are working
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.inference.intelligent_hybrid import IntelligentHybridEngine

# Initialize engine
print("Initializing Intelligent Hybrid Engine...")
engine = IntelligentHybridEngine(
    model_path="experiments/emotion_pretrained_sota/checkpoint_best.pt",
    use_neural=True
)

# Test cases that were failing
test_cases = [
    "I am really glad about this situation",
    "I'm so incredibly happy and excited about this",
    "I'm feeling really down and hopeless",
    "I'm absolutely furious about this",
]

print("\n" + "="*80)
print("TESTING EMOTION DETECTION")
print("="*80)

for i, text in enumerate(test_cases, 1):
    print(f"\n{i}. Text: \"{text}\"")
    result = engine.predict(text=text, audio=None)
    print(f"   Emotion: {result.emotion}")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Model Used: {result.model_used}")
    print(f"   Reason: {result.reason}")
    print(f"   Bias Detected: {engine.bias_detected}")
    
print("\n" + "="*80)
print("Bias detection status:", "ACTIVE" if engine.bias_detected else "NORMAL")
print("Recent predictions:", list(engine.recent_predictions))
print("="*80)
