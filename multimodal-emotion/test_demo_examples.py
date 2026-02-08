"""
Quick test script to validate demo examples before presentation.
Run this to ensure your demo examples work correctly!
"""

import sys
sys.path.append('.')

from src.inference.engine import EmotionInferenceEngine

def test_demo_examples():
    print("=" * 60)
    print("DEMO EXAMPLE VALIDATION")
    print("=" * 60)
    print("Loading model...\n")
    
    engine = EmotionInferenceEngine(
        model_path='experiments/emotion_pretrained_sota/checkpoint_best.pt'
    )
    
    # Test cases with expected emotions
    test_cases = [
        # HAPPINESS examples
        ("This is absolutely amazing! I'm so excited and thrilled!", "Happiness"),
        ("Wow, what wonderful news! I couldn't be happier!", "Happiness"),
        ("Yay! This is the best day ever!", "Happiness"),
        ("I'm so grateful and blessed to have this opportunity!", "Happiness"),
        
        # SADNESS examples
        ("I'm feeling really down and hopeless about everything", "Sadness"),
        ("This is so depressing and makes me feel terrible", "Sadness"),
        ("I'm heartbroken and devastated by this news", "Sadness"),
        ("Everything feels so lonely and empty right now", "Sadness"),
        
        # ANGER examples
        ("This is absolutely infuriating and unacceptable!", "Anger"),
        ("I'm so angry I could scream! This is ridiculous!", "Anger"),
        ("How dare you! This makes my blood boil!", "Anger"),
        ("I'm furious about this situation, it's outrageous!", "Anger"),
        
        # PROBLEMATIC examples (test these to confirm they fail)
        ("I am glad to have worked with you", "Happiness"),
        ("Thank you for your professional assistance", "Happiness"),
        ("I appreciate your collaboration on this project", "Happiness"),
    ]
    
    working_examples = []
    failing_examples = []
    
    for text, expected in test_cases:
        try:
            result = engine.predict(text=text, audio=None)
            
            is_correct = result.emotion == expected
            status = "✅ PASS" if is_correct else "❌ FAIL"
            
            print(f"\n{status}")
            print(f"Input: {text[:60]}...")
            print(f"Expected: {expected.upper()}")
            print(f"Got: {result.emotion.upper()} (confidence: {result.confidence:.1%})")
            
            if result.confidence < 0.5:
                print(f"⚠️  LOW CONFIDENCE - May be unreliable")
            
            # Show top 3 predictions
            sorted_scores = sorted(result.all_probabilities.items(), 
                                  key=lambda x: x[1], reverse=True)[:3]
            print(f"Top 3: {', '.join([f'{e}:{s:.1%}' for e, s in sorted_scores])}")
            
            if is_correct and result.confidence > 0.5:
                working_examples.append(text)
            else:
                failing_examples.append((text, expected, result.emotion, result.confidence))
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            failing_examples.append((text, expected, "ERROR", 0.0))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Working examples: {len(working_examples)}/{len(test_cases)}")
    print(f"❌ Failing examples: {len(failing_examples)}/{len(test_cases)}")
    
    if working_examples:
        print("\n✅ USE THESE IN YOUR DEMO:")
        for i, example in enumerate(working_examples, 1):
            print(f"  {i}. {example[:70]}...")
    
    if failing_examples:
        print("\n❌ AVOID THESE IN YOUR DEMO:")
        for text, expected, got, conf in failing_examples:
            print(f"  - {text[:60]}...")
            print(f"    Expected {expected} but got {got} ({conf:.1%})")
    
    print("\n" + "=" * 60)
    print("💡 TIP: Only use examples that PASS with >50% confidence")
    print("=" * 60)

if __name__ == "__main__":
    test_demo_examples()
