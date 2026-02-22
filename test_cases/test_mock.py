"""
Mock test without API calls - demonstrates the pipeline structure
"""

# Load test data
with open('reviews.txt', 'r') as f:
    data = [line.strip() for line in f.readlines() if line.strip()]

print(f"Testing with {len(data)} reviews...")
print("=" * 50)

# Mock results (simulating what the API would return)
mock_results = []
for i, text in enumerate(data):
    if "amazing" in text.lower() or "excellent" in text.lower() or "five stars" in text.lower():
        label = "positive"
        confidence = 0.95
        reasoning = "Strong positive language indicators"
    elif "waste" in text.lower() or "terrible" in text.lower() or "disappointing" in text.lower():
        label = "negative" 
        confidence = 0.92
        reasoning = "Clear negative sentiment expressed"
    elif "meh" in text.lower() or "average" in text.lower() or "okay" in text.lower():
        label = "neutral"
        confidence = 0.85
        reasoning = "Moderate or mixed sentiment"
    else:
        label = "neutral"
        confidence = 0.75
        reasoning = "No strong sentiment indicators"
    
    mock_results.append({
        "original_text": text,
        "label": label,
        "confidence": confidence,
        "reasoning": reasoning
    })

print('=== MOCK SENTIMENT ANALYSIS RESULTS ===')
for r in mock_results:
    print(f'[{r["label"]}] {r["original_text"]}')
    print(f'   Confidence: {r["confidence"]}')
    print(f'   Reasoning: {r["reasoning"]}')
    print()

# Summary
from collections import Counter
label_counts = Counter(r["label"] for r in mock_results)
print('=== SUMMARY ===')
for label, count in label_counts.most_common():
    print(f'{label}: {count} reviews')
