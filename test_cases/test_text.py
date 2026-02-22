from labelling_agent import label_data_inline
import os

# Set API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCYhALjCno19sykM9DO2MON7ZpyiU4NgKA'

# Load test data
with open('reviews.txt', 'r') as f:
    data = [line.strip() for line in f.readlines() if line.strip()]

print(f"Testing with {len(data)} reviews...")
print("=" * 50)

# Test sentiment analysis
results = label_data_inline(
    data=data,
    labels=['positive', 'negative', 'neutral'],
    task_description='Classify customer reviews by sentiment'
)

print('=== SENTIMENT ANALYSIS RESULTS ===')
for r in results:
    print(f'[{r["label"]}] {r["original_text"]}')
    print(f'   Confidence: {r["confidence"]}')
    print(f'   Reasoning: {r["reasoning"]}')
    print()
