"""
Test CSV pipeline with reviews_unlabelled.csv data
"""
import os
import csv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

# Set API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCzA8Uf_xfghL2ypS6EhuCHIy1eI2zzdt8'

# Load CSV data
data = []
with open('reviews_unlabelled.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print(f"Testing with {len(data)} CSV reviews...")
print("=" * 50)

# Extract review texts for classification
reviews = [item['review_text'] for item in data]

# Simple LangChain setup
llm = ChatGoogleGenerativeAI(model="gemma-3-4b-it", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Classify each product review as positive, negative, or neutral.

Output Format: JSON array with elements containing:
- "original_text": the review text
- "label": assigned sentiment (positive/negative/neutral)
- "confidence": a float between 0.0 and 1.0
- "reasoning": a brief explanation

Respond ONLY with valid JSON. No markdown code fences, no extra text.

Reviews to classify:
{data_items}
""")

parser = JsonOutputParser()
chain = prompt | llm | parser

# Test with all reviews
test_data = reviews
numbered = "\n".join(f"{i+1}. {item}" for i, item in enumerate(test_data))

try:
    result = chain.invoke({"data_items": numbered})
    print('=== CSV SENTIMENT ANALYSIS RESULTS ===')
    for r in result:
        print(f'[{r["label"]}] {r["original_text"][:80]}...')
        print(f'   Confidence: {r["confidence"]}')
        print(f'   Reasoning: {r["reasoning"]}')
        print()
        
    # Summary
    from collections import Counter
    label_counts = Counter(r["label"] for r in result)
    print('=== SUMMARY ===')
    for label, count in label_counts.most_common():
        print(f'{label}: {count} reviews')
        
    # Save results to new CSV
    output_data = []
    for i, r in enumerate(result):
        # Find original review_id from data
        original_review = next((item for item in data if item['review_text'] == r['original_text']), None)
        review_id = original_review['review_id'] if original_review else i + 1
        
        output_data.append({
            'review_id': review_id,
            'original_text': r['original_text'],
            'sentiment_label': r['label'],
            'confidence': r['confidence'],
            'reasoning': r['reasoning']
        })
    
    with open('reviews_labelled.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['review_id', 'original_text', 'sentiment_label', 'confidence', 'reasoning']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)
    
    print(f'\n✅ Saved {len(output_data)} labelled reviews to reviews_labelled.csv')
        
except Exception as e:
    print(f"Error: {e}")
    print("Make sure GOOGLE_API_KEY is set correctly")
