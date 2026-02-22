"""
Test JSON pipeline with headlines data
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

# Set API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCzA8Uf_xfghL2ypS6EhuCHIy1eI2zzdt8'

# Load JSON data
with open('headlines.json', 'r') as f:
    data = json.load(f)

print(f"Testing with {len(data)} headlines...")
print("=" * 50)

# Extract headlines for classification
headlines = [item['headline'] for item in data]

# Simple LangChain setup
llm = ChatGoogleGenerativeAI(model="gemma-3-4b-it", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Classify each news headline into one of these categories: finance, science, sports, politics, technology, health, environment, business, entertainment.

Output Format: JSON array with elements containing:
- "original_text": the headline
- "label": assigned category
- "confidence": a float between 0.0 and 1.0
- "reasoning": a brief explanation

Respond ONLY with valid JSON. No markdown code fences, no extra text.

Headlines to classify:
{data_items}
""")

parser = JsonOutputParser()
chain = prompt | llm | parser

# Test with first 5 headlines
test_data = headlines[:5]
numbered = "\n".join(f"{i+1}. {item}" for i, item in enumerate(test_data))

try:
    result = chain.invoke({"data_items": numbered})
    print('=== HEADLINE CLASSIFICATION RESULTS ===')
    for r in result:
        print(f'[{r["label"]}] {r["original_text"]}')
        print(f'   Confidence: {r["confidence"]}')
        print(f'   Reasoning: {r["reasoning"]}')
        print()
        
    # Summary
    from collections import Counter
    label_counts = Counter(r["label"] for r in result)
    print('=== SUMMARY ===')
    for label, count in label_counts.most_common():
        print(f'{label}: {count} headlines')
        
except Exception as e:
    print(f"Error: {e}")
    print("Make sure GOOGLE_API_KEY is set correctly")
