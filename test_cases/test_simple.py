"""
Simple test without ZyndAI dependencies
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

# Set API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCzA8Uf_xfghL2ypS6EhuCHIy1eI2zzdt8'

# Load test data
with open('reviews.txt', 'r') as f:
    data = [line.strip() for line in f.readlines() if line.strip()]

print(f"Testing with {len(data)} reviews...")
print("=" * 50)

# Simple LangChain setup
llm = ChatGoogleGenerativeAI(model="gemma-3-4b-it", temperature=0)

system_prompt = """Classify each review as positive, negative, or neutral.

Output Format: JSON array with elements containing:
- "original_text": the input text
- "label": the assigned label (positive/negative/neutral)
- "confidence": a float between 0.0 and 1.0
- "reasoning": a brief explanation

Respond ONLY with valid JSON. No markdown code fences, no extra text."""

prompt = ChatPromptTemplate.from_template("""
Classify each review as positive, negative, or neutral.

Output Format: JSON array with elements containing:
- "original_text": the input text
- "label": the assigned label (positive/negative/neutral)
- "confidence": a float between 0.0 and 1.0
- "reasoning": a brief explanation

Respond ONLY with valid JSON. No markdown code fences, no extra text.

Reviews to classify:
{data_items}
""")

parser = JsonOutputParser()
chain = prompt | llm | parser

# Test with all reviews
test_data = data
numbered = "\n".join(f"{i+1}. {item}" for i, item in enumerate(test_data))

try:
    result = chain.invoke({"data_items": numbered})
    print('=== SENTIMENT ANALYSIS RESULTS ===')
    for r in result:
        print(f'[{r["label"]}] {r["original_text"]}')
        print(f'   Confidence: {r["confidence"]}')
        print(f'   Reasoning: {r["reasoning"]}')
        print()
except Exception as e:
    print(f"Error: {e}")
    print("Make sure GOOGLE_API_KEY is set correctly")
