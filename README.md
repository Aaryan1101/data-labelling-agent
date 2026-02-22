# LangChain Data Labelling Agent

A LangChain-based agent that takes **unlabelled data** as input and produces **labelled data** as output using LLMs.

## Features

- **3 Modes**: Inline API, Chain pipeline, Autonomous Agent
- **Flexible Input**: CSV, JSON, plain text files
- **Predefined or Auto-discovered Labels**: Provide allowed labels or let the LLM discover them
- **Few-shot Learning**: Supply examples for better accuracy
- **Batch Processing**: Handles large datasets with automatic batching
- **Structured Output**: JSON with label, confidence score, and reasoning

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## Usage

### 1. Python API (Inline Mode)

```python
from labelling_agent import label_data_inline

data = [
    "This product is amazing!",
    "Terrible quality, want a refund.",
    "It's okay, nothing special.",
]

results = label_data_inline(
    data=data,
    labels=["positive", "negative", "neutral"],
    task_description="Classify customer reviews by sentiment.",
)

for r in results:
    print(f"[{r['label']}] {r['original_text']}")
```

### 2. CLI - Chain Mode (Direct Pipeline)

```bash
# With predefined labels
python labelling_agent.py \
  -i sample_data/reviews.txt \
  -o output_labelled.json \
  -l positive negative neutral \
  -t "Classify reviews by sentiment"

# Auto-discover labels
python labelling_agent.py \
  -i sample_data/headlines.csv \
  -o headlines_labelled.csv
```

### 3. CLI - Agent Mode (Autonomous)

```bash
python labelling_agent.py \
  --mode agent \
  -i sample_data/reviews.txt \
  -o reviews_labelled.json \
  -l positive negative neutral \
  -t "Classify reviews by sentiment"
```

### 4. Few-Shot Examples (Python API)

```python
results = label_data_inline(
    data=headlines,
    labels=["finance", "science", "sports", "politics"],
    few_shot_examples=[
        {"text": "Stock market hits high", "label": "finance"},
        {"text": "New particle discovered", "label": "science"},
    ],
    task_description="Classify news headlines.",
)
```

## Output Format

```json
[
  {
    "original_text": "This product is amazing!",
    "label": "positive",
    "confidence": 0.95,
    "reasoning": "Strong positive language with exclamation"
  }
]
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Unlabelled Data │────▶│  LangChain Agent  │────▶│  Labelled Data  │
│  (CSV/JSON/TXT)  │     │  + LLM (GPT-4o)  │     │  (CSV/JSON)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              Few-shot Examples   Predefined Labels
              (optional)         (optional)
```

## Modes Explained

| Mode | Use Case | How It Works |
|------|----------|-------------|
| **Inline** | Python scripts, notebooks | Direct function call, returns list of dicts |
| **Chain** | Batch file processing | Reads file → labels → saves output |
| **Agent** | Complex workflows | Autonomous agent with tool access |
