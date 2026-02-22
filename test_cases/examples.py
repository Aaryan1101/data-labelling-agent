"""
Example Usage of the LangChain Data Labelling Agent
====================================================
Demonstrates all 3 modes:
  1. Inline labelling (direct Python API)
  2. Chain-based file labelling (CLI)
  3. Autonomous Agent mode (CLI)

Prerequisites:
    pip install langchain langchain-openai
    export OPENAI_API_KEY="sk-..."
"""

from labelling_agent import label_data_inline, build_labelling_chain, build_labelling_agent

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EXAMPLE 1: Inline Labelling (Programmatic API)                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def example_sentiment_analysis():
    """Label customer reviews with sentiment."""
    reviews = [
        "This product is amazing! Best purchase I've ever made.",
        "Terrible quality. Broke after one day. Want a refund.",
        "It's okay, nothing special but gets the job done.",
        "Absolutely love it! Exceeded all my expectations.",
        "Worst customer service I've ever experienced.",
        "Decent product for the price. Would consider buying again.",
    ]

    results = label_data_inline(
        data=reviews,
        labels=["positive", "negative", "neutral"],
        task_description="Classify customer reviews by sentiment.",
    )

    print("=== Sentiment Analysis Results ===")
    for item in results:
        print(f"\n📝 {item['original_text'][:60]}...")
        print(f"   🏷️  Label: {item['label']} (confidence: {item['confidence']})")
        print(f"   💬 Reason: {item['reasoning']}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EXAMPLE 2: Topic Classification with Few-Shot Examples                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def example_topic_classification():
    """Classify news headlines into topics using few-shot examples."""
    headlines = [
        "Fed raises interest rates by 0.25%",
        "New study finds link between sleep and memory",
        "Lakers defeat Celtics in overtime thriller",
        "SpaceX launches 40 satellites into orbit",
        "Senate passes new infrastructure bill",
        "Researchers develop new cancer treatment approach",
    ]

    few_shot = [
        {"text": "Stock market hits all-time high", "label": "finance"},
        {"text": "Scientists discover high-energy particle", "label": "science"},
        {"text": "World Cup finals draw massive viewership", "label": "sports"},
        {"text": "Congress debates new tax reform", "label": "politics"},
    ]

    results = label_data_inline(
        data=headlines,
        labels=["finance", "science", "sports", "politics", "technology"],
        few_shot_examples=few_shot,
        task_description="Classify news headlines into topic categories.",
    )

    print("\n=== Topic Classification Results ===")
    for item in results:
        print(f"  [{item['label']:>12}] {item['original_text']}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EXAMPLE 3: Auto-Discovery Labels (No Predefined Labels)              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def example_auto_discovery():
    """Let the LLM discover appropriate labels automatically."""
    emails = [
        "Hi, I can't log into my account. Please help!",
        "When will my order #12345 arrive?",
        "I'd like to cancel my subscription effective immediately.",
        "Your product is great! Just wanted to say thanks.",
        "Can I get a refund for my last purchase?",
        "How do I upgrade to the premium plan?",
    ]

    results = label_data_inline(
        data=emails,
        labels=None,  # Let the model discover labels
        task_description="Categorize customer support emails by intent/type.",
    )

    print("\n=== Auto-Discovered Labels ===")
    for item in results:
        print(f"  [{item['label']:>20}] {item['original_text'][:50]}...")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EXAMPLE 4: Using the Autonomous Agent                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def example_agent_mode():
    """Use the autonomous agent to read, label, and save data."""
    agent = build_labelling_agent(
        model_name="gpt-4o-mini",
        labels=["spam", "not_spam"],
        task_description="Classify text messages as spam or not spam.",
    )

    result = agent.invoke({
        "input": (
            "Read the data from 'sample_data/messages.txt', "
            "label each message as spam or not_spam, "
            "and save the labelled results to 'sample_data/messages_labelled.json'."
        )
    })
    print("\n=== Agent Output ===")
    print(result["output"])


# ── Run Examples ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("LangChain Data Labelling Agent - Examples")
    print("=" * 50)
    print("\nNote: Set OPENAI_API_KEY environment variable before running.\n")

    # Uncomment the example you want to run:
    # example_sentiment_analysis()
    # example_topic_classification()
    # example_auto_discovery()
    # example_agent_mode()

    # Quick demo (prints structure without API call)
    print("Available examples:")
    print("  1. example_sentiment_analysis()  - Sentiment on reviews")
    print("  2. example_topic_classification() - News headline topics (few-shot)")
    print("  3. example_auto_discovery()       - Auto-discover email labels")
    print("  4. example_agent_mode()           - Autonomous file-based agent")
    print("\nUncomment the desired function in __main__ and run with:")
    print("  OPENAI_API_KEY=sk-... python examples.py")
