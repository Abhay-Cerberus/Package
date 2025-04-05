from modules import SentimentAnalyzer

sample_text = "I love this product! It works great and exceeds expectations."
print("Single Text Analysis:", analyze_text(sample_text))

batch_texts = [
    "I love this product!",
    "This is the worst experience ever.",
    "It's okay, not too bad.",
    "Absolutely fantastic!",
    "I'm not happy with this."
]
batch_result = analyze_batch(batch_texts)
print("Batch Analysis:", batch_result)