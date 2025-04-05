from textblob import TextBlob

class SentimentAnalyzer:
    def analyze_text(text):
        """
        Analyzes the sentiment of a single text.

        Parameters:
            text (str): The text to analyze.

        Returns:
            str: Sentiment label ('positive', 'negative', or 'neutral').
        """
        # Clean text and create a TextBlob object
        blob = TextBlob(text.lower())

        # Calculate sentiment polarity (-1 to 1)
        polarity = blob.sentiment.polarity

        # Determine sentiment based on polarity
        if polarity > 0:
            return 'positive'
        elif polarity < 0:
            return 'negative'
        else:
            return 'neutral'

    def analyze_batch(texts):
        """
        Processes a batch of texts and aggregates sentiment results.

        Parameters:
            texts (list of str): List of texts to analyze.

        Returns:
            dict: Aggregated sentiment counts and percentages.
        """
        results = {"positive": 0, "negative": 0, "neutral": 0}

        for text in texts:
            sentiment = analyze_text(text)
            results[sentiment] += 1

        # Calculate percentages
        total = len(texts)
        percentages = {k: (v / total) * 100 for k, v in results.items()}

        return {"counts": results, "percentages": percentages}