from AiPackageWrapper.sentiment_analyzer_module import SentimentAnalyzer as analyzer

def test_analyze_text_positive():
    result = analyzer.analyze_text("I love this!")
    assert result == "positive"

def test_analyze_text_negative():
    result = analyzer.analyze_text("I hate this!")
    assert result == "negative"

def test_analyze_text_neutral():
    result = analyzer.analyze_text("It is a table.")
    assert result == "neutral"
