import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

def get_news_sentiment(ticker_symbol):
    """
    Fetches the latest news for a ticker from Yahoo Finance,
    runs VADER sentiment analysis on the title and summary, 
    and returns an aggregate score along with the headlines.
    Score ranges from -1.0 (Most Negative) to +1.0 (Most Positive).
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
    except Exception as e:
        print(f"Error fetching news from yfinance for {ticker_symbol}: {e}")
        return {"score": 0.0, "headlines": ["Could not fetch live news."]}

    headlines = []
    texts_to_analyze = []
    
    if news:
        for item in news[:6]:  # Get top 6 news items
            title = item.get('title', '')
            summary = item.get('summary', '') # Summary provides more context
            
            if title:
                headlines.append(title)
                # Combine title and summary for richer sentiment analysis
                combined_text = f"{title}. {summary}"
                texts_to_analyze.append(combined_text)

    if not texts_to_analyze:
        return {"score": 0.0, "headlines": ["No recent news available."]}

    analyzer = SentimentIntensityAnalyzer()
    total_compound = 0.0
    
    # We use up to 5 texts for final analysis
    final_texts = texts_to_analyze[:5]
    final_headlines = headlines[:5]
    
    for text in final_texts:
        sentiment = analyzer.polarity_scores(text)
        total_compound += sentiment['compound']
        
    avg_score = total_compound / len(final_texts)
    
    return {
        "score": avg_score,
        "headlines": final_headlines
    }

if __name__ == '__main__':
    # Test
    res = get_news_sentiment('MSFT')
    print("MSFT Sentiment Score:", res['score'])
    print("Headlines:", res['headlines'])

