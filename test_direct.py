import traceback
from data_loader import fetch_live_ticker_data
from feature_engineering import add_technical_indicators
from model_multistep import MultiStepForecaster
from news_analyzer import get_news_sentiment

ticker = 'RELIANCE.NS'
try:
    print("1. News...")
    news_data = get_news_sentiment(ticker)
    
    print("2. Data fetch...")
    df = fetch_live_ticker_data(ticker, period="5y")
    
    print("3. Feature engineering...")
    df = add_technical_indicators(df)
    
    print("4. Forecaster training...")
    features = [col for col in df.columns if col not in ['date', 'ticker']]
    forecaster = MultiStepForecaster(steps=2)
    forecaster.train_all(df, features_list=features)
    
    print("SUCCESS")
except Exception as e:
    print("FAILED")
    print(traceback.format_exc())
