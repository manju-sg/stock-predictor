import yfinance as yf
import pandas as pd

def fetch_live_ticker_data(ticker_symbol, period="5y"):
    """
    Downloads raw daily stock prices directly from Yahoo Finance.
    Returns a pandas DataFrame formatted identically to the previous Kaggle dataset
    so it plugs effortlessly into our feature engineering pipeline.
    """
    print(f"Fetching live data for {ticker_symbol} over the last {period}...")
    
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=period)
    
    if df.empty:
        return pd.DataFrame()
        
    df = df.reset_index()
    
    # yfinance returns columns like 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
    # We will standardize these to lowercase to match our existing system
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Ensure date has no timezone offset (for simplicity)
    if 'date' in df.columns:
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
            
    df['ticker'] = ticker_symbol
    return df

if __name__ == '__main__':
    # Test Indian Stock
    print("Testing Live Data fetch on Reliance Industries (RELIANCE.NS):")
    df = fetch_live_ticker_data("RELIANCE.NS", period="1mo")
    print(df.tail())
