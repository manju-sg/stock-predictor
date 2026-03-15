from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from data_loader import fetch_live_ticker_data
from feature_engineering import add_technical_indicators
from model_multistep import MultiStepForecaster
from news_analyzer import get_news_sentiment
import traceback
import urllib.request
import json
import pandas as pd
import datetime

app = Flask(__name__)
CORS(app)

# We no longer cache a giant static CSV!
print("Backend Live Data Pipeline Initialized.")

@app.route('/')
def serve_ui():
    return render_template('index.html')

@app.route('/api/tickers', methods=['GET'])
def get_tickers():
    try:
        # Suggested popular tickers. The UI will now allow ANY custom input though!
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", 
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ZOMATO.NS", "TATAMOTORS.NS"
        ]
        tickers.sort()
        return jsonify({"tickers": tickers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_tickers():
    query = request.args.get('q', '')
    if len(query) < 1:
        return jsonify([])
        
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        res = urllib.request.urlopen(req, timeout=5)
        data = json.loads(res.read())
        
        results = []
        if 'quotes' in data:
            for q in data['quotes']:
                # Filter out cryptocurrencies or weird indices if mostly looking for stocks/mutual funds
                if q.get('quoteType') in ['EQUITY', 'MUTUALFUND', 'ETF', 'INDEX']:
                    symbol = q.get('symbol')
                    name = q.get('shortname') or q.get('longname') or 'Unknown'
                    exch = q.get('exchange', '')
                    results.append({
                        "symbol": symbol,
                        "name": name,
                        "exchange": exch
                    })
        return jsonify(results[:10]) # Return top 10
    except Exception as e:
        print(f"Error fetching search from Yahoo: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker')
        
        if not ticker:
            return jsonify({"error": "Ticker is required."}), 400
            
        # Ensure proper casing for YFinance (e.g. aapl -> AAPL)
        ticker = ticker.upper()
        print(f"Prediction requested for: {ticker}")
        
        # 0. Fetch Real-time News Sentiment
        news_data = get_news_sentiment(ticker)
        sentiment_score = news_data['score']
        headlines = news_data['headlines']
        
        # 1. Fetch live up-to-the-minute dataset from Yahoo Finance
        df = fetch_live_ticker_data(ticker, period="5y")
        if df.empty:
            return jsonify({"error": f"No data found for ticker '{ticker}'. Please ensure it is a valid Yahoo Finance symbol."}), 404
            
        df = add_technical_indicators(df)
        
        cols_to_drop = ['brand_name', 'industry_tag', 'country', 'dividends', 'stock splits', 'capital gains']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        features = [col for col in df.columns if col not in ['date', 'ticker']]
        
        # 2. Train 30-day (1-Month) Multi-Step Forecaster
        # This will train 30 separate XGBoost models
        forecaster = MultiStepForecaster(steps=30)
        forecaster.train_all(df, features_list=features)
        
        # 3. Forecast Future from Latest Row
        latest_row = df.iloc[[-1]].copy()
        raw_predictions = forecaster.forecast_from_latest(latest_row)
        
        # Extract recent volatility to scale sentiment impact
        # If volatility_30 is NaN (e.g., not enough data), default to a moderate 0.2
        recent_volatility = latest_row['volatility_30'].iloc[0]
        if pd.isna(recent_volatility) or recent_volatility == 0:
            recent_volatility = 0.2
            
        # 4. Integrate Sentiment (Mathematical Adjustment)
        # We apply a multiplier based on sentiment AND recent stock volatility.
        # Highly volatile stocks react more to sentiment than stable stocks.
        final_predictions = []
        for i, pred in enumerate(raw_predictions):
            if pred is not None:
                # Base impact scaled by volatility (e.g., 0.2 vol * 0.15 scalar = max 3% impact)
                # impact decays slightly over the 30 days
                impact = sentiment_score * recent_volatility * 0.15 * (1 - (i / 60)) 
                adjusted_pred = pred * (1 + impact)
                final_predictions.append(float(adjusted_pred))
            else:
                final_predictions.append(None)
        
        # Prepare historical data for the chart (last 90 days for better 30-day context)
        historic_df = df.dropna(subset=['close']).tail(90)
        historical_data = {
            "dates": historic_df['date'].astype(str).tolist(),
            "prices": historic_df['close'].tolist()
        }
        
        # Extract the price we will link the graph to
        latest_price = float(latest_row['close'].iloc[0])
        
        # 5. Lock Dates strictly to Live Real-World Time (Ignorance of Kaggle stale dates)
        
        # We enforce that the forecast strictly starts tomorrow from the REAL WORLD date
        live_today = datetime.datetime.now().date()
        
        future_dates = []
        curr_date = pd.to_datetime(live_today)
        
        # Generate 30 future active market days
        for i in range(1, 31):
            curr_date += pd.Timedelta(days=1)
            while curr_date.weekday() >= 5: # Skip weekends
                curr_date += pd.Timedelta(days=1)
            future_dates.append(str(curr_date.date()))
            
        return jsonify({
            "ticker": ticker,
            "latest_date": str(live_today),
            "latest_price": latest_price,
            "predictions": final_predictions,
            "future_dates": future_dates,
            "historical_data": historical_data,
            "sentiment": {
                "score": sentiment_score,
                "headlines": headlines
            }
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
