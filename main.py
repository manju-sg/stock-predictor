import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_stock_data, get_ticker_data
from feature_engineering import add_technical_indicators, generate_target_variable
from model_xgboost import XGBoostPredictor

def plot_predictions(dates, actuals, predictions, ticker, days_ahead):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actuals, label='Actual Price', color='blue')
    plt.plot(dates, predictions, label='Predicted Price', color='orange', linestyle='--')
    plt.title(f'{ticker} Stock Price Prediction ({days_ahead} Days Ahead)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{ticker}_prediction_{days_ahead}d.png')
    print(f"Plot saved to '{ticker}_prediction_{days_ahead}d.png'")

def main(ticker, days_ahead, test_size_days):
    print(f"--- Starting Advanced Stock Predictor for {ticker} ---")
    
    # 1. Load Data
    all_data = load_stock_data()
    df = get_ticker_data(all_data, ticker)
    
    if df.empty:
        print(f"No data found for ticker '{ticker}'. Please ensure it exists in the dataset. Common tickers: 'AAPL', 'MSFT', 'GOOGL'.")
        return

    # 2. Feature Engineering
    print("Applying technical indicators and feature engineering...")
    df = add_technical_indicators(df)
    
    # Generate Target
    # We want to predict the price `days_ahead` in the future
    target_col = f'target_{days_ahead}d'
    df = generate_target_variable(df, target_col='close', days_ahead=days_ahead)
    
    # Drop columns that are mostly empty or not useful for ML
    cols_to_drop = ['brand_name', 'industry_tag', 'country', 'dividends', 'stock splits', 'capital gains']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Drop NaNs created by indicators & shifting
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        print("Not enough data points after adding technical indicators. Try a different ticker.")
        return

    # 3. Train-Test Split (Chronological)
    print(f"Total available records for {ticker}: {len(df)}")
    
    # The last `test_size_days` are used as the hold-out test set
    train_df = df.iloc[:-test_size_days].copy()
    test_df = df.iloc[-test_size_days:].copy()
    
    print(f"Training on {len(train_df)} records, testing on the most recent {len(test_df)} records.")

    # 4. Define Features to use
    features = [
        col for col in df.columns 
        if col not in ['date', 'brand_name', 'ticker', 'industry_tag', 'country', target_col]
    ]
    
    # 5. Model Training & Evaluation
    print("Training XGBoost Model...")
    predictor = XGBoostPredictor(target_col=target_col, features=features)
    predictor.train(train_df)
    
    print("Evaluating Model on hold-out test set...")
    predictions = predictor.predict(test_df)
    metrics = predictor.evaluate(test_df, predictions, current_price_col='close')
    
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        if "Accuracy" in k:
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v:.4f}")
            
    # 6. Forecasting the Future (Unknown territory)
    # The last available rows in df originally had target=NaN before dropna. 
    # To truly forecast tomorrow, we construct a feature set from the very last known day
    latest_data = get_ticker_data(all_data, ticker)
    latest_data = add_technical_indicators(latest_data)
    
    # Extract the absolute most recent finalized day
    latest_day = latest_data.iloc[[-1]].copy()
    future_pred = predictor.predict(latest_day)
    last_date = latest_day['date'].iloc[0]
    curr_price = latest_day['close'].iloc[0]
    
    print(f"\n--- FORECAST ---")
    print(f"Latest Known Data Date: {last_date}")
    print(f"Latest Known Current Price: ${curr_price:.2f}")
    print(f"Model Prediction for {days_ahead} market day(s) from then: ${future_pred[0]:.2f}")
    
    # Calculate % change predicted
    pct_change = ((future_pred[0] - curr_price) / curr_price) * 100
    print(f"Predicted Move: {pct_change:+.2f}%")

    # 7. Visualization
    plot_predictions(test_df['date'], test_df[target_col], predictions, ticker, days_ahead)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Stock Predictor')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol to model (e.g., AAPL, MSFT, TSLA)')
    parser.add_argument('--days_ahead', type=int, default=1, help='Number of days ahead to forecast price')
    parser.add_argument('--test_size', type=int, default=60, help='Number of recent days to hold out for evaluation (backtesting)')
    args = parser.parse_args()
    
    main(args.ticker, args.days_ahead, args.test_size)
