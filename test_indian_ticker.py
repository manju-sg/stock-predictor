import urllib.request
import json
import time

url = "http://127.0.0.1:5000/api/predict"
data = json.dumps({"ticker": "RELIANCE.NS"}).encode('utf-8')

req = urllib.request.Request(
    url, 
    data=data, 
    headers={'Content-Type': 'application/json'}
)

print("Requesting 1-Month forecast for RELIANCE.NS... (This involves training 30 models, please wait ~10-15s)")
start_time = time.time()

try:
    response = urllib.request.urlopen(req)
    result = json.loads(response.read().decode('utf-8'))
    
    print(f"\n--- SUCCESS in {time.time() - start_time:.2f} seconds ---")
    print(f"Ticker: {result['ticker']}")
    print(f"Latest Date (Real World Today): {result['latest_date']}")
    print(f"Latest Price (Live 2026): {result['latest_price']}")
    
    print("\nSentiment Info:")
    print(f"Score: {result['sentiment']['score']}")
    print(f"Top Headline: {result['sentiment']['headlines'][0]}")
    
    print(f"\nForecast for Day 30 ({result['future_dates'][-1]}): {result['predictions'][-1]:.2f}")
    
except Exception as e:
    print(f"\n--- ERROR ---")
    print(e)
