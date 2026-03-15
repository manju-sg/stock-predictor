import urllib.request
try:
    req = urllib.request.Request('http://127.0.0.1:5000/api/predict', data=b'{"ticker": "RELIANCE.NS"}', headers={'Content-Type': 'application/json'})
    urllib.request.urlopen(req)
except Exception as e:
    print(e.read().decode('utf-8'))
