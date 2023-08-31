import yfinance as yf
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from xgboost import XGBClassifier

btcTicker = yf.Ticker("BTC-USD")

if os.path.exists("btc.csv"):
    btc = pd.read_csv("btc.csv", index_col=0)
else:
    btc = btcTicker.history(period="max")
    btc.to_csv("btc.csv")

btc.index = pd.to_datetime(btc.index)

del btc["Dividends"]
del btc["Stock Splits"]

btc.columns = [c.lower() for c in btc.columns]

# btc.plot.line(y="close", use_index=True)

wikiLink = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)

wikiLink.index = pd.to_datetime(wikiLink.index).tz_localize('UTC')
btc = btc.merge(wikiLink, left_index=True, right_index=True)
btc["tomorrow"] = btc["close"].shift(-1)
btc["target"] = (btc["tomorrow"] > btc["close"]).astype(int)
# btc["target"].value_counts()

model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)

train = btc.iloc[:-200]
test = btc.iloc[-200:]

predictWords = ["close", "volume", "open", "high", "low", "edit_count", "sentiment", "neg_sentiment"]
model.fit(train[predictWords], train["target"])

predictions = model.predict(test[predictWords])
predictions = pd.Series(predictions, index=test.index)

def predict(train, test, predictWords, model):
    model.fit(train[predictWords], train["target"])
    predictions = model.predict(test[predictWords])
    predictions = pd.Series(predictions, index=test.index, name="predictions")
    combinations = pd.concat([test["target"], predictions], axis=1)
    return combinations

def backtest(data, model, predictWords, start=1095, step=150):
    predictionTotal = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictWords, model)
        predictionTotal.append(predictions)
    
    return pd.concat(predictionTotal)

model = XGBClassifier(random_state=1, learning_rate=.1, n_estimators=200)
predictions = backtest(btc, model, predictWords)

def compute_rolling(btc):
    horizons = [2,7,60,365]
    newPredictors = ["close", "sentiment", "neg_sentiment"]

    for horizon in horizons:
        currentAverages = btc.rolling(horizon, min_periods=1).mean()

        ratioColumn = f"close_ratio_{horizon}"
        btc[ratioColumn] = btc["close"] / currentAverages["close"]
        
        editColumn = f"edit_{horizon}"
        btc[editColumn] = currentAverages["edit_count"]

        rolling = btc.rolling(horizon, closed='left', min_periods=1).mean()
        trendColumn = f"trend_{horizon}"
        btc[trendColumn] = rolling["target"]

        newPredictors += [ratioColumn, trendColumn, editColumn]
    return btc, newPredictors

btc, newPredictors = compute_rolling(btc.copy())

predictions = backtest(btc, model, newPredictors)
print(predictions)
