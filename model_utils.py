import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from datetime import datetime, timedelta


def run_model_for(ticker: str, lookback: int = 30):
    """
    Train an XGBoost 5-day direction model for the given ticker
    and return a summary dictionary.
    """

    # 1) DOWNLOAD DATA
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y", interval="1d")

        # Earnings info (if available)
    earnings_date = None
    try:
        cal = stock.calendar
        if cal is not None and not cal.empty:
            for col in ["Earnings Date", "Earnings Date 0", "EarningsDate"]:
                if col in cal.index or col in cal.columns:
                    val = cal.loc[col].values[0] if col in cal.index else cal[col].iloc[0]
                    earnings_date = pd.to_datetime(val).to_pydatetime()
                    break
    except Exception:
        earnings_date = None

    if df.empty or len(df) < lookback + 10:
        raise ValueError(f"Not enough data for {ticker}")

    # 2) CREATE 5-DAY DIRECTION LABEL
    df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

    # 3) SIMPLE TECHNICAL INDICATORS
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["Vol_Change"] = df["Volume"].pct_change()

    df = df.dropna()

    features = ["Close", "RSI", "MACD", "SMA_20", "Vol_Change"]
    X = df[features].values
    y = df["Target"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 4) MAKE SEQUENCES
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - lookback):
        window = X_scaled[i:i + lookback]
        X_seq.append(window.flatten())
        y_seq.append(y[i + lookback])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # 5) TRAIN/TEST SPLIT
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # 6) TRAIN MODEL
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # 7) TEST ACCURACY
    y_pred_test = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
    test_acc = accuracy_score(y_test, y_pred_test)

    # 8) LATEST SIGNAL
    latest_window = X_scaled[-lookback:]
    latest_flat = latest_window.flatten().reshape(1, -1)
    prob_up = model.predict_proba(latest_flat)[0][1]

    direction = "CALL / Long (Up)" if prob_up > 0.5 else "PUT / Short (Down)"
    confidence = prob_up if prob_up > 0.5 else 1 - prob_up

    # 9) SIMPLE 2-WEEK PAPER TRADING
    signals = []
    start_idx = len(X_seq) - 10  # last ~2 weeks (10 trading days)

    for i in range(start_idx, len(X_seq)):
        probs = model.predict_proba(X_seq[i].reshape(1, -1))[0][1]
        sig = 1 if probs > 0.5 else 0
        date = df.index[i + lookback]
        close = df["Close"].iloc[i + lookback]
        signals.append({
            "Date": date,
            "Close": float(close),
            "Prob_Up": float(probs),
            "Signal": int(sig),
        })

    signals_df = pd.DataFrame(signals)

    # PnL: buy 1 share on signal=1, sell 5 days later
    pnl_records = []
    for row in signals_df.itertuples():
        entry_idx = df.index.get_loc(row.Date)
        exit_idx = entry_idx + 5
        if exit_idx >= len(df):
            continue
        entry_price = df["Close"].iloc[entry_idx]
        exit_price = df["Close"].iloc[exit_idx]
        pnl = (exit_price - entry_price) * row.Signal
        pnl_records.append({
            "Date": row.Date,
            "Signal": row.Signal,
            "PnL": float(pnl),
        })

    pnl_df = pd.DataFrame(pnl_records)
    total_pnl = float(pnl_df["PnL"].sum()) if not pnl_df.empty else 0.0

    # RETURN EVERYTHING IN A DICT (EASY FOR UI)
    return {
        "ticker": ticker,
        "test_accuracy": float(test_acc),
        "direction": direction,
        "confidence": float(confidence),
        "signals": signals_df,
        "pnl_table": pnl_df,
        "total_pnl": total_pnl,
        "earnings_date": earnings_date,
    }
    
