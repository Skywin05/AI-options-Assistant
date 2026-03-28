import streamlit as st
from datetime import datetime
from model_utils import run_model_for
import yfinance as yf
import datetime as dt
positive_words = ["beats", "beat", "surge", "rally", "upgrade", "strong",
                  "record", "bullish", "profit", "soars", "jump"]
negative_words = ["miss", "cuts", "cut", "downgrade", "plunge", "fall",
                  "weak", "loss", "bearish", "lawsuit", "drops", "drop"]

def get_sentiment_color(title: str):
    t = title.lower()
    if any(w in t for w in positive_words):
        return "green"   # good news
    if any(w in t for w in negative_words):
        return "red"     # bad news
    return "gray"        # neutral
st.set_page_config(page_title="AI Options Assistant", layout="wide")

st.title("AI Options Assistant")

st.markdown(
    "Enter a stock ticker (like **MU**, **WDC**, **AVGO**) and get a 5-day "
    "direction signal, confidence, and recent paper-trade performance."
)

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker symbol", value="MU").upper()
lookback = st.sidebar.slider("Lookback (days of history)", min_value=20, max_value=60, value=30, step=5)

if st.sidebar.button("Run Analysis"):
    try:
        with st.spinner(f"Running model for {ticker}..."):
            result = run_model_for(ticker, lookback=lookback)

                    # Earnings awareness
        earnings_date = result.get("earnings_date", None)
        today = datetime.now().date()

        if earnings_date:
            days_to_earnings = (earnings_date.date() - today).days
            if 0 <= days_to_earnings <= 7:
                st.warning(
                    f"Earnings around {earnings_date.date()} "
                    f"(~{days_to_earnings} days away). "
                    "Model is technical only; expect higher risk and IV moves."
                )
    else:
      st.info(
        "No earnings within the next 7 days. "
        "Signal is based on technicals only (no news/earnings input)."
    )

st.subheader("Latest news (Yahoo Finance)")
        try:
            tk = yf.Ticker(ticker)

            news_items = getattr(tk, "news", []) or []

            if not news_items:
                st.write("No recent news found for this ticker.")
            else:
                max_items = 10
                for item in news_items[:max_items]:
                    title = item.get("title", "No title")
                    link = item.get("link", "")
                    publisher = item.get("publisher", "Unknown source")
                    ts = item.get("providerPublishTime")

                    if isinstance(ts, (int, float)):
                        published = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                    else:
                        published = "Unknown time"

                    color = get_sentiment_color(title)

                    st.markdown(f":{color}[**{title}**]")
                    st.write(f"{publisher} • {published}")
                    if link:
                        st.markdown(f"[Read article]({link})")
                    st.markdown("---")
        except Exception as e:
            st.write("Could not load news for this ticker.")
            st.write(e)
                    
                )
        else:
            st.info(
                "Earnings date not available. Treat this as a pure technical signal "
                "and double-check the calendar/news manually."
            )
        
        st.subheader(f"{ticker} 5-Day Direction Signal")
        st.write(f"**Direction:** {result['direction']}")
        st.write(f"**Confidence:** {result['confidence']:.2%}")
        st.write(f"**Test Accuracy (past data):** {result['test_accuracy']:.2%}")
        st.write(f"**Total paper-trade PnL (1 share):** {result['total_pnl']:.2f}")

        st.markdown("---")
        st.subheader("Signals (last ~2 weeks)")
        st.dataframe(result["signals"])

        st.subheader("Paper-Trade PnL (1 share per signal)")
        st.dataframe(result["pnl_table"])

        bias = "bullish" if "CALL" in result["direction"] else "bearish"
        st.markdown("### Simple Trade Idea (Educational)")
        st.write(
            f"- Model bias is **{bias}** over the next 5 trading days.\n"
            f"- Confidence is **{result['confidence']:.2%}**, with historical accuracy "
            f"around **{result['test_accuracy']:.2%}**.\n"
            "- Consider this as **one input** only; always check chart, news, and implied volatility "
            "before taking real options trades."
        )

    except Exception as e:
        st.error(f"Error running model for {ticker}: {e}")
else:
    st.info("Set your ticker and click **Run Analysis** in the sidebar.")
