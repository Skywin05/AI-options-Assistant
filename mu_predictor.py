from model_utils import run_model_for

if __name__ == "__main__":
    # change this ticker to test different stocks
    ticker = "MU"

    result = run_model_for(ticker)

    print(f"Test Accuracy for {ticker}: {result['test_accuracy']:.2%}")
    print(f"\n{ticker} 5-Day Direction Signal")
    print("Direction:", result["direction"])
    print("Confidence:", f"{result['confidence']:.2%}")

    print("\nSignals (last ~2 weeks):")
    print(result["signals"])

    print("\nPaper-Trade PnL (1 share per signal):")
    print(result["pnl_table"])
    print("Total PnL:", result["total_pnl"])
