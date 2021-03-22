"""QF 435 - B
Group 7
Project 1
Question 1"""

import pandas as pd
from pandas_datareader.data import DataReader
import math
import pprint
import copy
import matplotlib.pyplot as plt
import numpy as np


def get_symbols():
    """Retrieve ticker price data from yahoo fin."""
    tickers = ["AAPL", "CSCO", "HON", "KO", "NKE",  "WBA", "AMGN", "CVX", "IBM", "MCD", "PG", "WMT", "AXP",
               "DIS", "INTC", "MMM", "TRV", "BA", "GS", "JNJ", "MRK", "UNH", "CAT", "HD", "JPM", "MSFT",  "VZ"]
    portfolio_df = []
    for ticker in tickers:
        print(f"Getting {ticker} data...")
        portfolio_df.append(DataReader(ticker, 'yahoo', "1999-05-01", "2020-12-31")['Adj Close'].rename(ticker))
    portfolio_df = pd.concat([stock_prices for stock_prices in portfolio_df], axis=1)
    portfolio_df.to_csv("stock_data.csv")


def load_price_data():
    """Load stock data from csv."""
    return pd.read_csv("stock_data.csv")


def abs_performance_summary(tickers, df_returns):
    # 1.1 - Absolute Performance Summary
    stock_summary = {"Mean Return": 0, "Volatility": 0, "Sharpe": 0}
    overall_summary = {}
    for ticker in tickers:
        stock_summary["Mean Return"] = df_returns[ticker].mean() * 252
        stock_summary["Volatility"] = math.sqrt(df_returns[ticker].var()) * math.sqrt(252)

        stock_ret = df_returns[ticker]
        spy_ret = df_returns["SPY"]
        excess_ret = stock_ret.subtract(spy_ret)
        vol = math.sqrt(excess_ret.var()) * math.sqrt(252)
        stock_summary["Sharpe"] = stock_summary["Mean Return"] / vol
        #stock_summary["Sharpe"] = stock_summary["Mean Return"] / stock_summary["Volatility"]  # Assume 0 rf.
        overall_summary[ticker] = copy.deepcopy(stock_summary)

    output_summary = {"Mean Return": {"Min": 0, "Mean": 0, "Max": 0}, "Volatility": {"Min": 0, "Mean": 0, "Max": 0},
                      "Sharpe": {"Min": 0, "Mean": 0, "Max": 0}}

    # Mean Return

    output_summary["Mean Return"]["Min"] = min(d["Mean Return"] for d in overall_summary.values())
    output_summary["Mean Return"]["Mean"] = sum(d["Mean Return"] for d in overall_summary.values()) / 27
    output_summary["Mean Return"]["Max"] = max(d["Mean Return"] for d in overall_summary.values())

    # Vol

    output_summary["Volatility"]["Min"] = min(d["Volatility"] for d in overall_summary.values())
    output_summary["Volatility"]["Mean"] = sum(d["Volatility"] for d in overall_summary.values()) / 27
    output_summary["Volatility"]["Max"] = max(d["Volatility"] for d in overall_summary.values())

    # Sharpe

    output_summary["Sharpe"]["Min"] = min(d["Sharpe"] for d in overall_summary.values())
    output_summary["Sharpe"]["Mean"] = sum(d["Sharpe"] for d in overall_summary.values()) / 27
    output_summary["Sharpe"]["Max"] = max(d["Sharpe"] for d in overall_summary.values())

    output_df = pd.concat({
        k: pd.DataFrame.from_dict(v, 'index').T for k, v in output_summary.items()
    },
        axis=0).T
    return output_df


def gen_plot(overall_summary):
    # 1.2 - Plot mean returns against volatility
    mean_returns = []
    volatility = []
    for d in overall_summary.values():
        mean_returns.append(d["Mean Return"])
        volatility.append(d["Volatility"])

    m, b = np.polyfit(volatility, mean_returns, 1)

    plt.plot(volatility, mean_returns, "o")
    plt.plot(volatility, m*np.array(volatility) + b)
    plt.xlabel("Volatiltiy")
    plt.ylabel("Mean Return")
    plt.show()


def get_risk_metrics(tickers, df_returns):
    # 1.3 - SPY ETF

    spy_var = df_returns["SPY"].var()
    spy_mean_ret = df_returns["SPY"].mean() * 252

    # (b) - Market Beta
    stock_betas = {}
    for ticker in tickers:
        cov_matrix = df_returns.cov()
        cov = cov_matrix[ticker]["SPY"]
        stock_betas[ticker] = cov / spy_var

    # (a) - Jensen Alpha
    jensen_alphas = {}
    for ticker in tickers:
        mean_ret = df_returns[ticker].mean() * 252
        market_beta = stock_betas[ticker]
        jensen_alphas[ticker] = mean_ret - (market_beta * spy_mean_ret)

    # (c) - Treynor Ratio
    treynor_ratios = {}
    for ticker in tickers:
        mean_ret = df_returns[ticker].mean() * 252
        market_beta = stock_betas[ticker]
        treynor_ratios[ticker] = mean_ret / market_beta

    # (d) - Tracking Error
    spy_ret = df_returns["SPY"]
    tracking_errors = {}
    for ticker in tickers:
        stock_ret = df_returns[ticker]
        diff = stock_ret.subtract(spy_ret)
        tracking_errors[ticker] = math.sqrt(diff.var()) * math.sqrt(252)

    # (e) - Information Ratio
    information_ratios = {}
    for ticker in tickers:
        stock_mean_ret = df_returns[ticker].mean() * 252
        stock_tracking_error = tracking_errors[ticker]
        information_ratios[ticker] = (stock_mean_ret - spy_mean_ret) / stock_tracking_error

    risk_metrics_summary = {"Jensen Alpha": {"Min": 0, "Mean": 0, "Max": 0},
                            "Beta": {"Min": 0, "Mean": 0, "Max": 0},
                            "Treynor Ratio": {"Min": 0, "Mean": 0, "Max": 0},
                            "Tracking Error": {"Min": 0, "Mean": 0, "Max": 0},
                            "Information Ratio": {"Min": 0, "Mean": 0, "Max": 0}}

    for j_a in jensen_alphas.values():
        risk_metrics_summary["Jensen Alpha"]["Min"] = min(jensen_alphas.values())
        risk_metrics_summary["Jensen Alpha"]["Mean"] = sum(jensen_alphas.values()) / 27
        risk_metrics_summary["Jensen Alpha"]["Max"] = max(jensen_alphas.values())

    for beta in stock_betas.values():
        risk_metrics_summary["Beta"]["Min"] = min(stock_betas.values())
        risk_metrics_summary["Beta"]["Mean"] = sum(stock_betas.values()) / 27
        risk_metrics_summary["Beta"]["Max"] = max(stock_betas.values())

    for t_r in treynor_ratios.values():
        risk_metrics_summary["Treynor Ratio"]["Min"] = min(treynor_ratios.values())
        risk_metrics_summary["Treynor Ratio"]["Mean"] = sum(treynor_ratios.values()) / 27
        risk_metrics_summary["Treynor Ratio"]["Max"] = max(treynor_ratios.values())

    for t_e in tracking_errors.values():
        risk_metrics_summary["Tracking Error"]["Min"] = min(tracking_errors.values())
        risk_metrics_summary["Tracking Error"]["Mean"] = sum(tracking_errors.values()) / 27
        risk_metrics_summary["Tracking Error"]["Max"] = max(tracking_errors.values())

    for i_r in information_ratios.values():
        risk_metrics_summary["Information Ratio"]["Min"] = min(information_ratios.values())
        risk_metrics_summary["Information Ratio"]["Mean"] = sum(information_ratios.values()) / 27
        risk_metrics_summary["Information Ratio"]["Max"] = max(information_ratios.values())

    output_df = pd.concat({
        k: pd.DataFrame.from_dict(v, 'index').T for k, v in risk_metrics_summary.items()
    }, axis=0).T

    #  1.4 - Plot of mean return against asset beta

    mean_ret = []
    for ticker in tickers:
        mean_ret.append(df_returns[ticker].mean() * 252)
    betas = list(stock_betas.values())

    m, b = np.polyfit(betas, mean_ret, 1)

    plt.plot(betas, mean_ret, "o")
    plt.plot(betas, m*np.array(betas) + b)
    plt.xlabel("Beta")
    plt.ylabel("Mean Return")
    plt.show()

    return output_df


if __name__ == '__main__':
    ticker_list = ["AAPL", "CSCO", "HON", "KO", "NKE",  "WBA", "AMGN", "CVX", "IBM", "MCD", "PG", "WMT", "AXP",
               "DIS", "INTC", "MMM", "TRV", "BA", "GS", "JNJ", "MRK", "UNH", "CAT", "HD", "JPM", "MSFT",  "VZ"]
    df_stock_prices = load_price_data()
    df_returns = df_stock_prices.copy()
    for ticker in ticker_list:
        df_returns[ticker] = df_returns[ticker].pct_change()
    spy_ret = DataReader("SPY", 'yahoo', "1999-05-01", "2020-12-31")['Adj Close'].rename("SPY")
    spy_ret = spy_ret.pct_change()
    df_returns["SPY"] = spy_ret.values

    abs_summary = abs_performance_summary(ticker_list, df_returns)
    print("1.1 | Absolute Performance Sumamry")
    print(abs_summary)

    print("1.2 | Risk Metrics")
    risk_metrics = get_risk_metrics(ticker_list, df_returns)
    print(risk_metrics)