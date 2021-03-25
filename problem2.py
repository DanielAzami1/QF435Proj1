"""QF 435 - B
Group 7
Project 1
Question 2"""

import pandas as pd
from pandas_datareader.data import DataReader
import math
import pprint
import copy
import matplotlib.pyplot as plt
import numpy as np
import datetime


def load_returns():
    """Load returns data for 27 assets + SPY"""
    return pd.read_csv("returns_data.csv")


def portfolio1(in_sample, out_sample, tickers):
    """Weights in this portfolio are given relative to variance"""
    in_sample = in_sample.drop(columns="SPY")
    out_sample = out_sample.drop(columns="SPY")

    #  Compute weights using in-sample.
    weights = {}
    var_all_stocks = in_sample.var()
    inverse_var = var_all_stocks.apply(lambda var: 1.00 / var)
    sum_inverse_var = inverse_var.sum()

    for ticker in tickers:
        weights[ticker] = inverse_var[ticker] / sum_inverse_var

    weights = pd.Series(weights)

    #  Compute returns using out-sample.
    weighted_return = []
    for row in out_sample.itertuples():
        row = row[2:]
        day_ret = 0
        for index, val in enumerate(row):
            day_ret += val * weights[index]
        weighted_return.append(day_ret)
    out_sample["Weighted Return"] = weighted_return

    return out_sample, weights


def portfolio2(in_sample, out_sample, tickers):
    """Weights in this portfolio are given relative to sharpe"""
    spy_ret = in_sample["SPY"]

    in_sample = in_sample.drop(columns="SPY")
    out_sample = out_sample.drop(columns="SPY")

    #  Calculate Sharpe Ratios
    sharpe_ratios = {}
    for ticker in tickers:
        stock_ret = in_sample[ticker]
        excess_ret = stock_ret.subtract(spy_ret)
        vol = math.sqrt(excess_ret.var())
        sharpe_ratios[ticker] = abs(stock_ret.mean() / vol)

    sharpe_ratios = pd.Series(sharpe_ratios)
    sum_sharpe = sharpe_ratios.sum()

    #  Compute weights using in-sample.
    weights = {}
    for ticker in tickers:
        weights[ticker] = sharpe_ratios[ticker] / sum_sharpe

    weights = pd.Series(weights)

    #  Compute returns using out-sample.
    weighted_return = []
    for row in out_sample.itertuples():
        row = row[2:]
        day_ret = 0
        for index, val in enumerate(row):
            day_ret += val * weights[index]
        weighted_return.append(day_ret)
    out_sample["Weighted Return"] = weighted_return

    return out_sample, weights


def portfolio3(in_sample, out_sample, tickers):
    """Weights in this portfolio are given relative to number of assets"""
    weight = 1/27
    weighted_return = []
    for row in out_sample.itertuples():
        row = row[2:]
        day_ret = 0
        for index, val in enumerate(row):
            day_ret += val * weight
        weighted_return.append(day_ret)
    out_sample["Weighted Return"] = weighted_return

    weights = {}
    for ticker in tickers:
        weights[ticker] = 1/27

    weights = pd.Series(weights)

    return out_sample, weights


if __name__ == '__main__':

    symbols = ["AAPL", "CSCO", "HON", "KO", "NKE",  "WBA", "AMGN", "CVX", "IBM", "MCD", "PG", "WMT", "AXP",
                             "DIS", "INTC", "MMM", "TRV", "BA", "GS", "JNJ", "MRK", "UNH", "CAT", "HD", "JPM", "MSFT",  "VZ"]

    df_returns = load_returns()

    df_returns['Date'] = pd.to_datetime(df_returns['Date'])  # Convert Date to datetime object.

    #  Get in-sample (2017 & 2018).
    df_in = df_returns[(df_returns['Date'] > datetime.datetime(2016, 12, 31))
                       & (df_returns['Date'] < datetime.datetime(2019, 1, 1))]
    #  Get out-sample (2019 & 2020).
    df_out = df_returns[(df_returns['Date'] > datetime.datetime(2018, 12, 31)) &
                        (df_returns['Date'] < datetime.datetime(2021, 1, 1))]
    
    out_sample_1, weights_1 = portfolio1(df_in.copy(), df_out.copy(), symbols)
    out_sample_2, weights_2 = portfolio2(df_in.copy(), df_out.copy(), symbols)
    out_sample_3, weights_3 = portfolio3(df_in.copy(), df_out.copy(), symbols)

    #  1.2
    weights_1 = weights_1.apply(lambda x: x*100)
    weights_2 = weights_2.apply(lambda x: x*100)
    weights_3 = weights_3.apply(lambda x: x*100)
    df = pd.DataFrame({"Portfolio 1": weights_1, "Portfolio 2": weights_2, "Portfolio 3": weights_3})

    #df.to_csv("Weights.csv")