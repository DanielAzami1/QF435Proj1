"""QF 435 - B
Group 7
Project 1
Question 4"""

import pandas as pd
from pandas_datareader.data import DataReader
import math
import pprint
import copy
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns


def load_returns():
    """Load returns data for 27 assets + SPY"""
    return pd.read_csv("returns_data.csv")


def load_portfolio_returns():
    """Load weighted returns data for the 3 portfolios"""
    return pd.read_csv("PortfolioWeightedReturns.csv")


def calibrate(df):
    calibrate_df = {"Mean Return": {"Portfolio 1": 0,
                                    "Portfolio 2": 0,
                                    "Portfolio 3": 0},
                    "Volatility": {"Portfolio 1": 0,
                                   "Portfolio 2": 0,
                                   "Portfolio 3": 0}}
    # Mean Return
    calibrate_df["Mean Return"]["Portfolio 1"] = df['Portfolio 1'].mean() * 252
    calibrate_df["Mean Return"]["Portfolio 2"] = df['Portfolio 2'].mean() * 252
    calibrate_df["Mean Return"]["Portfolio 3"] = df['Portfolio 3'].mean() * 252

    # Vol
    calibrate_df["Volatility"]["Portfolio 1"] = math.sqrt(df['Portfolio 1'].var() * 252)
    calibrate_df["Volatility"]["Portfolio 2"] = math.sqrt(df['Portfolio 2'].var() * 252)
    calibrate_df["Volatility"]["Portfolio 3"] = math.sqrt(df['Portfolio 3'].var() * 252)

    return calibrate_df


def simulate_path(mu, sigma):
    T_MAX = 252
    F_0 = 100
    t = 0
    dt = 1 / 252
    simulated_prices = []

    def get_sim_price(F_t, mu, sigma, dt):
        dR = np.random.normal(dt * (mu - 0.5 * (sigma ** 2)),
                              sigma * math.sqrt(dt))
        return F_t * math.exp(dR)

    while t < T_MAX:
        new_price = get_sim_price(F_0, mu, sigma, dt)
        simulated_prices.append(new_price)
        F_0 = new_price
        t += 1

    return simulated_prices


def gen_plot(price_path, portfolio_num):
    dt = list(range(1, 253))
    plt.plot(dt, price_path)
    plt.xlabel("Day (t)")
    plt.ylabel(f"Simulated Price | Portfolio {portfolio_num}")
    plt.show()


def gen_hist(price_path, portfolio_num):
    # plt.hist(price_path)
    # plt.hist(price_path, 40, density=True, facecolor='g', alpha=0.75)
    # plt.xlabel(f"Price | Portfolio {portfolio_num}")
    # plt.ylabel("Frequency")
    # plt.show()
    sns.displot(price_path)
    plt.show()


def task_two(df_out):
    pass


def get_beta_portfolio(df_out, portfolio_num):
    spy_var = df_out["SPY"].var()
    cov_matrix = df_out.cov()
    if portfolio_num == 1:
        cov = cov_matrix['Portfolio 1']['SPY']
    elif portfolio_num == 2:
        cov = cov_matrix['Portfolio 2']['SPY']
    else:
        cov = cov_matrix['Portfolio 3']['SPY']
    return cov / spy_var


if __name__ == '__main__':

    df_returns = load_portfolio_returns()
    spy_ret = list(DataReader("SPY", 'yahoo', "2019-01-02", "2020-12-31")["Adj Close"].pct_change().rename("SPY"))
    df_returns["SPY"] = spy_ret

    portfolio_1_beta = get_beta_portfolio(df_returns, 1)
    portfolio_2_beta = get_beta_portfolio(df_returns, 2)
    portfolio_3_beta = get_beta_portfolio(df_returns, 3)

    spy_vol = math.sqrt(df_returns["SPY"].var()) * math.sqrt(252)

    calibrated_df = calibrate(df_returns)

    mu_one = calibrated_df['Mean Return']['Portfolio 1']
    sigma_one = calibrated_df['Volatility']['Portfolio 1']
    mu_one += 0.5 * (sigma_one ** 2)

    mu_two = calibrated_df['Mean Return']['Portfolio 2']
    sigma_two = calibrated_df['Volatility']['Portfolio 2']
    mu_two += 0.5 * (sigma_two ** 2)

    mu_three = calibrated_df['Mean Return']['Portfolio 3']
    sigma_three = calibrated_df['Volatility']['Portfolio 3']
    mu_three += 0.5 * (sigma_three ** 2)

    price_path_1 = simulate_path(mu_one, sigma_one + (portfolio_1_beta * 0.275))

    price_path_2 = simulate_path(mu_two, sigma_two + (portfolio_2_beta * 0.275))

    price_path_3 = simulate_path(mu_three, sigma_three + (portfolio_3_beta * 0.275))

    # gen_plot(price_path_3, 3)

    # calibrated_df = calibrate(df_returns)
    #
    # price_path_1 = simulate_path(calibrated_df['Mean Return']['Portfolio 1'],
    #                              calibrated_df['Volatility']['Portfolio 1'])
    # price_path_2 = simulate_path(calibrated_df['Mean Return']['Portfolio 2'],
    #                              calibrated_df['Volatility']['Portfolio 2'])
    # price_path_3 = simulate_path(calibrated_df['Mean Return']['Portfolio 3'],
    #                              calibrated_df['Volatility']['Portfolio 3'])
    #
    # portfolio = []
    # for i in range(0, 1000):
    #     price_path = simulate_path(calibrated_df['Mean Return']['Portfolio 1'],
    #                                calibrated_df['Volatility']['Portfolio 1'])
    #     portfolio.append(sum(price_path) / len(price_path))
    #
    #
    # # Portfolio 1
    mean = sum(price_path_1) / len(price_path_1)
    pct = np.percentile(price_path_1, 0.05)
    print(mean - pct)

    # Portfolio 2
    mean = sum(price_path_2) / len(price_path_2)
    pct = np.percentile(price_path_2, 0.05)
    print(mean - pct)

    # Portfolio 3
    mean = sum(price_path_3) / len(price_path_3)
    pct = np.percentile(price_path_3, 0.05)
    print(mean - pct)


    # one_year_price = 0
    # for i in range(0, 1000):
    #     sim_price = simulate_path(calibrated_df['Mean Return']['Portfolio 3'],
    #                                     calibrated_df['Volatility']['Portfolio 3'])[-1]
    #     one_year_price += sim_price
    # print(one_year_price / 1000)

