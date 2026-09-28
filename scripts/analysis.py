import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns, rf=0.0682, periods_per_year=52):
    assert len(returns) > 1, "Returns series must have more than one data point."
    periodic_rf = (1 + rf) ** (1 / periods_per_year) - 1
    excess_returns = returns - periodic_rf
    excess_std = excess_returns.std()
    if excess_std == 0:
        return np.nan
    return (
        excess_returns.mean() / excess_std
    ) * np.sqrt(periods_per_year)


def calculate_max_drawdown(returns):
    assert len(returns)>1, "Returns series must have more than one data point."
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown


def calculate_beta(returns, benchmark_returns):
    data = pd.concat(
        [returns.rename("fund"), benchmark_returns.rename("benchmark")],
        axis=1
    ).dropna()

    return data["fund"].cov(data["benchmark"]) / data["benchmark"].var()


def calculate_alpha(returns, benchmark_returns, rf=0.0682, periods_per_year=52):
    assert len(returns) > 1
    assert len(benchmark_returns) > 1
    data = pd.concat(
        [returns.rename("fund"), benchmark_returns.rename("benchmark")],
        axis=1
    ).dropna()
    fund = data["fund"]
    benchmark = data["benchmark"]
    beta = calculate_beta(fund, benchmark)
    if np.isnan(beta):
        return np.nan
    periodic_rf = (1 + rf) ** (1 / periods_per_year) - 1
    alpha_periodic = (
        fund.mean()
        - periodic_rf
        - beta * (benchmark.mean() - periodic_rf)
    )
    return alpha_periodic * periods_per_year


def calculate_cagr(returns, periods_per_year=52):
    assert len(returns)>1, "Returns series must have more than one data point."
    
    compounded = (1 + returns).prod()
    cagr = compounded**(periods_per_year / len(returns)) - 1
    return cagr


def calculate_information_ratio(returns, benchmark_returns, periods_per_year=52):
    data = pd.concat(
        [
            returns.rename("fund"),
            benchmark_returns.rename("benchmark")
        ],
        axis=1
    ).dropna()
    active_returns = data["fund"] - data["benchmark"]
    tracking_error = active_returns.std()
    if tracking_error == 0:
        return np.nan
    return (
        active_returns.mean() / tracking_error
    ) * np.sqrt(periods_per_year)


def calculate_metrics_df(returns_df, benchmark_returns, rf=0.0682, periods_per_year=52):
    metrics = {}

    for fund in returns_df.columns:
        returns = returns_df[fund].dropna()

        metrics[fund] = {
            "CAGR": calculate_cagr(returns, periods_per_year=periods_per_year),
            "Volatility": returns.std() * np.sqrt(periods_per_year),
            "Alpha": calculate_alpha(returns, benchmark_returns, rf=rf, periods_per_year=periods_per_year),
            "Beta": calculate_beta(returns, benchmark_returns),
            "Sharpe Ratio": calculate_sharpe_ratio(returns, rf=rf, periods_per_year=periods_per_year),
            "Max Drawdown": calculate_max_drawdown(returns),
            "Information Ratio": calculate_information_ratio(returns, benchmark_returns, periods_per_year=periods_per_year),
        }

    return pd.DataFrame.from_dict(metrics, orient="index")