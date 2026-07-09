import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

class BlackLittermanEngine:

    @staticmethod
    def calculate_bl_weights(prior_weights: Dict[str, float], cov_matrix: pd.DataFrame, views: Dict[str, float], confidences: Dict[str, float]=None, risk_aversion: float=2.5, risk_free_rate: float=0.02) -> Dict[str, float]:
        try:
            from pypfopt import black_litterman, risk_models
            from pypfopt import BlackLittermanModel
            from pypfopt import EfficientFrontier
        except ImportError:
            print('PyPortfolioOpt not installed. Returning prior weights.')
            return prior_weights
        tickers = list(prior_weights.keys())
        if set(cov_matrix.columns) != set(tickers):
            common = list(set(tickers).intersection(set(cov_matrix.columns)))
            if not common:
                return prior_weights
            tickers = common
            prior_weights = {k: prior_weights[k] for k in tickers}
        cov = cov_matrix.loc[tickers, tickers]
        prior_w = pd.Series(prior_weights)[tickers]
        market_prior_rets = black_litterman.market_implied_prior_returns(market_caps=prior_w, risk_aversion=risk_aversion, cov_matrix=cov, risk_free_rate=risk_free_rate)
        if confidences:
            valid_confidences = [confidences.get(k, 0.5) for k in views.keys()]
            bl = BlackLittermanModel(cov, pi=market_prior_rets, absolute_views=views, risk_aversion=risk_aversion, omega='idzorek', view_confidences=list(valid_confidences))
        else:
            bl = BlackLittermanModel(cov, pi=market_prior_rets, absolute_views=views, risk_aversion=risk_aversion)
        ret_bl = bl.bl_returns()
        cov_bl = bl.bl_cov()
        ef = EfficientFrontier(ret_bl, cov_bl)
        try:
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        except:
            weights = ef.min_volatility()
        return dict(weights)

    @staticmethod
    def risk_parity_weights(cov_matrix: pd.DataFrame) -> Dict[str, float]:
        try:
            from pypfopt import EfficientFrontier
            ef = EfficientFrontier(None, cov_matrix)
            weights = ef.min_volatility()
            return dict(weights)
        except Exception as e:
            print(f'Risk parity calculation failed: {e}')
            vols = np.sqrt(np.diag(cov_matrix))
            inv_vols = 1.0 / vols
            weights = inv_vols / np.sum(inv_vols)
            return dict(zip(cov_matrix.columns, weights))

class RiskMetrics:

    @staticmethod
    def calculate_var_cvar(returns_df: pd.DataFrame, weights: Dict[str, float], confidence_level: float=0.95) -> Tuple[float, float]:
        if returns_df.empty or not weights:
            return (0.0, 0.0)
        tickers = list(weights.keys())
        valid_tickers = [t for t in tickers if t in returns_df.columns]
        if not valid_tickers:
            return (0.0, 0.0)
        w_values = np.array([weights[t] for t in valid_tickers])
        if w_values.sum() > 0:
            w_values = w_values / w_values.sum()
        portfolio_returns = returns_df[valid_tickers].dot(w_values)
        if len(portfolio_returns) < 50:
            return (0.0, 0.0)
        var_quantile = 1 - confidence_level
        var = -np.percentile(portfolio_returns, var_quantile * 100)
        loss_exceeding_var = portfolio_returns[portfolio_returns <= -var]
        if len(loss_exceeding_var) > 0:
            cvar = -loss_exceeding_var.mean()
        else:
            cvar = var
        return (max(0.0, var), max(0.0, cvar))

    @staticmethod
    def calculate_drawdown(price_series: pd.Series) -> Tuple[float, float]:
        if price_series.empty:
            return (0.0, 0.0)
        peak = price_series.expanding(min_periods=1).max()
        drawdown = (peak - price_series) / peak
        current_dd = drawdown.iloc[-1]
        max_dd = drawdown.max()
        return (current_dd, max_dd)

def llm_signals_to_bl_views(signals: Dict[str, float], signal_ic: float=0.1, implied_returns: Optional[Dict[str, float]]=None, *, alpha_premium_annual: float=0.08, view_confidence_base: float=0.8, ic_confidence_scaling: bool=True) -> Tuple[Dict[str, float], Dict[str, float]]:
    views: Dict[str, float] = {}
    confidences: Dict[str, float] = {}
    if not signals:
        return (views, confidences)
    sig_values = list(signals.values())
    sig_mean = sum(sig_values) / len(sig_values) if sig_values else 50.0
    sig_std = (sum(((v - sig_mean) ** 2 for v in sig_values)) / max(len(sig_values) - 1, 1)) ** 0.5 if len(sig_values) > 1 else 15.0
    sig_std = max(sig_std, 1.0)
    ic_scale = max(0.0, min(1.0, signal_ic / 0.15))
    for ticker, strength in signals.items():
        implied_ret = (implied_returns or {}).get(ticker, 0.08)
        z = (strength - sig_mean) / sig_std
        if strength >= 75:
            alpha = alpha_premium_annual * ic_scale * max(0, z)
            view = implied_ret + alpha
        elif strength < 45:
            alpha = alpha_premium_annual * ic_scale * abs(min(0, z))
            view = implied_ret - alpha
        else:
            view = implied_ret + alpha_premium_annual * ic_scale * z * 0.2
        views[ticker] = round(view, 6)
        if ic_confidence_scaling:
            conf = view_confidence_base * ic_scale * min(abs(z) / 2.0, 1.0)
        else:
            conf = view_confidence_base * min(abs(z) / 2.0, 1.0)
        confidences[ticker] = round(max(0.01, min(0.99, conf)), 4)
    return (views, confidences)
