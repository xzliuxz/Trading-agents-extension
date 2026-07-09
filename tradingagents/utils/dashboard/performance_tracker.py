import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class PerformanceTracker:

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.04)
        self._daily_returns: List[float] = []
        self._daily_values: List[float] = []
        self._dates: List[datetime] = []
        self._benchmark_returns: List[float] = []

    def add_daily_return(self, date: datetime, portfolio_value: float, benchmark_return: float=None) -> None:
        self._dates.append(date)
        self._daily_values.append(portfolio_value)
        if len(self._daily_values) > 1:
            daily_return = portfolio_value / self._daily_values[-2] - 1
        else:
            daily_return = 0.0
        self._daily_returns.append(daily_return)
        if benchmark_return is not None:
            self._benchmark_returns.append(benchmark_return)

    def get_returns_series(self) -> pd.Series:
        if not self._dates:
            return pd.Series()
        return pd.Series(self._daily_returns, index=pd.DatetimeIndex(self._dates))

    def get_performance_metrics(self) -> Dict[str, float]:
        if len(self._daily_returns) < 2:
            return self._empty_metrics()
        returns = np.array(self._daily_returns)
        values = np.array(self._daily_values)
        total_return = values[-1] / values[0] - 1 if values[0] > 0 else 0
        n_days = len(returns)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        annual_vol = np.std(returns) * np.sqrt(252)
        excess_return = annual_return - self.risk_free_rate
        sharpe = excess_return / annual_vol if annual_vol > 0 else 0
        negative_returns = returns[returns < 0]
        downside_vol = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else annual_vol
        sortino = excess_return / downside_vol if downside_vol > 0 else 0
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        max_drawdown = np.max(drawdown)
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
        avg_win = np.mean(returns[returns > 0]) if np.sum(returns > 0) > 0 else 0
        avg_loss = np.mean(returns[returns < 0]) if np.sum(returns < 0) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        return {'total_return': float(total_return), 'annual_return': float(annual_return), 'annual_volatility': float(annual_vol), 'sharpe_ratio': float(sharpe), 'sortino_ratio': float(sortino), 'max_drawdown': float(max_drawdown), 'calmar_ratio': float(calmar), 'win_rate': float(win_rate), 'profit_factor': float(profit_factor), 'avg_daily_return': float(np.mean(returns)), 'best_day': float(np.max(returns)), 'worst_day': float(np.min(returns)), 'trading_days': n_days}

    def get_rolling_metrics(self, window: int=63) -> Dict[str, pd.Series]:
        if len(self._daily_returns) < window:
            return {}
        returns = pd.Series(self._daily_returns, index=pd.DatetimeIndex(self._dates))
        rolling_return = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return.rolling(1).apply(lambda x: (x.iloc[-1] * (252 / window) - self.risk_free_rate) / rolling_vol.iloc[-1] if rolling_vol.iloc[-1] > 0 else 0)
        cumulative = (1 + returns).cumprod()
        rolling_peak = cumulative.rolling(window, min_periods=1).max()
        rolling_drawdown = (cumulative - rolling_peak) / rolling_peak
        return {'rolling_return': rolling_return, 'rolling_volatility': rolling_vol, 'rolling_sharpe': rolling_sharpe, 'rolling_drawdown': rolling_drawdown}

    def get_benchmark_comparison(self) -> Dict[str, Any]:
        if not self._benchmark_returns or len(self._benchmark_returns) != len(self._daily_returns):
            return {}
        port_returns = np.array(self._daily_returns)
        bench_returns = np.array(self._benchmark_returns)
        port_cumulative = np.prod(1 + port_returns) - 1
        bench_cumulative = np.prod(1 + bench_returns) - 1
        excess_returns = port_returns - bench_returns
        total_excess = port_cumulative - bench_cumulative
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        annual_excess = total_excess * (252 / len(port_returns))
        information_ratio = annual_excess / tracking_error if tracking_error > 0 else 0
        cov = np.cov(port_returns, bench_returns)[0, 1]
        var_bench = np.var(bench_returns)
        beta = cov / var_bench if var_bench > 0 else 1
        port_annual = (1 + port_cumulative) ** (252 / len(port_returns)) - 1
        bench_annual = (1 + bench_cumulative) ** (252 / len(bench_returns)) - 1
        alpha = port_annual - (self.risk_free_rate + beta * (bench_annual - self.risk_free_rate))
        return {'portfolio_return': float(port_cumulative), 'benchmark_return': float(bench_cumulative), 'excess_return': float(total_excess), 'tracking_error': float(tracking_error), 'information_ratio': float(information_ratio), 'beta': float(beta), 'alpha': float(alpha)}

    def get_return_attribution(self, weights_history: List[Dict[str, float]], asset_returns: pd.DataFrame) -> Dict[str, Any]:
        if not weights_history or asset_returns.empty:
            return {}
        attribution = {'by_asset': {}, 'by_period': []}
        for i, weights in enumerate(weights_history):
            period_contribution = {}
            for asset, weight in weights.items():
                if asset in asset_returns.columns and i < len(asset_returns):
                    ret = asset_returns.iloc[i].get(asset, 0)
                    contribution = weight * ret
                    period_contribution[asset] = contribution
                    if asset not in attribution['by_asset']:
                        attribution['by_asset'][asset] = 0
                    attribution['by_asset'][asset] += contribution
            attribution['by_period'].append(period_contribution)
        total = sum(attribution['by_asset'].values())
        attribution['total'] = total
        attribution['by_asset'] = dict(sorted(attribution['by_asset'].items(), key=lambda x: x[1], reverse=True))
        return attribution

    def get_drawdown_analysis(self) -> Dict[str, Any]:
        if len(self._daily_values) < 2:
            return {}
        values = np.array(self._daily_values)
        dates = self._dates
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        in_drawdown = drawdown > 0
        drawdown_periods = []
        current_start = None
        for i, (dd, date) in enumerate(zip(drawdown, dates)):
            if dd > 0 and current_start is None:
                current_start = i
            elif dd == 0 and current_start is not None:
                period_dd = drawdown[current_start:i]
                drawdown_periods.append({'start_date': dates[current_start].isoformat(), 'end_date': date.isoformat(), 'max_drawdown': float(max(period_dd)), 'duration_days': i - current_start})
                current_start = None
        if current_start is not None:
            period_dd = drawdown[current_start:]
            drawdown_periods.append({'start_date': dates[current_start].isoformat(), 'end_date': dates[-1].isoformat(), 'max_drawdown': float(max(period_dd)), 'duration_days': len(dates) - current_start, 'ongoing': True})
        sorted_periods = sorted(drawdown_periods, key=lambda x: x['max_drawdown'], reverse=True)
        return {'current_drawdown': float(drawdown[-1]), 'max_drawdown': float(max(drawdown)), 'drawdown_periods': sorted_periods[:10], 'avg_drawdown': float(np.mean(drawdown[drawdown > 0])) if np.any(drawdown > 0) else 0, 'time_in_drawdown_pct': float(np.mean(drawdown > 0.01))}

    def _empty_metrics(self) -> Dict[str, float]:
        return {'total_return': 0.0, 'annual_return': 0.0, 'annual_volatility': 0.0, 'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'max_drawdown': 0.0, 'calmar_ratio': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0, 'avg_daily_return': 0.0, 'best_day': 0.0, 'worst_day': 0.0, 'trading_days': 0}

    def reset(self) -> None:
        self._daily_returns = []
        self._daily_values = []
        self._dates = []
        self._benchmark_returns = []
