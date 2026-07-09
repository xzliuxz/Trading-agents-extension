import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    num_trades: int
    equity_curve: pd.Series
    drawdown_series: pd.Series
    trade_log: pd.DataFrame
    metadata: Dict[str, Any]

class EnhancedBacktestEngine:

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', 100000)
        self.commission_rate = self.config.get('commission_rate', 0.001)
        self.slippage_rate = self.config.get('slippage_rate', 0.001)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.04)

    def run_walk_forward(self, price_data: pd.DataFrame, signal_generator: Callable[[pd.DataFrame, Dict], pd.DataFrame], train_period: int=252, test_period: int=63, refit_frequency: int=21, signal_params: Dict=None) -> Dict[str, Any]:
        if price_data.empty:
            raise ValueError('Empty price data')
        signal_params = signal_params or {}
        total_days = len(price_data)
        min_start = train_period
        if total_days < min_start + test_period:
            raise ValueError(f'Insufficient data: need at least {min_start + test_period} days')
        period_results = []
        all_signals = pd.DataFrame(index=price_data.index, columns=price_data.columns)
        all_signals = all_signals.fillna(0)
        current_start = min_start
        while current_start + test_period <= total_days:
            train_start = max(0, current_start - train_period)
            train_end = current_start
            train_data = price_data.iloc[train_start:train_end]
            test_start = current_start
            test_end = min(current_start + test_period, total_days)
            test_data = price_data.iloc[test_start:test_end]
            logger.debug(f'Walk-forward window: train={train_start}-{train_end}, test={test_start}-{test_end}')
            try:
                signals = signal_generator(train_data, signal_params)
                for col in signals.columns:
                    if col in all_signals.columns:
                        if not signals.empty:
                            last_signal = signals.iloc[-1]
                            all_signals.loc[test_data.index, col] = last_signal[col] if col in last_signal.index else 0
                test_returns = test_data.pct_change().fillna(0)
                test_signals = all_signals.loc[test_data.index]
                portfolio_returns = (test_signals.shift(1) * test_returns).sum(axis=1).fillna(0)
                period_result = self._calculate_period_metrics(portfolio_returns, test_data.index)
                period_results.append(period_result)
            except Exception as e:
                logger.warning(f'Walk-forward period failed: {e}')
            current_start += refit_frequency
        if not period_results:
            raise ValueError('No valid walk-forward periods')
        aggregate_result = self._aggregate_walk_forward_results(period_results, price_data, all_signals)
        train_returns = [r['train_return'] for r in period_results if 'train_return' in r]
        test_returns = [r['return'] for r in period_results]
        if train_returns and np.mean(train_returns) != 0:
            wfe = np.mean(test_returns) / np.mean(train_returns)
        else:
            wfe = 1.0
        if np.mean(test_returns) != 0:
            stability = 1 - np.std(test_returns) / abs(np.mean(test_returns))
        else:
            stability = 0
        return {'aggregate_result': aggregate_result, 'period_results': period_results, 'walk_forward_efficiency': float(np.clip(wfe, 0, 2)), 'stability_ratio': float(np.clip(stability, 0, 1)), 'num_periods': len(period_results)}

    def run_monte_carlo(self, base_returns: pd.Series, n_simulations: int=1000, n_years: int=5, method: str='bootstrap') -> Dict[str, Any]:
        if base_returns.empty:
            raise ValueError('Empty returns data')
        n_days = n_years * 252
        np.random.seed(42)
        if method == 'bootstrap':
            simulated_returns = self._bootstrap_simulation(base_returns, n_simulations, n_days)
        elif method == 'parametric':
            simulated_returns = self._parametric_simulation(base_returns, n_simulations, n_days)
        elif method == 'block_bootstrap':
            simulated_returns = self._block_bootstrap_simulation(base_returns, n_simulations, n_days)
        else:
            raise ValueError(f'Unknown method: {method}')
        terminal_wealth = np.cumprod(1 + simulated_returns, axis=1)[:, -1]
        wealth_paths = np.cumprod(1 + simulated_returns, axis=1)
        max_drawdowns = self._calculate_path_drawdowns(wealth_paths)
        percentiles = {'1%': float(np.percentile(terminal_wealth, 1)), '5%': float(np.percentile(terminal_wealth, 5)), '10%': float(np.percentile(terminal_wealth, 10)), '25%': float(np.percentile(terminal_wealth, 25)), '50%': float(np.percentile(terminal_wealth, 50)), '75%': float(np.percentile(terminal_wealth, 75)), '90%': float(np.percentile(terminal_wealth, 90)), '95%': float(np.percentile(terminal_wealth, 95)), '99%': float(np.percentile(terminal_wealth, 99))}
        prob_loss = (terminal_wealth < 1.0).mean()
        prob_targets = {'10%_annual': (terminal_wealth > 1.1 ** n_years).mean(), '15%_annual': (terminal_wealth > 1.15 ** n_years).mean(), '20%_annual': (terminal_wealth > 1.2 ** n_years).mean(), '2x': (terminal_wealth > 2.0).mean(), '3x': (terminal_wealth > 3.0).mean()}
        dd_percentiles = {'50%': float(np.percentile(max_drawdowns, 50)), '90%': float(np.percentile(max_drawdowns, 90)), '95%': float(np.percentile(max_drawdowns, 95)), '99%': float(np.percentile(max_drawdowns, 99))}
        return {'terminal_wealth_distribution': terminal_wealth, 'percentiles': percentiles, 'probability_of_loss': float(prob_loss), 'probability_of_target': prob_targets, 'max_drawdown_distribution': max_drawdowns, 'max_drawdown_percentiles': dd_percentiles, 'n_simulations': n_simulations, 'n_years': n_years, 'method': method}

    def _bootstrap_simulation(self, returns: pd.Series, n_sims: int, n_days: int) -> np.ndarray:
        returns_array = returns.values
        indices = np.random.randint(0, len(returns_array), size=(n_sims, n_days))
        return returns_array[indices]

    def _parametric_simulation(self, returns: pd.Series, n_sims: int, n_days: int) -> np.ndarray:
        mu = returns.mean()
        sigma = returns.std()
        return np.random.normal(mu, sigma, size=(n_sims, n_days))

    def _block_bootstrap_simulation(self, returns: pd.Series, n_sims: int, n_days: int, block_size: int=21) -> np.ndarray:
        returns_array = returns.values
        n_blocks = n_days // block_size + 1
        result = np.zeros((n_sims, n_days))
        for i in range(n_sims):
            blocks = []
            for _ in range(n_blocks):
                start = np.random.randint(0, len(returns_array) - block_size)
                blocks.append(returns_array[start:start + block_size])
            path = np.concatenate(blocks)[:n_days]
            result[i] = path
        return result

    def _calculate_path_drawdowns(self, wealth_paths: np.ndarray) -> np.ndarray:
        n_sims = wealth_paths.shape[0]
        max_drawdowns = np.zeros(n_sims)
        for i in range(n_sims):
            path = wealth_paths[i]
            peak = np.maximum.accumulate(path)
            drawdown = (peak - path) / peak
            max_drawdowns[i] = drawdown.max()
        return max_drawdowns

    def calculate_strategy_capacity(self, signals: pd.DataFrame, volumes: pd.DataFrame, prices: pd.DataFrame, max_participation_rate: float=0.05, target_execution_days: int=3) -> Dict[str, Any]:
        if signals.empty or volumes.empty or prices.empty:
            return {'estimated_capacity': 0, 'capacity_by_asset': {}, 'limiting_factor': 'NO_DATA', 'recommendations': []}
        daily_turnover = (volumes * prices).mean()
        signal_turnover = signals.diff().abs().mean()
        capacity_by_asset = {}
        recommendations = []
        for asset in signals.columns:
            if asset in daily_turnover.index:
                asset_turnover = daily_turnover[asset]
                max_position = asset_turnover * max_participation_rate * target_execution_days
                asset_signal_turnover = signal_turnover.get(asset, 0.1)
                if asset_signal_turnover > 0.5:
                    max_position *= 0.5
                    recommendations.append(f'{asset}: 高换手，容量受限')
                capacity_by_asset[asset] = float(max_position)
        if capacity_by_asset:
            avg_weight = 1.0 / len(capacity_by_asset)
            estimated_capacity = min((cap / avg_weight for cap in capacity_by_asset.values() if cap > 0))
        else:
            estimated_capacity = 0
        if capacity_by_asset:
            limiting_asset = min(capacity_by_asset, key=capacity_by_asset.get)
            limiting_factor = f'Liquidity constraint: {limiting_asset}'
        else:
            limiting_factor = 'NO_DATA'
        return {'estimated_capacity': float(estimated_capacity), 'capacity_by_asset': capacity_by_asset, 'limiting_factor': limiting_factor, 'recommendations': recommendations, 'assumptions': {'max_participation_rate': max_participation_rate, 'target_execution_days': target_execution_days}}

    def run_sensitivity_analysis(self, price_data: pd.DataFrame, signal_generator: Callable, base_params: Dict, param_ranges: Dict[str, List]) -> Dict[str, Any]:
        results = {'sensitivity': {}}
        base_signals = signal_generator(price_data, base_params)
        base_result = self._run_simple_backtest(price_data, base_signals)
        results['base_result'] = base_result
        for param_name, values in param_ranges.items():
            results['sensitivity'][param_name] = {}
            for value in values:
                test_params = base_params.copy()
                test_params[param_name] = value
                try:
                    signals = signal_generator(price_data, test_params)
                    result = self._run_simple_backtest(price_data, signals)
                    results['sensitivity'][param_name][value] = result
                except Exception as e:
                    logger.warning(f'Sensitivity test failed for {param_name}={value}: {e}')
        results['robust_params'] = self._identify_robust_params(results['sensitivity'], base_result)
        return results

    def _run_simple_backtest(self, prices: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, float]:
        returns = prices.pct_change().fillna(0)
        common_cols = signals.columns.intersection(returns.columns)
        common_idx = signals.index.intersection(returns.index)
        if len(common_cols) == 0 or len(common_idx) == 0:
            return {'sharpe': 0, 'return': 0, 'volatility': 0, 'max_dd': 0}
        signals_aligned = signals.loc[common_idx, common_cols]
        returns_aligned = returns.loc[common_idx, common_cols]
        portfolio_returns = (signals_aligned.shift(1) * returns_aligned).sum(axis=1).fillna(0)
        turnover = signals_aligned.diff().abs().sum(axis=1)
        costs = turnover * (self.commission_rate + self.slippage_rate)
        net_returns = portfolio_returns - costs
        total_return = (1 + net_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(net_returns)) - 1 if len(net_returns) > 0 else 0
        annual_vol = net_returns.std() * np.sqrt(252)
        sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        cumulative = (1 + net_returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        return {'sharpe': float(sharpe), 'return': float(annual_return), 'volatility': float(annual_vol), 'max_dd': float(max_dd), 'total_return': float(total_return)}

    def _calculate_period_metrics(self, returns: pd.Series, dates: pd.Index) -> Dict[str, float]:
        if returns.empty or len(returns) < 5:
            return {'return': 0, 'sharpe': 0, 'volatility': 0}
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        return {'return': float(total_return), 'annual_return': float(annual_return), 'volatility': float(annual_vol), 'sharpe': float(sharpe), 'start_date': str(dates[0]), 'end_date': str(dates[-1]), 'n_days': len(returns)}

    def _aggregate_walk_forward_results(self, period_results: List[Dict], prices: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        all_returns = []
        for pr in period_results:
            pass
        returns = [r['return'] for r in period_results]
        sharpes = [r['sharpe'] for r in period_results]
        avg_return = np.mean(returns)
        avg_sharpe = np.mean(sharpes)
        equity = [1.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        equity_series = pd.Series(equity)
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        return BacktestResult(total_return=equity[-1] - 1, annual_return=avg_return, annual_volatility=np.std(returns) * np.sqrt(len(returns)), sharpe_ratio=avg_sharpe, sortino_ratio=0, max_drawdown=drawdown.min(), calmar_ratio=avg_return / abs(drawdown.min()) if drawdown.min() != 0 else 0, win_rate=sum((1 for r in returns if r > 0)) / len(returns) if returns else 0, profit_factor=0, avg_trade_return=avg_return, num_trades=len(period_results), equity_curve=equity_series, drawdown_series=drawdown, trade_log=pd.DataFrame(period_results), metadata={'method': 'walk_forward'})

    def _identify_robust_params(self, sensitivity: Dict, base_result: Dict) -> Dict[str, Any]:
        robust_params = {}
        base_sharpe = base_result.get('sharpe', 0)
        for param_name, values_results in sensitivity.items():
            robust_values = []
            for value, result in values_results.items():
                if result.get('sharpe', 0) > base_sharpe * 0.8:
                    robust_values.append(value)
            if robust_values:
                robust_params[param_name] = {'robust_range': [min(robust_values), max(robust_values)], 'best_value': max(values_results.items(), key=lambda x: x[1].get('sharpe', 0))[0]}
        return robust_params
