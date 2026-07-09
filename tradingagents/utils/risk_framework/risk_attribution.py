import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class RiskAttributor:
    STYLE_FACTORS = ['market', 'size', 'value', 'momentum', 'quality', 'volatility']
    MACRO_FACTORS = ['growth', 'inflation', 'rates', 'credit', 'volatility']

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}

    def calculate_risk_contribution(self, weights: Dict[str, float], covariance_matrix: pd.DataFrame) -> Dict[str, Any]:
        available = [k for k in weights.keys() if k in covariance_matrix.columns]
        if not available:
            return {'portfolio_volatility': 0.0, 'absolute_contribution': {}, 'relative_contribution': {}, 'marginal_contribution': {}}
        w = np.array([weights[k] for k in available])
        cov = covariance_matrix.loc[available, available].values
        portfolio_variance = w @ cov @ w
        portfolio_vol = np.sqrt(portfolio_variance)
        if portfolio_vol < 1e-10:
            return {'portfolio_volatility': 0.0, 'absolute_contribution': {k: 0.0 for k in available}, 'relative_contribution': {k: 0.0 for k in available}, 'marginal_contribution': {k: 0.0 for k in available}}
        marginal = cov @ w / portfolio_vol
        absolute = w * marginal
        relative = absolute / portfolio_vol
        return {'portfolio_volatility': float(portfolio_vol), 'absolute_contribution': {k: float(v) for k, v in zip(available, absolute)}, 'relative_contribution': {k: float(v) for k, v in zip(available, relative)}, 'marginal_contribution': {k: float(v) for k, v in zip(available, marginal)}}

    def decompose_return_attribution(self, portfolio_return: float, benchmark_return: float, portfolio_weights: Dict[str, float], benchmark_weights: Dict[str, float], asset_returns: Dict[str, float]) -> Dict[str, Any]:
        all_assets = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
        allocation_effect = {}
        selection_effect = {}
        interaction_effect = {}
        for asset in all_assets:
            wp = portfolio_weights.get(asset, 0)
            wb = benchmark_weights.get(asset, 0)
            ra = asset_returns.get(asset, 0)
            rb = benchmark_return
            allocation_effect[asset] = (wp - wb) * ra
            selection_effect[asset] = wb * (ra - rb)
            interaction_effect[asset] = (wp - wb) * (ra - rb)
        total_allocation = sum(allocation_effect.values())
        total_selection = sum(selection_effect.values())
        total_interaction = sum(interaction_effect.values())
        total_active = portfolio_return - benchmark_return
        return {'total_active_return': total_active, 'allocation_effect': allocation_effect, 'selection_effect': selection_effect, 'interaction_effect': interaction_effect, 'summary': {'allocation': total_allocation, 'selection': total_selection, 'interaction': total_interaction, 'residual': total_active - total_allocation - total_selection - total_interaction}}

    def calculate_factor_exposure(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, Any]:
        if asset_returns.empty or factor_returns.empty:
            return {'factor_betas': {}, 'factor_contribution': {}, 'specific_risk': 0.0, 'r_squared': 0.0}
        available = [k for k in weights.keys() if k in asset_returns.columns]
        if not available:
            return {'factor_betas': {}, 'factor_contribution': {}, 'specific_risk': 0.0, 'r_squared': 0.0}
        w = np.array([weights[k] for k in available])
        w = w / w.sum()
        portfolio_returns = asset_returns[available] @ w
        common_index = portfolio_returns.index.intersection(factor_returns.index)
        if len(common_index) < 30:
            return {'factor_betas': {}, 'factor_contribution': {}, 'specific_risk': 0.0, 'r_squared': 0.0}
        y = portfolio_returns.loc[common_index].values
        X = factor_returns.loc[common_index].values
        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            betas, residuals, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
            alpha = betas[0]
            factor_betas = betas[1:]
            y_pred = X_with_const @ betas
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            factor_var = np.var(factor_returns.loc[common_index], axis=0)
            factor_contribution = {}
            for i, factor in enumerate(factor_returns.columns):
                if i < len(factor_betas):
                    factor_contribution[factor] = factor_betas[i] ** 2 * factor_var.iloc[i]
            specific_risk = np.sqrt(np.var(y - y_pred)) * np.sqrt(252)
            return {'factor_betas': {f: float(b) for f, b in zip(factor_returns.columns, factor_betas)}, 'alpha': float(alpha), 'factor_contribution': factor_contribution, 'specific_risk': float(specific_risk), 'r_squared': float(r_squared)}
        except Exception as e:
            logger.warning(f'因子分析失败: {e}')
            return {'factor_betas': {}, 'factor_contribution': {}, 'specific_risk': 0.0, 'r_squared': 0.0}

    def calculate_tracking_error(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 20:
            return {'tracking_error': 0.0, 'information_ratio': 0.0, 'active_return': 0.0}
        p_ret = portfolio_returns.loc[common_index]
        b_ret = benchmark_returns.loc[common_index]
        active_returns = p_ret - b_ret
        active_return_annualized = active_returns.mean() * 252
        tracking_error = active_returns.std() * np.sqrt(252)
        if tracking_error > 0:
            information_ratio = active_return_annualized / tracking_error
        else:
            information_ratio = 0.0
        return {'tracking_error': float(tracking_error), 'information_ratio': float(information_ratio), 'active_return': float(active_return_annualized)}

    def decompose_var(self, weights: Dict[str, float], covariance_matrix: pd.DataFrame, confidence_level: float=0.95) -> Dict[str, Any]:
        available = [k for k in weights.keys() if k in covariance_matrix.columns]
        if not available:
            return {'portfolio_var': 0.0, 'component_var': {}, 'marginal_var': {}, 'incremental_var': {}}
        w = np.array([weights[k] for k in available])
        cov = covariance_matrix.loc[available, available].values
        portfolio_vol = np.sqrt(w @ cov @ w)
        from scipy import stats
        z = stats.norm.ppf(confidence_level)
        portfolio_var = z * portfolio_vol
        if portfolio_vol > 0:
            marginal_var = z * (cov @ w) / portfolio_vol
        else:
            marginal_var = np.zeros(len(available))
        component_var = w * marginal_var
        incremental_var = {}
        for i, asset in enumerate(available):
            w_without = w.copy()
            w_without[i] = 0
            if w_without.sum() > 0:
                w_without = w_without / w_without.sum()
            var_without = z * np.sqrt(w_without @ cov @ w_without)
            incremental_var[asset] = float(portfolio_var - var_without)
        return {'portfolio_var': float(portfolio_var), 'component_var': {k: float(v) for k, v in zip(available, component_var)}, 'marginal_var': {k: float(v) for k, v in zip(available, marginal_var)}, 'incremental_var': incremental_var}

    def generate_risk_report(self, weights: Dict[str, float], returns: pd.DataFrame, covariance_matrix: pd.DataFrame, benchmark_weights: Optional[Dict[str, float]]=None) -> Dict[str, Any]:
        report = {'timestamp': pd.Timestamp.now().isoformat(), 'portfolio_weights': weights}
        risk_contrib = self.calculate_risk_contribution(weights, covariance_matrix)
        report['risk_contribution'] = risk_contrib
        var_decomp = self.decompose_var(weights, covariance_matrix)
        report['var_decomposition'] = var_decomp
        if benchmark_weights:
            available_p = [k for k in weights.keys() if k in returns.columns]
            available_b = [k for k in benchmark_weights.keys() if k in returns.columns]
            if available_p and available_b:
                w_p = np.array([weights[k] for k in available_p])
                w_b = np.array([benchmark_weights[k] for k in available_b])
                portfolio_ret = returns[available_p] @ w_p
                benchmark_ret = returns[available_b] @ w_b
                tracking = self.calculate_tracking_error(portfolio_ret, benchmark_ret)
                report['tracking_analysis'] = tracking
        report['summary'] = {'portfolio_volatility': risk_contrib['portfolio_volatility'], 'var_95': var_decomp['portfolio_var'], 'largest_risk_contributor': max(risk_contrib['relative_contribution'].items(), key=lambda x: x[1]) if risk_contrib['relative_contribution'] else ('N/A', 0)}
        return report
