import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class RiskDashboard:

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}
        self.var_limit = self.config.get('var_limit', 0.03)
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 0.15)
        self.concentration_limit = self.config.get('concentration_limit', 0.3)
        self.correlation_limit = self.config.get('correlation_limit', 0.8)
        self._last_update: Optional[datetime] = None
        self._risk_snapshots: List[Dict] = []

    def calculate_current_risk(self, weights: Dict[str, float], returns: pd.DataFrame, covariance_matrix: pd.DataFrame=None) -> Dict[str, Any]:
        self._last_update = datetime.now()
        result = {'timestamp': self._last_update.isoformat(), 'var_cvar': self._calculate_var_cvar(weights, returns), 'portfolio_volatility': self._calculate_portfolio_vol(weights, returns, covariance_matrix), 'concentration': self._calculate_concentration(weights), 'correlation_risk': self._calculate_correlation_risk(returns, weights), 'risk_budget_usage': self._calculate_risk_budget_usage(weights, covariance_matrix)}
        result['breaches'] = self._check_breaches(result)
        result['risk_level'] = self._assess_risk_level(result)
        self._risk_snapshots.append(result)
        if len(self._risk_snapshots) > 500:
            self._risk_snapshots = self._risk_snapshots[-500:]
        return result

    def _calculate_var_cvar(self, weights: Dict[str, float], returns: pd.DataFrame, confidence: float=0.95) -> Dict[str, float]:
        available = [k for k in weights if k in returns.columns]
        if not available:
            return {'var_95': 0, 'cvar_95': 0, 'var_99': 0, 'cvar_99': 0}
        w = np.array([weights[k] for k in available])
        w = w / w.sum() if w.sum() > 0 else w
        portfolio_returns = returns[available].values @ w
        var_95 = -np.percentile(portfolio_returns, 5)
        var_99 = -np.percentile(portfolio_returns, 1)
        cvar_95 = -portfolio_returns[portfolio_returns <= -var_95].mean() if np.any(portfolio_returns <= -var_95) else var_95
        cvar_99 = -portfolio_returns[portfolio_returns <= -var_99].mean() if np.any(portfolio_returns <= -var_99) else var_99
        return {'var_95': float(var_95), 'cvar_95': float(cvar_95), 'var_99': float(var_99), 'cvar_99': float(cvar_99)}

    def _calculate_portfolio_vol(self, weights: Dict[str, float], returns: pd.DataFrame, covariance_matrix: pd.DataFrame=None) -> float:
        available = [k for k in weights if k in returns.columns]
        if not available:
            return 0.0
        w = np.array([weights[k] for k in available])
        w = w / w.sum() if w.sum() > 0 else w
        if covariance_matrix is not None:
            cov_available = [k for k in available if k in covariance_matrix.columns]
            if cov_available:
                cov = covariance_matrix.loc[cov_available, cov_available].values
                w_cov = np.array([weights[k] for k in cov_available])
                w_cov = w_cov / w_cov.sum() if w_cov.sum() > 0 else w_cov
                portfolio_vol = np.sqrt(w_cov @ cov @ w_cov) * np.sqrt(252)
                return float(portfolio_vol)
        portfolio_returns = returns[available].values @ w
        return float(np.std(portfolio_returns) * np.sqrt(252))

    def _calculate_concentration(self, weights: Dict[str, float]) -> Dict[str, Any]:
        if not weights:
            return {'max_weight': 0, 'herfindahl': 0, 'effective_n': 0}
        w = np.array(list(weights.values()))
        w = w / w.sum() if w.sum() > 0 else w
        max_weight = np.max(w)
        herfindahl = np.sum(w ** 2)
        effective_n = 1 / herfindahl if herfindahl > 0 else 0
        return {'max_weight': float(max_weight), 'herfindahl': float(herfindahl), 'effective_n': float(effective_n), 'top_3_weight': float(np.sum(np.sort(w)[-3:]))}

    def _calculate_correlation_risk(self, returns: pd.DataFrame, weights: Dict[str, float], lookback: int=63) -> Dict[str, Any]:
        available = [k for k in weights if k in returns.columns]
        if len(available) < 2:
            return {'avg_correlation': 0, 'max_correlation': 0, 'regime': 'UNKNOWN'}
        corr = returns[available].iloc[-lookback:].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        correlations = corr.values[mask]
        avg_corr = np.nanmean(correlations)
        max_corr = np.nanmax(correlations)
        if avg_corr > 0.7:
            regime = 'HIGH_CORRELATION'
        elif avg_corr > 0.5:
            regime = 'ELEVATED'
        else:
            regime = 'NORMAL'
        return {'avg_correlation': float(avg_corr), 'max_correlation': float(max_corr), 'regime': regime}

    def _calculate_risk_budget_usage(self, weights: Dict[str, float], covariance_matrix: pd.DataFrame=None) -> Dict[str, float]:
        if not weights:
            return {'utilization': 0, 'remaining': 1}
        target_vol = 0.12
        if covariance_matrix is not None:
            available = [k for k in weights if k in covariance_matrix.columns]
            if available:
                w = np.array([weights[k] for k in available])
                w = w / w.sum() if w.sum() > 0 else w
                cov = covariance_matrix.loc[available, available].values
                current_vol = np.sqrt(w @ cov @ w) * np.sqrt(252)
                utilization = current_vol / target_vol
                return {'utilization': float(min(1, utilization)), 'remaining': float(max(0, 1 - utilization)), 'current_vol': float(current_vol), 'target_vol': target_vol}
        return {'utilization': 0.5, 'remaining': 0.5}

    def _check_breaches(self, risk_data: Dict) -> List[Dict]:
        breaches = []
        var = risk_data['var_cvar'].get('var_95', 0)
        if var > self.var_limit:
            breaches.append({'type': 'VAR_BREACH', 'limit': self.var_limit, 'actual': var, 'severity': 'HIGH' if var > self.var_limit * 1.5 else 'MEDIUM'})
        max_weight = risk_data['concentration'].get('max_weight', 0)
        if max_weight > self.concentration_limit:
            breaches.append({'type': 'CONCENTRATION_BREACH', 'limit': self.concentration_limit, 'actual': max_weight, 'severity': 'MEDIUM'})
        avg_corr = risk_data['correlation_risk'].get('avg_correlation', 0)
        if avg_corr > self.correlation_limit:
            breaches.append({'type': 'CORRELATION_BREACH', 'limit': self.correlation_limit, 'actual': avg_corr, 'severity': 'MEDIUM'})
        return breaches

    def _assess_risk_level(self, risk_data: Dict) -> str:
        breaches = risk_data.get('breaches', [])
        high_breaches = sum((1 for b in breaches if b['severity'] == 'HIGH'))
        medium_breaches = sum((1 for b in breaches if b['severity'] == 'MEDIUM'))
        if high_breaches >= 2 or (high_breaches >= 1 and medium_breaches >= 2):
            return 'CRITICAL'
        elif high_breaches >= 1 or medium_breaches >= 2:
            return 'HIGH'
        elif medium_breaches >= 1:
            return 'ELEVATED'
        else:
            return 'NORMAL'

    def get_risk_summary(self) -> Dict[str, Any]:
        if not self._risk_snapshots:
            return {'status': 'NO_DATA'}
        latest = self._risk_snapshots[-1]
        return {'timestamp': latest['timestamp'], 'risk_level': latest['risk_level'], 'var_95': latest['var_cvar']['var_95'], 'portfolio_vol': latest['portfolio_volatility'], 'max_weight': latest['concentration']['max_weight'], 'avg_correlation': latest['correlation_risk']['avg_correlation'], 'breaches': latest['breaches'], 'risk_budget_usage': latest['risk_budget_usage']['utilization']}

    def get_risk_history(self, n: int=100) -> List[Dict]:
        return self._risk_snapshots[-n:]

    def generate_risk_report(self) -> str:
        if not self._risk_snapshots:
            return '无风险数据'
        latest = self._risk_snapshots[-1]
        report = f"\n=== 风险报告 ===\n生成时间: {latest['timestamp']}\n\n1. 风险等级: {latest['risk_level']}\n\n2. VaR/CVaR:\n   - VaR (95%): {latest['var_cvar']['var_95']:.2%}\n   - CVaR (95%): {latest['var_cvar']['cvar_95']:.2%}\n   - VaR (99%): {latest['var_cvar']['var_99']:.2%}\n\n3. 组合波动率: {latest['portfolio_volatility']:.2%}\n\n4. 集中度:\n   - 最大单一权重: {latest['concentration']['max_weight']:.2%}\n   - 有效资产数: {latest['concentration']['effective_n']:.1f}\n\n5. 相关性风险:\n   - 平均相关性: {latest['correlation_risk']['avg_correlation']:.2f}\n   - 相关性体制: {latest['correlation_risk']['regime']}\n\n6. 风险预算:\n   - 使用率: {latest['risk_budget_usage']['utilization']:.1%}\n\n7. 限制突破: {len(latest['breaches'])} 项\n"
        for breach in latest['breaches']:
            report += f"   - {breach['type']}: 限制 {breach['limit']:.2%}, 实际 {breach['actual']:.2%}\n"
        return report
