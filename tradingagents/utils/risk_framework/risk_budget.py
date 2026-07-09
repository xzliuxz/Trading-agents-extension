import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class RiskBudgetManager:

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}
        self.total_risk_budget = self.config.get('total_risk_budget', 0.12)
        self.min_risk_budget = self.config.get('min_risk_budget', 0.05)
        self.max_risk_budget = self.config.get('max_risk_budget', 0.2)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.04)
        self._current_allocation: Dict[str, float] = {}
        self._current_total_budget = self.total_risk_budget
        self._budget_history: List[Dict] = []

    def set_total_budget(self, budget: float) -> None:
        budget = max(self.min_risk_budget, min(self.max_risk_budget, budget))
        self._current_total_budget = budget
        logger.info(f'风险预算设置为: {budget:.1%}')

    def get_total_budget(self) -> float:
        return self._current_total_budget

    def allocate_risk_budget(self, asset_classes: List[str], target_weights: Dict[str, float], covariance_matrix: pd.DataFrame, method: str='risk_parity') -> Dict[str, float]:
        if method == 'equal':
            n = len(asset_classes)
            allocation = {ac: self._current_total_budget / n for ac in asset_classes}
        elif method == 'proportional':
            total_weight = sum(target_weights.values())
            if total_weight > 0:
                allocation = {ac: self._current_total_budget * target_weights.get(ac, 0) / total_weight for ac in asset_classes}
            else:
                allocation = {ac: self._current_total_budget / len(asset_classes) for ac in asset_classes}
        elif method == 'risk_parity':
            allocation = self._risk_parity_allocation(asset_classes, covariance_matrix)
        else:
            raise ValueError(f'Unknown method: {method}')
        self._current_allocation = allocation
        self._record_allocation(allocation, method)
        return allocation

    def _risk_parity_allocation(self, asset_classes: List[str], covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        if covariance_matrix is None or covariance_matrix.empty:
            return {ac: self._current_total_budget / len(asset_classes) for ac in asset_classes}
        available = [ac for ac in asset_classes if ac in covariance_matrix.columns]
        if not available:
            return {ac: self._current_total_budget / len(asset_classes) for ac in asset_classes}
        cov = covariance_matrix.loc[available, available]
        volatilities = np.sqrt(np.diag(cov))
        inv_vol = 1.0 / np.maximum(volatilities, 0.01)
        weights = inv_vol / inv_vol.sum()
        portfolio_vol = np.sqrt(weights @ cov.values @ weights)
        marginal_risk = cov.values @ weights / portfolio_vol
        risk_contribution = weights * marginal_risk
        total_risk_contrib = risk_contribution.sum()
        if total_risk_contrib > 0:
            normalized_risk = risk_contribution / total_risk_contrib
        else:
            normalized_risk = np.ones(len(available)) / len(available)
        allocation = {}
        for i, ac in enumerate(available):
            allocation[ac] = self._current_total_budget * normalized_risk[i]
        for ac in asset_classes:
            if ac not in allocation:
                allocation[ac] = self._current_total_budget * 0.05
        return allocation

    def calculate_risk_utilization(self, current_weights: Dict[str, float], covariance_matrix: pd.DataFrame) -> Dict[str, Any]:
        if covariance_matrix is None or covariance_matrix.empty:
            available = []
        else:
            available = [ac for ac in current_weights.keys() if ac in covariance_matrix.columns]
        if not available:
            return {'portfolio_volatility': 0.0, 'utilization_rate': 0.0, 'risk_contribution': {}, 'over_budget': [], 'under_budget': []}
        weights = np.array([current_weights.get(ac, 0) for ac in available])
        cov = covariance_matrix.loc[available, available].values
        portfolio_vol = np.sqrt(weights @ cov @ weights)
        if portfolio_vol > 0:
            marginal_risk = cov @ weights / portfolio_vol
            risk_contribution = weights * marginal_risk
        else:
            risk_contribution = np.zeros(len(available))
        risk_contrib_dict = {ac: rc for ac, rc in zip(available, risk_contribution)}
        utilization_rate = portfolio_vol / self._current_total_budget if self._current_total_budget > 0 else 0
        over_budget = []
        under_budget = []
        for ac in available:
            actual_contrib = risk_contrib_dict.get(ac, 0)
            budget = self._current_allocation.get(ac, self._current_total_budget / len(available))
            if actual_contrib > budget * 1.2:
                over_budget.append((ac, actual_contrib, budget))
            elif actual_contrib < budget * 0.5:
                under_budget.append((ac, actual_contrib, budget))
        return {'portfolio_volatility': float(portfolio_vol), 'utilization_rate': float(utilization_rate), 'risk_contribution': risk_contrib_dict, 'over_budget': over_budget, 'under_budget': under_budget}

    def adjust_budget_for_drawdown(self, current_drawdown: float, drawdown_threshold: float=0.1) -> float:
        if current_drawdown <= 0:
            new_budget = self.total_risk_budget
        elif current_drawdown < drawdown_threshold:
            reduction_factor = 1 - current_drawdown / drawdown_threshold * 0.5
            new_budget = self.total_risk_budget * reduction_factor
        else:
            reduction_factor = 0.5 * (1 - min(current_drawdown / 0.25, 1))
            new_budget = self.total_risk_budget * max(0.3, reduction_factor)
        new_budget = max(self.min_risk_budget, min(self.max_risk_budget, new_budget))
        if abs(new_budget - self._current_total_budget) > 0.01:
            logger.info(f'风险预算调整: {self._current_total_budget:.1%} -> {new_budget:.1%} (回撤: {current_drawdown:.1%})')
            self._current_total_budget = new_budget
        return new_budget

    def calculate_optimal_leverage(self, target_return: float, expected_return: float, expected_volatility: float) -> float:
        if expected_return <= self.risk_free_rate:
            return 0.0
        max_leverage_by_risk = self._current_total_budget / expected_volatility
        excess_return_needed = target_return - self.risk_free_rate
        excess_return_available = expected_return - self.risk_free_rate
        if excess_return_available > 0:
            leverage_for_target = excess_return_needed / excess_return_available
        else:
            leverage_for_target = 1.0
        optimal_leverage = min(max_leverage_by_risk, leverage_for_target, 2.0)
        return max(0.0, optimal_leverage)

    def _record_allocation(self, allocation: Dict[str, float], method: str) -> None:
        self._budget_history.append({'timestamp': datetime.now().isoformat(), 'total_budget': self._current_total_budget, 'allocation': allocation.copy(), 'method': method})
        if len(self._budget_history) > 100:
            self._budget_history = self._budget_history[-100:]

    def get_allocation(self) -> Dict[str, float]:
        return self._current_allocation.copy()

    def get_history(self, n: int=10) -> List[Dict]:
        return self._budget_history[-n:]
