import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class DynamicRiskManager:

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}
        self.target_vol = self.config.get('target_volatility', 0.1)
        self.max_vol = self.config.get('max_volatility', 0.2)
        self.min_vol = self.config.get('min_volatility', 0.05)
        self.max_drawdown = self.config.get('max_drawdown', 0.15)
        self.floor_pct = self.config.get('floor_pct', 0.85)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.correlation_lookback = self.config.get('correlation_lookback', 21)
        self._peak_value = 1.0
        self._current_value = 1.0
        self._risk_state = 'NORMAL'
        self._vol_scalar = 1.0

    def update_nav(self, nav: float) -> None:
        self._current_value = nav
        if nav > self._peak_value:
            self._peak_value = nav

    def get_current_drawdown(self) -> float:
        if self._peak_value > 0:
            return 1 - self._current_value / self._peak_value
        return 0.0

    def calculate_vol_scalar(self, realized_vol: float, forecast_vol: Optional[float]=None) -> float:
        vol = forecast_vol if forecast_vol else realized_vol
        if vol <= 0.01:
            return 1.0
        raw_scalar = self.target_vol / vol
        scalar = max(0.3, min(2.0, raw_scalar))
        if vol > self.max_vol:
            scalar = min(scalar, 0.5)
        self._vol_scalar = scalar
        return scalar

    def calculate_cppi_multiplier(self, current_value: float, floor_value: float, max_multiplier: float=3.0) -> float:
        if current_value <= floor_value:
            return 0.0
        cushion = current_value - floor_value
        cushion_pct = cushion / current_value
        multiplier = max_multiplier * cushion_pct
        return min(max_multiplier, max(0.0, multiplier))

    def calculate_drawdown_scalar(self, current_drawdown: float=None) -> float:
        if current_drawdown is None:
            current_drawdown = self.get_current_drawdown()
        if current_drawdown <= 0:
            return 1.0
        if current_drawdown < self.max_drawdown * 0.5:
            scalar = 1.0 - current_drawdown * 0.5
        elif current_drawdown < self.max_drawdown:
            excess = current_drawdown - self.max_drawdown * 0.5
            base_reduction = self.max_drawdown * 0.5 * 0.5
            additional_reduction = excess * 1.5
            scalar = 1.0 - base_reduction - additional_reduction
        else:
            scalar = max(0.2, 0.5 * (self.max_drawdown / current_drawdown))
        return max(0.1, min(1.0, scalar))

    def detect_correlation_regime(self, returns: pd.DataFrame, lookback: int=None) -> Dict[str, Any]:
        if lookback is None:
            lookback = self.correlation_lookback
        if returns.empty or len(returns) < lookback:
            return {'regime': 'UNKNOWN', 'avg_correlation': 0.0, 'correlation_matrix': pd.DataFrame(), 'risk_adjustment': 1.0}
        recent_returns = returns.iloc[-lookback:]
        corr_matrix = recent_returns.corr()
        n = len(corr_matrix)
        if n < 2:
            avg_corr = 0.0
        else:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper_values = corr_matrix.values[mask]
            avg_corr = np.nanmean(upper_values)
        if avg_corr > 0.8:
            regime = 'CRISIS'
            risk_adjustment = 0.5
        elif avg_corr > 0.6:
            regime = 'ELEVATED'
            risk_adjustment = 0.7
        elif avg_corr > 0.4:
            regime = 'NORMAL'
            risk_adjustment = 1.0
        else:
            regime = 'LOW_CORRELATION'
            risk_adjustment = 1.1
        return {'regime': regime, 'avg_correlation': float(avg_corr), 'correlation_matrix': corr_matrix, 'risk_adjustment': risk_adjustment}

    def calculate_risk_on_off(self, vix: float=None, credit_spread: float=None, momentum: float=None, drawdown: float=None) -> Dict[str, Any]:
        score = 0
        components = {}
        if vix is not None:
            if vix > 30:
                vix_score = -1
            elif vix > 20:
                vix_score = -0.5
            elif vix < 15:
                vix_score = 0.5
            else:
                vix_score = 0
            score += vix_score * 0.3
            components['vix'] = {'value': vix, 'score': vix_score}
        if credit_spread is not None:
            if credit_spread > 6:
                credit_score = -1
            elif credit_spread > 4:
                credit_score = -0.5
            elif credit_spread < 3:
                credit_score = 0.5
            else:
                credit_score = 0
            score += credit_score * 0.25
            components['credit_spread'] = {'value': credit_spread, 'score': credit_score}
        if momentum is not None:
            if momentum > 0.05:
                momentum_score = 0.8
            elif momentum > 0:
                momentum_score = 0.3
            elif momentum > -0.05:
                momentum_score = -0.3
            else:
                momentum_score = -0.8
            score += momentum_score * 0.25
            components['momentum'] = {'value': momentum, 'score': momentum_score}
        if drawdown is not None:
            if drawdown > 0.15:
                dd_score = -1
            elif drawdown > 0.1:
                dd_score = -0.5
            elif drawdown > 0.05:
                dd_score = -0.2
            else:
                dd_score = 0.2
            score += dd_score * 0.2
            components['drawdown'] = {'value': drawdown, 'score': dd_score}
        if score > 0.3:
            state = 'RISK_ON'
            position_scalar = 1.0 + score * 0.3
        elif score < -0.3:
            state = 'RISK_OFF'
            position_scalar = max(0.3, 1.0 + score)
        else:
            state = 'NEUTRAL'
            position_scalar = 1.0
        self._risk_state = state
        return {'state': state, 'score': float(score), 'components': components, 'position_scalar': float(position_scalar)}

    def get_combined_position_scalar(self, realized_vol: float, current_drawdown: float, correlation_adjustment: float=1.0, risk_on_off_scalar: float=1.0) -> float:
        vol_scalar = self.calculate_vol_scalar(realized_vol)
        dd_scalar = self.calculate_drawdown_scalar(current_drawdown)
        combined = vol_scalar * dd_scalar * correlation_adjustment * risk_on_off_scalar
        combined = max(0.1, min(2.0, combined))
        logger.debug(f'仓位调整: vol={vol_scalar:.2f}, dd={dd_scalar:.2f}, corr={correlation_adjustment:.2f}, risk={risk_on_off_scalar:.2f} -> {combined:.2f}')
        return combined

    def apply_position_limits(self, target_weights: Dict[str, float], position_scalar: float, max_position: float=0.25, min_position: float=0.02) -> Dict[str, float]:
        adjusted = {}
        for asset, weight in target_weights.items():
            new_weight = weight * position_scalar
            if new_weight < min_position:
                new_weight = 0.0
            elif new_weight > max_position:
                new_weight = max_position
            adjusted[asset] = new_weight
        total = sum(adjusted.values())
        if total > 0 and total != 1.0:
            if total < 1.0:
                adjusted['CASH'] = 1.0 - total
            else:
                adjusted = {k: v / total for k, v in adjusted.items()}
        return adjusted

    def generate_risk_report(self) -> Dict[str, Any]:
        return {'timestamp': datetime.now().isoformat(), 'risk_state': self._risk_state, 'vol_scalar': self._vol_scalar, 'current_drawdown': self.get_current_drawdown(), 'peak_value': self._peak_value, 'current_value': self._current_value, 'parameters': {'target_vol': self.target_vol, 'max_drawdown': self.max_drawdown, 'floor_pct': self.floor_pct}}

    def reset(self, initial_value: float=1.0) -> None:
        self._peak_value = initial_value
        self._current_value = initial_value
        self._risk_state = 'NORMAL'
        self._vol_scalar = 1.0
