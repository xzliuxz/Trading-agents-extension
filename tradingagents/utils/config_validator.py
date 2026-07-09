from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple
logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    pass

class ConfigValidator:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def validate(self, strict: bool=False) -> 'ConfigValidator':
        self._check_weight_sums()
        self._check_llm_settings()
        self._check_research_decision_weights()
        self._check_backtest_settings()
        self._check_portfolio_settings()
        self._check_macro_settings()
        self._check_risk_budget()
        for w in self.warnings:
            logger.warning('[Config Validation] ⚠️  %s', w)
        for e in self.errors:
            logger.error('[Config Validation] ❌ %s', e)
        if self.errors and strict:
            raise ConfigValidationError(f'Config validation failed with {len(self.errors)} error(s):\n' + '\n'.join((f'  - {e}' for e in self.errors)))
        return self

    def _check_weight_sums(self):
        base = self.config.get('all_weather_base_weights', {})
        if base:
            total = sum(base.values())
            if abs(total - 1.0) > 0.01:
                self.errors.append(f'all_weather_base_weights sum to {total:.3f}, expected 1.0')
        quads = self.config.get('macro_quadrant_allocations', {})
        for qname, weights in quads.items():
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                self.errors.append(f"macro_quadrant_allocations['{qname}'] sums to {total:.3f}, expected 1.0")

    def _check_research_decision_weights(self):
        keys = ['research_decision_fn_weight_trend', 'research_decision_fn_weight_fundamentals', 'research_decision_fn_weight_macro', 'research_decision_fn_weight_event']
        total = sum((self.config.get(k, 0) for k in keys))
        if abs(total - 1.0) > 0.01:
            self.errors.append(f'research_decision_fn_weight_* sum to {total:.3f}, expected 1.0. Keys: {keys}')

    def _check_llm_settings(self):
        provider = self.config.get('llm_provider', '').lower()
        if not provider:
            self.errors.append('llm_provider is empty')
            return
        valid_providers = {'openai', 'anthropic', 'google', 'ollama', 'openrouter', 'deepseek'}
        if provider not in valid_providers:
            self.errors.append(f"llm_provider '{provider}' not in supported set: {valid_providers}")
        for key in ('deep_think_llm', 'quick_think_llm'):
            model = self.config.get(key, '')
            if not model:
                self.warnings.append(f'{key} is empty')

    def _check_backtest_settings(self):
        bt = self.config.get('backtest', {})
        if not bt:
            return
        trend_score_weights = bt.get('trend_score_weights', {})
        if trend_score_weights:
            total = sum(trend_score_weights.values())
            if abs(total - 100) > 5:
                self.warnings.append(f'backtest.trend_score_weights sum to {total}, expected ~100 (tolerance 5)')
        if bt.get('trend_target_vol', 0) <= 0:
            self.warnings.append('backtest.trend_target_vol should be > 0')
        if bt.get('trend_max_position', 0) < bt.get('trend_min_position', 0):
            self.errors.append('backtest.trend_max_position < trend_min_position')

    def _check_portfolio_settings(self):
        pf = self.config.get('portfolio', {})
        if not pf:
            return
        core = pf.get('core_total_weight', 0)
        sat = pf.get('satellite_total_weight', 0)
        if abs(core + sat - 1.0) > 0.01:
            self.errors.append(f'portfolio.core_total_weight ({core}) + satellite_total_weight ({sat}) = {core + sat:.3f}, expected 1.0')

    def _check_macro_settings(self):
        if self.config.get('macro_lookback_days', 0) < 30:
            self.warnings.append('macro_lookback_days < 30 may produce unreliable signals')
        if self.config.get('macro_momentum_window', 0) >= self.config.get('macro_lookback_days', 120):
            self.warnings.append('macro_momentum_window >= macro_lookback_days — the momentum window should be smaller than the data window')

    def _check_risk_budget(self):
        target_vol = self.config.get('research_risk_budget_target_vol_pct', 0)
        if target_vol and target_vol < 5:
            self.warnings.append(f'research_risk_budget_target_vol_pct = {target_vol}% seems very low')
        max_dd = self.config.get('research_risk_budget_max_drawdown_pct', 0)
        if max_dd and max_dd > 0:
            self.warnings.append(f'research_risk_budget_max_drawdown_pct should be negative, got {max_dd}')

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [f'Config Validation: {len(self.errors)} error(s), {len(self.warnings)} warning(s)']
        for e in self.errors:
            lines.append(f'  ❌ {e}')
        for w in self.warnings:
            lines.append(f'  ⚠️  {w}')
        if self.is_valid and (not self.warnings):
            lines.append('  ✅ All checks passed')
        return '\n'.join(lines)

def validate_config(config: Dict[str, Any], strict: bool=False) -> ConfigValidator:
    return ConfigValidator(config).validate(strict=strict)
