import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
logger = logging.getLogger(__name__)

class StrategicLayer:
    ASSET_CLASSES = ['equities', 'bonds', 'commodities', 'gold', 'cash']
    DEFAULT_ALLOCATION = {'equities': 0.3, 'bonds': 0.4, 'commodities': 0.15, 'gold': 0.1, 'cash': 0.05}
    QUADRANT_NAMES = {'rising_growth_rising_inflation': '增长上行 + 通胀上行', 'rising_growth_falling_inflation': '增长上行 + 通胀下行', 'falling_growth_rising_inflation': '增长下行 + 通胀上行', 'falling_growth_falling_inflation': '增长下行 + 通胀下行', 'unknown': '未确定', 'neutral': '中性'}
    RISK_LEVEL_MAP = {'critical': 'CRITICAL', 'very_high': 'CRITICAL', 'high': 'HIGH', 'medium': 'MODERATE', 'moderate': 'MODERATE', 'low': 'LOW'}

    def __init__(self, config: Dict[str, Any]=None, regime_detector=None, risk_budget_manager=None):
        self.config = config or {}
        self.regime_detector = regime_detector
        self.risk_budget_manager = risk_budget_manager
        self._decision_history: List[Dict] = []
        self._last_decision: Optional[Dict] = None
        self._last_decision_date: Optional[datetime] = None
        self.min_rebalance_interval = timedelta(days=self.config.get('min_strategic_interval', 20))
        self.force_rebalance_threshold = self.config.get('force_rebalance_threshold', 0.15)

    def make_decision(self, as_of_date: str=None, force: bool=False, macro_context: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        current_date = datetime.strptime(as_of_date, '%Y-%m-%d') if as_of_date else datetime.now()
        if not force and self._last_decision_date:
            time_since_last = current_date - self._last_decision_date
            if time_since_last < self.min_rebalance_interval:
                logger.debug(f'跳过战略决策：距上次仅 {time_since_last.days} 天')
                return self._last_decision or self._default_decision()
        logger.info(f"执行战略层决策: {as_of_date or 'today'}")
        regime_source = 'default'
        agent_regime = self._extract_agent_regime(macro_context)
        if agent_regime:
            regime = agent_regime
            regime_source = 'agent_macro'
        elif self.regime_detector:
            regime = self._normalize_regime(self.regime_detector.detect_regime(as_of_date))
            regime_source = 'regime_detector'
        else:
            regime = self._default_regime()
        agent_allocation = self._extract_agent_allocation_hint(macro_context)
        allocation = self._determine_allocation(regime, agent_allocation=agent_allocation)
        if self.risk_budget_manager:
            risk_budget = self.risk_budget_manager.allocate_risk_budget(self.ASSET_CLASSES, allocation, None, method='risk_parity')
        else:
            risk_budget = self._default_risk_budget(allocation)
        themes = self._identify_themes(regime)
        action_required = self._check_action_required(allocation)
        reasoning = self._generate_reasoning(regime, allocation, themes, regime_source=regime_source, macro_guidance=str((macro_context or {}).get('macro_guidance', '') or ''))
        decision = {'timestamp': current_date.isoformat(), 'as_of_date': as_of_date or current_date.strftime('%Y-%m-%d'), 'regime': regime, 'allocation': allocation, 'risk_budget': risk_budget, 'investment_themes': themes, 'action_required': action_required, 'reasoning': reasoning, 'regime_source': regime_source, 'agent_macro_used': regime_source == 'agent_macro', 'layer': 'strategic'}
        self._last_decision = decision
        self._last_decision_date = current_date
        self._decision_history.append(decision)
        if len(self._decision_history) > 50:
            self._decision_history = self._decision_history[-50:]
        return decision

    def _determine_allocation(self, regime: Dict, agent_allocation: Optional[Dict[str, float]]=None) -> Dict[str, float]:
        quadrant = regime.get('quadrant', 'unknown')
        confidence = regime.get('confidence', 0.5)
        base = self.DEFAULT_ALLOCATION.copy()
        quadrant_adjustments = {'rising_growth_rising_inflation': {'equities': 0.1, 'commodities': 0.1, 'bonds': -0.15, 'gold': 0.0, 'cash': -0.05}, 'rising_growth_falling_inflation': {'equities': 0.15, 'bonds': 0.05, 'commodities': -0.1, 'gold': -0.05, 'cash': -0.05}, 'falling_growth_rising_inflation': {'equities': -0.15, 'bonds': -0.05, 'commodities': 0.1, 'gold': 0.15, 'cash': -0.05}, 'falling_growth_falling_inflation': {'equities': -0.1, 'bonds': 0.15, 'commodities': -0.1, 'gold': 0.05, 'cash': 0.0}}
        adjustments = quadrant_adjustments.get(quadrant, {})
        for asset, adj in adjustments.items():
            if asset in base:
                base[asset] += adj * confidence
        risk_level = regime.get('risk_level', 'MODERATE')
        if risk_level == 'CRITICAL':
            scale = 0.7
            base = {k: v * scale for k, v in base.items()}
            base['cash'] = 1 - sum((v for k, v in base.items() if k != 'cash'))
        elif risk_level == 'HIGH':
            scale = 0.85
            for asset in ['equities', 'commodities']:
                base[asset] *= scale
            base['cash'] += (1 - scale) * (base.get('equities', 0) + base.get('commodities', 0))
        base = self._apply_allocation_constraints(base)
        if agent_allocation:
            blend = float(self.config.get('strategic_agent_allocation_blend', 0.65))
            blend = max(0.0, min(1.0, blend))
            merged = {}
            for asset in self.ASSET_CLASSES:
                merged[asset] = (1 - blend) * base.get(asset, 0.0) + blend * agent_allocation.get(asset, 0.0)
            base = self._apply_allocation_constraints(merged)
        return base

    def _apply_allocation_constraints(self, allocation: Dict[str, float]) -> Dict[str, float]:
        min_weight = 0.05
        max_weight = 0.5
        normalized = {asset: max(0.0, float(allocation.get(asset, 0.0))) for asset in self.ASSET_CLASSES}
        total = sum(normalized.values())
        if total <= 0:
            normalized = self.DEFAULT_ALLOCATION.copy()
            total = sum(normalized.values())
        normalized = {k: v / total for k, v in normalized.items()}
        for asset in normalized:
            normalized[asset] = max(min_weight, min(max_weight, normalized[asset]))
        total = sum(normalized.values())
        return {k: v / total for k, v in normalized.items()}

    def _identify_themes(self, regime: Dict) -> List[str]:
        themes = []
        quadrant = regime.get('quadrant', '')
        growth_signal = regime.get('growth_signal', '')
        inflation_signal = regime.get('inflation_signal', '')
        risk_level = regime.get('risk_level', '')
        if 'rising_growth' in quadrant:
            themes.append('GROWTH_TILT')
            themes.append('CYCLICAL_EXPOSURE')
        if 'falling_growth' in quadrant:
            themes.append('DEFENSIVE_TILT')
            themes.append('QUALITY_FOCUS')
        if 'rising_inflation' in quadrant:
            themes.append('INFLATION_HEDGE')
            themes.append('REAL_ASSETS')
        if 'falling_inflation' in quadrant:
            themes.append('DURATION_EXTENSION')
            themes.append('DIVIDEND_YIELD')
        if risk_level in ['HIGH', 'CRITICAL']:
            themes.append('RISK_REDUCTION')
            themes.append('VOLATILITY_TARGETING')
        return themes

    def _check_action_required(self, new_allocation: Dict) -> bool:
        if not self._last_decision:
            return True
        old_allocation = self._last_decision.get('allocation', {})
        total_change = sum((abs(new_allocation.get(asset, 0) - old_allocation.get(asset, 0)) for asset in self.ASSET_CLASSES))
        return total_change > self.force_rebalance_threshold

    def _generate_reasoning(self, regime: Dict, allocation: Dict, themes: List[str], regime_source: str='default', macro_guidance: str='') -> str:
        quadrant_name = regime.get('quadrant_name', '未知')
        confidence = regime.get('confidence', 0)
        risk_level = regime.get('risk_level', '未知')
        reasoning = f"\n战略层决策说明：\n\n1. 宏观环境判断：\n   - 当前象限：{quadrant_name}\n   - 置信度：{confidence:.0%}\n   - 风险等级：{risk_level}\n\n2. 大类资产配置：\n{chr(10).join((f'   - {k}: {v:.1%}' for k, v in allocation.items()))}\n\n3. 投资主题：\n{chr(10).join((f'   - {t}' for t in themes))}\n\n4. 决策依据：\n{regime.get('reasoning', '基于资产动量和宏观指标的综合判断')}\n\n5. 体制来源：\n   - {regime_source}\n"
        if macro_guidance:
            reasoning += f'\n\n6. Macro Agent 指导摘要：\n   - {macro_guidance[:260]}'
        return reasoning.strip()

    def _extract_agent_regime(self, macro_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(macro_context, dict):
            return None
        candidates: List[Dict[str, Any]] = []
        for key in ('macro_analysis', 'regime'):
            value = macro_context.get(key)
            if isinstance(value, dict) and value:
                candidates.append(value)
        candidates.append(macro_context)
        selected: Dict[str, Any] = {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if any((k in candidate for k in ('quadrant', 'growth_signal', 'inflation_signal', 'risk_level'))):
                selected = candidate
                break
        if not selected:
            return None
        merged = dict(selected)
        if 'reasoning' not in merged:
            merged['reasoning'] = macro_context.get('analysis_summary') or macro_context.get('macro_guidance') or selected.get('analysis_summary') or selected.get('reasoning')
        if 'confidence' not in merged:
            merged['confidence'] = float(self.config.get('agent_macro_default_confidence', 0.7))
        return self._normalize_regime(merged)

    def _normalize_regime(self, regime: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(regime, dict):
            return self._default_regime()
        normalized = self._default_regime()
        normalized.update(regime)
        confidence_raw = normalized.get('confidence', 0.5)
        try:
            confidence = float(confidence_raw)
            if confidence > 1.0:
                confidence = confidence / 100.0
            normalized['confidence'] = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            normalized['confidence'] = 0.5
        growth = str(normalized.get('growth_signal', 'neutral') or 'neutral').lower()
        inflation = str(normalized.get('inflation_signal', 'neutral') or 'neutral').lower()
        quadrant = str(normalized.get('quadrant', '') or '').strip().lower()
        if not quadrant and growth and inflation:
            if growth in ('rising', 'falling') and inflation in ('rising', 'falling'):
                quadrant = f'{growth}_growth_{inflation}_inflation'
            else:
                quadrant = 'neutral'
        normalized['quadrant'] = quadrant or 'unknown'
        normalized['quadrant_name'] = str(normalized.get('quadrant_name') or self.QUADRANT_NAMES.get(normalized['quadrant'], self.QUADRANT_NAMES['unknown']))
        risk_level = str(normalized.get('risk_level', 'MODERATE') or 'MODERATE').lower()
        normalized['risk_level'] = self.RISK_LEVEL_MAP.get(risk_level, str(normalized['risk_level']).upper())
        normalized['growth_signal'] = growth
        normalized['inflation_signal'] = inflation
        return normalized

    def _extract_agent_allocation_hint(self, macro_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if not isinstance(macro_context, dict):
            return None
        alloc_payload = None
        for key in ('target_allocations', 'allocation', 'target_allocation'):
            value = macro_context.get(key)
            if isinstance(value, dict) and value:
                alloc_payload = value
                break
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict) and parsed:
                    alloc_payload = parsed
                    break
        if not isinstance(alloc_payload, dict) or not alloc_payload:
            return None
        normalized = {asset: 0.0 for asset in self.ASSET_CLASSES}
        ticker_map = {'equities': {'SPY', 'QQQ', 'VTI', 'IWM', 'EFA', 'VEA', 'EEM', 'VWO'}, 'bonds': {'TLT', 'IEF', 'AGG', 'BND', 'LQD', 'HYG', 'SHY'}, 'commodities': {'DBC', 'GSG', 'USO', 'PDBC'}, 'gold': {'GLD', 'IAU', 'SGOL'}, 'cash': {'CASH', 'BIL', 'SHV'}}
        for raw_key, raw_value in alloc_payload.items():
            try:
                weight = float(raw_value)
            except (TypeError, ValueError):
                continue
            key = str(raw_key).strip().lower()
            if key in normalized:
                normalized[key] += max(0.0, weight)
                continue
            ticker = str(raw_key).strip().upper()
            for asset_class, tickers in ticker_map.items():
                if ticker in tickers:
                    normalized[asset_class] += max(0.0, weight)
                    break
        if sum(normalized.values()) <= 0:
            return None
        return self._apply_allocation_constraints(normalized)

    def _default_decision(self) -> Dict[str, Any]:
        return {'timestamp': datetime.now().isoformat(), 'regime': self._default_regime(), 'allocation': self.DEFAULT_ALLOCATION.copy(), 'risk_budget': self._default_risk_budget(self.DEFAULT_ALLOCATION), 'investment_themes': ['BALANCED'], 'action_required': False, 'reasoning': '无足够数据，维持默认配置', 'regime_source': 'default', 'agent_macro_used': False, 'layer': 'strategic'}

    def _default_regime(self) -> Dict[str, Any]:
        return {'quadrant': 'unknown', 'quadrant_name': '未确定', 'confidence': 0.5, 'growth_signal': 'neutral', 'inflation_signal': 'neutral', 'risk_level': 'MODERATE'}

    def _default_risk_budget(self, allocation: Dict) -> Dict[str, float]:
        base_budget = 0.12
        return {k: base_budget * v for k, v in allocation.items()}

    def get_last_decision(self) -> Optional[Dict]:
        return self._last_decision

    def get_history(self, n: int=10) -> List[Dict]:
        return self._decision_history[-n:]
