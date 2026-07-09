import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
logger = logging.getLogger(__name__)

class TacticalLayer:
    FACTOR_KEYS = ['market', 'size', 'value', 'momentum', 'quality', 'low_vol']
    SECTOR_KEYS = ['XLF', 'XLE', 'XLK', 'XLV', 'XLU', 'XLI', 'XLP', 'XLB']
    BULLISH_WORDS = ('bullish', 'upside', 'beat', 'growth', 'strong', 'outperform', 'upgrade', 'tailwind', 'improving', 'momentum', 'expansion')
    BEARISH_WORDS = ('bearish', 'downside', 'miss', 'weak', 'underperform', 'downgrade', 'headwind', 'deteriorat', 'risk', 'slowdown', 'recession', 'drawdown')

    def __init__(self, config: Dict[str, Any]=None, factor_analyst=None, dynamic_risk_manager=None):
        self.config = config or {}
        self.factor_analyst = factor_analyst
        self.dynamic_risk_manager = dynamic_risk_manager
        self._decision_history: List[Dict] = []
        self._last_decision: Optional[Dict] = None
        self._last_decision_date: Optional[datetime] = None
        self.min_rebalance_interval = timedelta(days=self.config.get('min_tactical_interval', 5))

    def make_decision(self, strategic_decision: Dict, market_data: Dict=None, as_of_date: str=None, force: bool=False, analyst_context: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        current_date = datetime.strptime(as_of_date, '%Y-%m-%d') if as_of_date else datetime.now()
        if not force and self._last_decision_date:
            time_since_last = current_date - self._last_decision_date
            if time_since_last < self.min_rebalance_interval:
                logger.debug(f'跳过战术决策：距上次仅 {time_since_last.days} 天')
                if self._last_decision:
                    return self._last_decision
        logger.info(f"执行战术层决策: {as_of_date or 'today'}")
        strategic_allocation = strategic_decision.get('allocation', {})
        investment_themes = strategic_decision.get('investment_themes', [])
        regime = strategic_decision.get('regime', {})
        agent_hints = self._build_agent_hints(analyst_context)
        factor_tilts = self._determine_factor_tilts(regime, investment_themes, agent_hints=agent_hints)
        asset_selection = self._select_assets_within_class(strategic_allocation, factor_tilts, market_data)
        sector_weights = self._determine_sector_weights(regime, market_data, agent_hints=agent_hints)
        risk_adjustments = self._calculate_risk_adjustments(market_data)
        target_weights = self._calculate_target_weights(strategic_allocation, asset_selection, sector_weights, risk_adjustments)
        reasoning = self._generate_reasoning(factor_tilts, asset_selection, sector_weights, risk_adjustments, agent_summary=agent_hints.get('summary', ''))
        decision = {'timestamp': current_date.isoformat(), 'as_of_date': as_of_date or current_date.strftime('%Y-%m-%d'), 'strategic_reference': strategic_decision.get('timestamp'), 'factor_tilts': factor_tilts, 'asset_selection': asset_selection, 'sector_weights': sector_weights, 'risk_adjustments': risk_adjustments, 'target_weights': target_weights, 'reasoning': reasoning, 'agent_context_used': bool(agent_hints.get('used')), 'agent_context_summary': agent_hints.get('summary', ''), 'agent_sentiment_score': agent_hints.get('sentiment_score'), 'layer': 'tactical'}
        self._last_decision = decision
        self._last_decision_date = current_date
        self._decision_history.append(decision)
        if len(self._decision_history) > 100:
            self._decision_history = self._decision_history[-100:]
        return decision

    def _determine_factor_tilts(self, regime: Dict, themes: List[str], agent_hints: Optional[Dict[str, Any]]=None) -> Dict[str, float]:
        tilts = {'market': 0.0, 'size': 0.0, 'value': 0.0, 'momentum': 0.0, 'quality': 0.0, 'low_vol': 0.0}
        quadrant = regime.get('quadrant', '')
        if 'rising_growth' in quadrant:
            tilts['market'] += 0.1
            tilts['momentum'] += 0.15
            tilts['size'] += 0.05
        if 'falling_growth' in quadrant:
            tilts['market'] -= 0.1
            tilts['quality'] += 0.15
            tilts['low_vol'] += 0.1
        if 'rising_inflation' in quadrant:
            tilts['value'] += 0.1
            tilts['quality'] -= 0.05
        if 'falling_inflation' in quadrant:
            tilts['quality'] += 0.1
            tilts['value'] += 0.05
        if 'GROWTH_TILT' in themes:
            tilts['momentum'] += 0.1
        if 'DEFENSIVE_TILT' in themes:
            tilts['low_vol'] += 0.15
            tilts['quality'] += 0.1
        if 'RISK_REDUCTION' in themes:
            tilts['market'] -= 0.15
            tilts['low_vol'] += 0.2
        if self.factor_analyst:
            recommendation = self.factor_analyst.get_factor_tilt_recommendation(quadrant, tilts)
            if recommendation.get('recommended_tilts'):
                for factor, rec_tilt in recommendation['recommended_tilts'].items():
                    if factor in tilts:
                        tilts[factor] = 0.5 * tilts[factor] + 0.5 * rec_tilt
        hint_tilts = self._normalize_factor_tilts((agent_hints or {}).get('factor_tilts', {}))
        if hint_tilts:
            blend = float(self.config.get('tactical_agent_factor_blend', 0.6))
            blend = max(0.0, min(1.0, blend))
            for factor in self.FACTOR_KEYS:
                tilts[factor] = (1 - blend) * tilts.get(factor, 0.0) + blend * hint_tilts.get(factor, 0.0)
        max_abs = float(self.config.get('tactical_max_abs_tilt', 0.4))
        max_abs = max(0.05, max_abs)
        for factor in list(tilts.keys()):
            tilts[factor] = max(-max_abs, min(max_abs, float(tilts[factor])))
        return tilts

    def _select_assets_within_class(self, strategic_allocation: Dict[str, float], factor_tilts: Dict[str, float], market_data: Dict=None) -> Dict[str, Dict[str, float]]:
        asset_universe = {'equities': {'US_LARGE': ['SPY', 'QQQ', 'VUSA.L'], 'US_SMALL': ['IWM'], 'INTL_DM': ['EFA', 'VEA'], 'INTL_EM': ['EEM', 'VWO'], 'SECTORS': ['XLF', 'XLE', 'XLK', 'XLV', 'XLU']}, 'bonds': {'LONG_TERM': ['TLT'], 'INTERMEDIATE': ['IEF', 'AGG'], 'CORPORATE': ['LQD'], 'HIGH_YIELD': ['HYG']}, 'commodities': {'BROAD': ['DBC', 'GSG'], 'ENERGY': ['USO']}, 'gold': {'PHYSICAL': ['GLD', 'IAU', 'SGLN.L']}, 'cash': {'SHORT_TERM': ['SHV', 'BIL']}}
        selection = {}
        for asset_class, class_weight in strategic_allocation.items():
            if asset_class not in asset_universe:
                continue
            universe = asset_universe[asset_class]
            class_selection = {}
            if asset_class == 'equities':
                if factor_tilts.get('momentum', 0) > 0.1:
                    class_selection['QQQ'] = 0.4
                    class_selection['SPY'] = 0.4
                    class_selection['IWM'] = 0.2
                elif factor_tilts.get('low_vol', 0) > 0.1:
                    class_selection['SPY'] = 0.6
                    class_selection['XLV'] = 0.2
                    class_selection['XLU'] = 0.2
                else:
                    class_selection['SPY'] = 0.5
                    class_selection['QQQ'] = 0.3
                    class_selection['EFA'] = 0.2
            elif asset_class == 'bonds':
                if factor_tilts.get('quality', 0) > 0.1:
                    class_selection['TLT'] = 0.5
                    class_selection['IEF'] = 0.5
                else:
                    class_selection['IEF'] = 0.4
                    class_selection['LQD'] = 0.4
                    class_selection['HYG'] = 0.2
            elif asset_class == 'commodities':
                class_selection['DBC'] = 1.0
            elif asset_class == 'gold':
                class_selection['GLD'] = 1.0
            elif asset_class == 'cash':
                class_selection['SHV'] = 1.0
            selection[asset_class] = class_selection
        return selection

    def _determine_sector_weights(self, regime: Dict, market_data: Dict=None, agent_hints: Optional[Dict[str, Any]]=None) -> Dict[str, float]:
        base_weight = 1.0 / len(self.SECTOR_KEYS)
        weights = {s: base_weight for s in self.SECTOR_KEYS}
        quadrant = regime.get('quadrant', '')
        if 'rising_growth' in quadrant:
            weights['XLF'] *= 1.3
            weights['XLI'] *= 1.2
            weights['XLK'] *= 1.2
            weights['XLU'] *= 0.7
        if 'falling_growth' in quadrant:
            weights['XLV'] *= 1.3
            weights['XLP'] *= 1.2
            weights['XLU'] *= 1.2
            weights['XLF'] *= 0.8
        if 'rising_inflation' in quadrant:
            weights['XLE'] *= 1.4
            weights['XLB'] *= 1.2
        hint_sector = self._normalize_sector_weights((agent_hints or {}).get('sector_weights', {}))
        if hint_sector:
            blend = float(self.config.get('tactical_agent_sector_blend', 0.55))
            blend = max(0.0, min(1.0, blend))
            merged = {}
            for sector in self.SECTOR_KEYS:
                merged[sector] = (1 - blend) * weights.get(sector, 0.0) + blend * hint_sector.get(sector, 0.0)
            weights = merged
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _calculate_risk_adjustments(self, market_data: Dict=None) -> Dict[str, Any]:
        adjustments = {'position_scalar': 1.0, 'volatility_adjustment': 1.0, 'correlation_adjustment': 1.0, 'reason': []}
        if self.dynamic_risk_manager and market_data:
            realized_vol = market_data.get('realized_vol', 0.15)
            vol_scalar = self.dynamic_risk_manager.calculate_vol_scalar(realized_vol)
            adjustments['volatility_adjustment'] = vol_scalar
            drawdown = market_data.get('current_drawdown', 0)
            dd_scalar = self.dynamic_risk_manager.calculate_drawdown_scalar(drawdown)
            adjustments['position_scalar'] = vol_scalar * dd_scalar
            if vol_scalar < 0.9:
                adjustments['reason'].append(f'波动率过高，调整系数 {vol_scalar:.2f}')
            if dd_scalar < 0.9:
                adjustments['reason'].append(f'回撤控制，调整系数 {dd_scalar:.2f}')
        return adjustments

    def _calculate_target_weights(self, strategic_allocation: Dict[str, float], asset_selection: Dict[str, Dict[str, float]], sector_weights: Dict[str, float], risk_adjustments: Dict[str, Any]) -> Dict[str, float]:
        target_weights = {}
        for asset_class, class_weight in strategic_allocation.items():
            if asset_class in asset_selection:
                for ticker, ticker_weight in asset_selection[asset_class].items():
                    final_weight = class_weight * ticker_weight
                    target_weights[ticker] = target_weights.get(ticker, 0) + final_weight
        position_scalar = risk_adjustments.get('position_scalar', 1.0)
        if position_scalar < 1.0:
            for ticker in target_weights:
                target_weights[ticker] *= position_scalar
            cash_addition = (1 - position_scalar) * sum(strategic_allocation.values())
            target_weights['CASH'] = target_weights.get('CASH', 0) + cash_addition
        total = sum(target_weights.values())
        if total > 0:
            target_weights = {k: v / total for k, v in target_weights.items()}
        return target_weights

    def _generate_reasoning(self, factor_tilts: Dict, asset_selection: Dict, sector_weights: Dict, risk_adjustments: Dict, agent_summary: str='') -> str:
        top_factors = sorted(factor_tilts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        factor_str = ', '.join((f'{f}: {v:+.1%}' for f, v in top_factors if abs(v) > 0.05))
        reasoning = f"\n战术层决策说明：\n\n1. 因子倾斜：{factor_str or '无显著倾斜'}\n\n2. 风险调整：\n   - 仓位系数: {risk_adjustments.get('position_scalar', 1.0):.2f}\n   - 调整原因: {', '.join(risk_adjustments.get('reason', [])) or '无'}\n\n3. 类内选择已完成，详见 asset_selection\n"
        if agent_summary:
            reasoning += f'\n\n4. Agent链路输入：{agent_summary}'
        return reasoning.strip()

    def _build_agent_hints(self, analyst_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        hints = {'used': False, 'factor_tilts': {}, 'sector_weights': {}, 'sentiment_score': None, 'summary': ''}
        if not isinstance(analyst_context, dict):
            return hints
        structured_factor = self._normalize_factor_tilts(analyst_context.get('factor_tilts') or analyst_context.get('recommended_tilts') or {})
        structured_sector = self._normalize_sector_weights(analyst_context.get('sector_weights') or analyst_context.get('tactical_sector_weights') or {})
        report_score = self._derive_sentiment_from_reports(analyst_context.get('analyst_reports') or {})
        signal_score = self._derive_signal_bias(analyst_context.get('analyst_signals') or {})
        sentiment_score = analyst_context.get('sentiment_score')
        if isinstance(sentiment_score, (int, float)):
            sentiment = float(sentiment_score)
        elif report_score is not None and signal_score is not None:
            sentiment = 0.6 * report_score + 0.4 * signal_score
        elif report_score is not None:
            sentiment = report_score
        elif signal_score is not None:
            sentiment = signal_score
        else:
            sentiment = None
        if sentiment is not None:
            sentiment = max(-1.0, min(1.0, float(sentiment)))
        if not structured_factor and sentiment is not None:
            structured_factor = self._factor_template_from_sentiment(sentiment)
        if not structured_sector and sentiment is not None:
            structured_sector = self._sector_template_from_sentiment(sentiment)
        hints['factor_tilts'] = structured_factor
        hints['sector_weights'] = structured_sector
        hints['sentiment_score'] = sentiment
        hints['used'] = bool(structured_factor or structured_sector or sentiment is not None)
        summary_parts: List[str] = []
        if structured_factor:
            summary_parts.append('已融合因子倾斜')
        if structured_sector:
            summary_parts.append('已融合板块偏好')
        if sentiment is not None:
            bias = '偏多' if sentiment > 0.12 else '偏空' if sentiment < -0.12 else '中性'
            summary_parts.append(f'分析师情绪={bias}({sentiment:+.2f})')
        hints['summary'] = '；'.join(summary_parts)
        return hints

    def _normalize_factor_tilts(self, payload: Dict[str, Any]) -> Dict[str, float]:
        if not isinstance(payload, dict):
            return {}
        normalized: Dict[str, float] = {}
        for key in self.FACTOR_KEYS:
            if key not in payload:
                continue
            try:
                normalized[key] = float(payload[key])
            except (TypeError, ValueError):
                continue
        return normalized

    def _normalize_sector_weights(self, payload: Dict[str, Any]) -> Dict[str, float]:
        if not isinstance(payload, dict):
            return {}
        normalized = {}
        for key in self.SECTOR_KEYS:
            if key not in payload:
                continue
            try:
                value = float(payload[key])
            except (TypeError, ValueError):
                continue
            normalized[key] = max(0.0, value)
        total = sum(normalized.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in normalized.items()}

    def _derive_sentiment_from_reports(self, reports: Dict[str, Any]) -> Optional[float]:
        if not isinstance(reports, dict) or not reports:
            return None
        text = ' '.join((str(v).lower() for v in reports.values() if v))
        if len(text) < 30:
            return None
        bullish = sum((text.count(token) for token in self.BULLISH_WORDS))
        bearish = sum((text.count(token) for token in self.BEARISH_WORDS))
        total = bullish + bearish
        if total <= 0:
            return None
        return (bullish - bearish) / total

    def _derive_signal_bias(self, signals: Dict[str, Any]) -> Optional[float]:
        if not isinstance(signals, dict) or not signals:
            return None
        mapping = {'BUY': 1.0, 'LONG': 1.0, 'BULLISH': 1.0, 'HOLD': 0.0, 'NEUTRAL': 0.0, 'REDUCE': -0.5, 'SELL': -1.0, 'SHORT': -1.0, 'BEARISH': -1.0}
        values: List[float] = []
        for raw in signals.values():
            if isinstance(raw, dict):
                raw = raw.get('signal') or raw.get('recommendation') or raw.get('stance')
            signal = str(raw or '').strip().upper()
            if signal in mapping:
                values.append(mapping[signal])
        if not values:
            return None
        return sum(values) / len(values)

    def _factor_template_from_sentiment(self, score: float) -> Dict[str, float]:
        if score > 0.12:
            return {'market': 0.12, 'size': 0.04, 'value': 0.02, 'momentum': 0.1, 'quality': -0.03, 'low_vol': -0.05}
        if score < -0.12:
            return {'market': -0.12, 'size': -0.05, 'value': 0.04, 'momentum': -0.08, 'quality': 0.1, 'low_vol': 0.12}
        return {}

    def _sector_template_from_sentiment(self, score: float) -> Dict[str, float]:
        if score > 0.12:
            template = {'XLK': 0.18, 'XLF': 0.16, 'XLI': 0.14, 'XLE': 0.12, 'XLV': 0.1, 'XLU': 0.09, 'XLP': 0.1, 'XLB': 0.11}
        elif score < -0.12:
            template = {'XLK': 0.1, 'XLF': 0.08, 'XLI': 0.1, 'XLE': 0.1, 'XLV': 0.18, 'XLU': 0.17, 'XLP': 0.17, 'XLB': 0.1}
        else:
            return {}
        return self._normalize_sector_weights(template)

    def get_last_decision(self) -> Optional[Dict]:
        return self._last_decision

    def get_history(self, n: int=20) -> List[Dict]:
        return self._decision_history[-n:]
