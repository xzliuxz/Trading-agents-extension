import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from .yield_curve import YieldCurveAnalyzer
from .credit_spreads import CreditSpreadAnalyzer
from .volatility import VolatilityAnalyzer
from .fund_flows import FundFlowAnalyzer
from .economic import EconomicIndicators
logger = logging.getLogger(__name__)

class RegimeDetector:
    QUADRANTS = {'rising_growth_rising_inflation': {'name': 'Reflation (再通胀)', 'growth': 'rising', 'inflation': 'rising', 'favored': ['equities', 'commodities'], 'unfavored': ['bonds']}, 'rising_growth_falling_inflation': {'name': 'Goldilocks (金发女郎)', 'growth': 'rising', 'inflation': 'falling', 'favored': ['equities', 'bonds'], 'unfavored': []}, 'falling_growth_rising_inflation': {'name': 'Stagflation (滞胀)', 'growth': 'falling', 'inflation': 'rising', 'favored': ['gold', 'commodities'], 'unfavored': ['equities', 'bonds']}, 'falling_growth_falling_inflation': {'name': 'Deflation (通缩)', 'growth': 'falling', 'inflation': 'falling', 'favored': ['bonds', 'gold'], 'unfavored': ['equities', 'commodities']}}

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}
        data_source = self.config.get('macro_data_source', 'fred')
        self.yield_curve = YieldCurveAnalyzer(data_source)
        self.credit_spreads = CreditSpreadAnalyzer(data_source)
        self.volatility = VolatilityAnalyzer()
        self.fund_flows = FundFlowAnalyzer()
        self.economic = EconomicIndicators(data_source)
        self._regime_history: List[Dict] = []
        self._last_detection: Optional[Dict] = None

    def detect_regime(self, as_of_date: str=None) -> Dict[str, Any]:
        logger.info(f"执行宏观体制检测: {as_of_date or 'today'}")
        components = self._collect_all_signals(as_of_date)
        growth_signal, growth_score = self._calculate_growth_signal(components)
        inflation_signal, inflation_score = self._calculate_inflation_signal(components)
        quadrant, scores = self._determine_quadrant(growth_score, inflation_score)
        confidence = self._calculate_confidence(scores, components)
        risk_level = self._assess_risk_level(components)
        recommendations = self._generate_recommendations(quadrant, confidence, components)
        warnings = self._generate_warnings(components, quadrant)
        result = {'quadrant': quadrant, 'quadrant_name': self.QUADRANTS[quadrant]['name'], 'confidence': confidence, 'growth_signal': growth_signal, 'growth_score': growth_score, 'inflation_signal': inflation_signal, 'inflation_score': inflation_score, 'risk_level': risk_level, 'scores': scores, 'components': components, 'recommendations': recommendations, 'warnings': warnings, 'detection_date': as_of_date or datetime.now().strftime('%Y-%m-%d'), 'timestamp': datetime.now().isoformat()}
        self._last_detection = result
        self._regime_history.append({'date': result['detection_date'], 'quadrant': quadrant, 'confidence': confidence})
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-100:]
        logger.info(f"体制检测完成: {result['quadrant_name']} (置信度: {confidence:.1%})")
        return result

    def _collect_all_signals(self, as_of_date: str=None) -> Dict[str, Any]:
        components = {}
        try:
            components['yield_curve'] = self.yield_curve.get_growth_signal(as_of_date)
        except Exception as e:
            logger.warning(f'收益率曲线分析失败: {e}')
            components['yield_curve'] = {'signal': 'NO_DATA', 'confidence': 0}
        try:
            components['credit'] = self.credit_spreads.get_risk_appetite_signal(as_of_date)
        except Exception as e:
            logger.warning(f'信用利差分析失败: {e}')
            components['credit'] = {'signal': 'NO_DATA', 'confidence': 0}
        try:
            components['volatility'] = self.volatility.get_volatility_regime(as_of_date)
        except Exception as e:
            logger.warning(f'波动率分析失败: {e}')
            components['volatility'] = {'regime': 'NORMAL', 'confidence': 0.5}
        try:
            components['fund_flows'] = self.fund_flows.get_flow_signal(as_of_date)
        except Exception as e:
            logger.warning(f'资金流向分析失败: {e}')
            components['fund_flows'] = {'signal': 'NEUTRAL', 'confidence': 0}
        try:
            components['economic_growth'] = self.economic.get_growth_signal(as_of_date)
        except Exception as e:
            logger.warning(f'经济增长分析失败: {e}')
            components['economic_growth'] = {'signal': 'NEUTRAL', 'confidence': 0}
        try:
            components['economic_inflation'] = self.economic.get_inflation_signal(as_of_date)
        except Exception as e:
            logger.warning(f'通胀分析失败: {e}')
            components['economic_inflation'] = {'signal': 'STABLE', 'confidence': 0}
        return components

    def _calculate_growth_signal(self, components: Dict) -> Tuple[str, float]:
        score = 0
        weights_sum = 0
        yc = components.get('yield_curve', {})
        if yc.get('signal') != 'NO_DATA':
            weight = 0.3
            if yc['signal'] == 'BULLISH':
                score += weight * 1
            elif yc['signal'] == 'BEARISH':
                score -= weight * 1
            weights_sum += weight
        credit = components.get('credit', {})
        if credit.get('signal') != 'NO_DATA':
            weight = 0.2
            if credit['signal'] == 'RISK_ON':
                score += weight * 0.8
            elif credit['signal'] == 'RISK_OFF':
                score -= weight * 1
            weights_sum += weight
        econ = components.get('economic_growth', {})
        if econ.get('signal') != 'NO_DATA':
            weight = 0.3
            if econ['signal'] == 'BULLISH':
                score += weight * 1
            elif econ['signal'] == 'BEARISH':
                score -= weight * 1
            weights_sum += weight
        flows = components.get('fund_flows', {})
        if flows.get('signal') != 'NO_DATA':
            weight = 0.2
            if flows['signal'] == 'RISK_ON':
                score += weight * 0.7
            elif flows['signal'] == 'RISK_OFF':
                score -= weight * 0.7
            weights_sum += weight
        if weights_sum > 0:
            score = score / weights_sum
        if score > 0.3:
            signal = 'RISING'
        elif score < -0.3:
            signal = 'FALLING'
        else:
            signal = 'NEUTRAL'
        return (signal, score)

    def _calculate_inflation_signal(self, components: Dict) -> Tuple[str, float]:
        score = 0
        weights_sum = 0
        inflation = components.get('economic_inflation', {})
        if inflation.get('signal') != 'NO_DATA':
            weight = 0.5
            if inflation['signal'] in ['RISING', 'HIGH_INFLATION']:
                score += weight * 1
            elif inflation['signal'] in ['FALLING', 'DEFLATION_RISK']:
                score -= weight * 1
            elif inflation['signal'] == 'ELEVATED':
                score += weight * 0.5
            weights_sum += weight
        yc = components.get('yield_curve', {})
        yc_components = yc.get('components', {})
        real_rate = yc_components.get('real_rate', {})
        if real_rate.get('signal') != 'NO_DATA':
            weight = 0.3
            if real_rate['signal'] in ['VERY_ACCOMMODATIVE', 'ACCOMMODATIVE']:
                score += weight * 0.5
            elif real_rate['signal'] == 'RESTRICTIVE':
                score -= weight * 0.5
            weights_sum += weight
        flows = components.get('fund_flows', {})
        flow_components = flows.get('components', {})
        relative_strength = flow_components.get('relative_strength', {})
        if relative_strength.get('rotation_signal') == 'INFLATION_HEDGE':
            weight = 0.2
            score += weight * 0.8
            weights_sum += weight
        if weights_sum > 0:
            score = score / weights_sum
        if score > 0.25:
            signal = 'RISING'
        elif score < -0.25:
            signal = 'FALLING'
        else:
            signal = 'NEUTRAL'
        return (signal, score)

    def _determine_quadrant(self, growth_score: float, inflation_score: float) -> Tuple[str, Dict[str, float]]:

        def sigmoid(x, scale=5):
            return 1 / (1 + np.exp(-x * scale))
        growth_rising_prob = sigmoid(growth_score)
        inflation_rising_prob = sigmoid(inflation_score)
        scores = {'rising_growth_rising_inflation': growth_rising_prob * inflation_rising_prob, 'rising_growth_falling_inflation': growth_rising_prob * (1 - inflation_rising_prob), 'falling_growth_rising_inflation': (1 - growth_rising_prob) * inflation_rising_prob, 'falling_growth_falling_inflation': (1 - growth_rising_prob) * (1 - inflation_rising_prob)}
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        quadrant = max(scores, key=scores.get)
        return (quadrant, scores)

    def _calculate_confidence(self, scores: Dict[str, float], components: Dict) -> float:
        dominant_score = max(scores.values())
        second_score = sorted(scores.values())[-2]
        score_diff = dominant_score - second_score
        if score_diff < 0.1:
            confidence_penalty = 0.2
        elif score_diff < 0.2:
            confidence_penalty = 0.1
        else:
            confidence_penalty = 0
        no_data_count = sum((1 for c in components.values() if isinstance(c, dict) and c.get('signal') == 'NO_DATA'))
        data_penalty = no_data_count * 0.05
        vol = components.get('volatility', {})
        if vol.get('regime') in ['CRISIS', 'HIGH_VOL']:
            vol_penalty = 0.1
        else:
            vol_penalty = 0
        confidence = dominant_score - confidence_penalty - data_penalty - vol_penalty
        return max(0.3, min(0.95, confidence))

    def _assess_risk_level(self, components: Dict) -> str:
        risk_score = 0
        vol = components.get('volatility', {})
        if vol.get('regime') == 'CRISIS':
            risk_score += 3
        elif vol.get('regime') == 'HIGH_VOL':
            risk_score += 2
        elif vol.get('regime') == 'LOW_VOL':
            risk_score += 0.5
        credit = components.get('credit', {})
        if credit.get('signal') == 'RISK_OFF':
            risk_score += 1.5
        yc = components.get('yield_curve', {})
        yc_components = yc.get('components', {})
        term_spread = yc_components.get('term_spread', {})
        if term_spread.get('is_inverted') and term_spread.get('inversion_days', 0) > 20:
            risk_score += 2
        elif term_spread.get('signal') == 'RECESSION_WARNING':
            risk_score += 1.5
        if risk_score >= 4:
            return 'CRITICAL'
        elif risk_score >= 2.5:
            return 'HIGH'
        elif risk_score >= 1.5:
            return 'ELEVATED'
        elif risk_score >= 0.5:
            return 'MODERATE'
        else:
            return 'LOW'

    def _generate_recommendations(self, quadrant: str, confidence: float, components: Dict) -> Dict[str, Any]:
        quadrant_info = self.QUADRANTS[quadrant]
        base_weights = {'equities': 0.25, 'bonds': 0.25, 'gold': 0.25, 'commodities': 0.25}
        adjustment_strength = confidence * 0.5
        for asset in quadrant_info['favored']:
            if asset in base_weights:
                base_weights[asset] += 0.15 * adjustment_strength
        for asset in quadrant_info['unfavored']:
            if asset in base_weights:
                base_weights[asset] -= 0.1 * adjustment_strength
        risk_level = self._assess_risk_level(components)
        if risk_level == 'CRITICAL':
            base_weights['equities'] *= 0.5
            base_weights['commodities'] *= 0.7
            base_weights['bonds'] += 0.15
            base_weights['gold'] += 0.1
        elif risk_level == 'HIGH':
            base_weights['equities'] *= 0.7
            base_weights['bonds'] += 0.1
        total = sum(base_weights.values())
        target_weights = {k: v / total for k, v in base_weights.items()}
        recommendations_text = self._generate_recommendations_text(quadrant, quadrant_info, confidence, risk_level)
        return {'target_weights': target_weights, 'favored_assets': quadrant_info['favored'], 'unfavored_assets': quadrant_info['unfavored'], 'text': recommendations_text, 'risk_adjustment': risk_level}

    def _generate_recommendations_text(self, quadrant: str, quadrant_info: Dict, confidence: float, risk_level: str) -> str:
        texts = {'rising_growth_rising_inflation': '当前处于再通胀环境：经济增长强劲，通胀压力上升。建议超配股票和大宗商品，低配长期债券。关注周期股和资源类股票。', 'rising_growth_falling_inflation': '当前处于金发女郎环境：经济增长稳健，通胀温和。这是最有利于风险资产的环境。建议超配成长股和优质债券。', 'falling_growth_rising_inflation': '警告：当前处于滞胀环境！经济放缓但通胀高企。这是最具挑战性的宏观环境。建议增持黄金和大宗商品，减持股票和债券。', 'falling_growth_falling_inflation': '当前处于通缩/衰退环境：经济放缓，通胀下降。建议超配长期国债和黄金（避险资产）。对股票保持谨慎，关注防御性板块。'}
        base_text = texts.get(quadrant, '宏观信号不明确，维持均衡配置。')
        if confidence < 0.5:
            confidence_text = f'\n注意：当前信号置信度较低 ({confidence:.0%})，建议谨慎调整。'
        else:
            confidence_text = f'\n信号置信度: {confidence:.0%}。'
        if risk_level in ['CRITICAL', 'HIGH']:
            risk_text = f'\n⚠️ 风险等级: {risk_level}，建议降低整体仓位。'
        else:
            risk_text = ''
        return base_text + confidence_text + risk_text

    def _generate_warnings(self, components: Dict, quadrant: str) -> List[str]:
        warnings = []
        yc = components.get('yield_curve', {})
        yc_components = yc.get('components', {})
        term_spread = yc_components.get('term_spread', {})
        if term_spread.get('signal') == 'RECESSION_WARNING':
            warnings.append('🔴 收益率曲线持续倒挂，历史上往往领先衰退 6-18 个月')
        elif term_spread.get('is_inverted'):
            warnings.append('🟡 收益率曲线出现倒挂，需密切关注')
        credit = components.get('credit', {})
        credit_components = credit.get('components', {})
        ig = credit_components.get('ig_spread', {})
        if ig.get('signal') in ['CRISIS', 'STRESSED']:
            warnings.append('🔴 投资级信用利差显著扩大，金融压力上升')
        hy = credit_components.get('hy_spread', {})
        if hy.get('signal') in ['CRISIS', 'STRESSED']:
            warnings.append('🔴 高收益债利差显著扩大，违约风险上升')
        vol = components.get('volatility', {})
        if vol.get('regime') == 'CRISIS':
            warnings.append('🔴 市场处于恐慌状态，VIX 极高')
        elif vol.get('regime') == 'HIGH_VOL':
            warnings.append('🟡 市场波动率升高，风险偏好下降')
        vol_components = vol.get('components', {})
        vix = vol_components.get('vix', {})
        if vix.get('term_structure') == 'BACKWARDATION':
            warnings.append('🟡 VIX 期限结构倒挂，短期风险高于长期预期')
        if quadrant == 'falling_growth_rising_inflation':
            warnings.append('🔴 滞胀环境是最具挑战性的宏观背景，传统资产可能双杀')
        return warnings

    def get_regime_stability(self, lookback: int=10) -> Dict[str, Any]:
        if len(self._regime_history) < 2:
            return {'stability_score': 0.5, 'dominant_quadrant': self._last_detection.get('quadrant', 'unknown') if self._last_detection else 'unknown', 'transition_probability': 0.5, 'history': self._regime_history}
        recent = self._regime_history[-lookback:]
        quadrant_counts = {}
        for entry in recent:
            q = entry['quadrant']
            quadrant_counts[q] = quadrant_counts.get(q, 0) + 1
        dominant = max(quadrant_counts, key=quadrant_counts.get)
        dominant_pct = quadrant_counts[dominant] / len(recent)
        stability_score = dominant_pct
        transitions = 0
        for i in range(1, len(recent)):
            if recent[i]['quadrant'] != recent[i - 1]['quadrant']:
                transitions += 1
        transition_probability = transitions / max(1, len(recent) - 1)
        return {'stability_score': stability_score, 'dominant_quadrant': dominant, 'transition_probability': transition_probability, 'history': recent}

    def get_last_detection(self) -> Optional[Dict[str, Any]]:
        return self._last_detection
