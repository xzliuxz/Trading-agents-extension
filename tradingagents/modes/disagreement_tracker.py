from typing import Any, Dict
from .schemas import LayeredResearchReport, TradingPlan, RiskReview
_CONVICTION_ORDER = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'VERY_HIGH': 3}

def _conviction_delta(level_a: str, level_b: str) -> int:
    return _CONVICTION_ORDER.get(level_a, 1) - _CONVICTION_ORDER.get(level_b, 1)

def track_disagreements(research_report: LayeredResearchReport, trading_plan: TradingPlan, risk_review: RiskReview) -> Dict[str, Any]:
    r2t: Dict[str, Any] = {}
    r2t_score = 0.0
    conviction_delta = _conviction_delta(trading_plan.trading_conviction, research_report.conviction_level)
    r2t['conviction_delta'] = conviction_delta
    if conviction_delta != 0:
        r2t['conviction_change'] = f'{research_report.conviction_level} → {trading_plan.trading_conviction}'
        r2t_score += min(abs(conviction_delta) * 0.2, 0.6)
    if trading_plan.trading_recommendation != research_report.recommendation:
        r2t['recommendation_change'] = f'{research_report.recommendation} → {trading_plan.trading_recommendation}'
        r2t_score += 0.3
    if trading_plan.disagreement_reasons:
        r2t['reasons'] = trading_plan.disagreement_reasons
    r2t['disagreement_score'] = min(r2t_score, 1.0)
    t2r: Dict[str, Any] = {}
    t2r_score = 0.0
    position_delta = risk_review.final_position_pct - trading_plan.position_target_pct
    if abs(position_delta) > 0.01:
        t2r['position_delta'] = round(position_delta, 2)
        t2r['position_change'] = f'{trading_plan.position_target_pct:.1f}% → {risk_review.final_position_pct:.1f}%'
        if trading_plan.position_target_pct != 0:
            reduction_pct = abs(position_delta / trading_plan.position_target_pct)
            t2r_score += min(reduction_pct * 0.5, 0.5)
    if risk_review.final_recommendation != trading_plan.trading_recommendation:
        t2r['recommendation_change'] = f'{trading_plan.trading_recommendation} → {risk_review.final_recommendation}'
        t2r_score += 0.3
    if risk_review.vetoed:
        t2r['vetoed'] = True
        t2r['veto_reason'] = risk_review.veto_reason
        t2r_score = 1.0
    if risk_review.risk_flags:
        t2r['risk_flags'] = risk_review.risk_flags
    if risk_review.reduction_reason:
        t2r['reduction_reason'] = risk_review.reduction_reason
    t2r['disagreement_score'] = min(t2r_score, 1.0)
    total_score = r2t.get('disagreement_score', 0) * 0.4 + t2r.get('disagreement_score', 0) * 0.6
    parts = []
    parts.append(f'Research: {research_report.recommendation} {research_report.conviction_level}')
    trading_part = f'Trading: {trading_plan.trading_recommendation} {trading_plan.trading_conviction} {trading_plan.position_target_pct:.1f}%'
    if trading_plan.disagreement_reasons:
        trading_part += f" ({'; '.join(trading_plan.disagreement_reasons)})"
    parts.append(trading_part)
    status = 'VETOED' if risk_review.vetoed else 'approved'
    risk_part = f'Risk: {status} {risk_review.final_recommendation} {risk_review.final_position_pct:.1f}%'
    if risk_review.reduction_reason:
        risk_part += f' ({risk_review.reduction_reason})'
    if risk_review.veto_reason:
        risk_part += f' ({risk_review.veto_reason})'
    parts.append(risk_part)
    return {'research_to_trading': r2t, 'trading_to_risk': t2r, 'total_disagreement_score': round(total_score, 4), 'audit_trail': ' → '.join(parts)}
