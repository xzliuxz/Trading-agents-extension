from datetime import datetime
from typing import Any, Dict, Optional
from tradingagents.default_config import DEFAULT_CONFIG
from .schemas import LayeredResearchReport, TradingPlan, RiskReview, LayerAudit
from .disagreement_tracker import track_disagreements
_CONVICTION_LEVELS = ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
_CONVICTION_ORDER = {v: i for i, v in enumerate(_CONVICTION_LEVELS)}
_POSITION_MAP = {'BUY': {'LOW': 5.0, 'MEDIUM': 10.0, 'HIGH': 20.0, 'VERY_HIGH': 30.0}, 'SELL': {'LOW': -5.0, 'MEDIUM': -10.0, 'HIGH': -20.0, 'VERY_HIGH': -30.0}, 'HOLD': {'LOW': 0.0, 'MEDIUM': 0.0, 'HIGH': 0.0, 'VERY_HIGH': 0.0}, 'REDUCE': {'LOW': -3.0, 'MEDIUM': -5.0, 'HIGH': -10.0, 'VERY_HIGH': -15.0}}

def _conviction_from_numeric(value: float) -> str:
    if value >= 0.8:
        return 'VERY_HIGH'
    if value >= 0.6:
        return 'HIGH'
    if value >= 0.4:
        return 'MEDIUM'
    return 'LOW'

def _downgrade_conviction(conviction: str, steps: int=1) -> str:
    idx = _CONVICTION_ORDER.get(conviction, 1)
    new_idx = max(0, idx - steps)
    return _CONVICTION_LEVELS[new_idx]

def _generate_research_report(decision: Dict[str, Any], state: Dict[str, Any], ticker: str, trade_date: str) -> LayeredResearchReport:
    signal = str(decision.get('signal', 'HOLD')).upper()
    if signal not in ('BUY', 'SELL', 'HOLD', 'REDUCE'):
        signal = 'HOLD'
    numeric_conviction = float(decision.get('conviction', 0.5) or 0.5)
    conviction = _conviction_from_numeric(numeric_conviction)
    analyst_reports = {}
    for key in ['market_report', 'sentiment_report', 'news_report', 'fundamentals_report', 'macro_analysis']:
        val = state.get(key)
        if val:
            analyst_reports[key] = str(val)
    thesis = str(decision.get('rationale', '') or '')
    debate_state = state.get('investment_debate_state') or {}
    if debate_state.get('judge_decision'):
        thesis = str(debate_state['judge_decision'])
    catalysts = []
    raw_risks = decision.get('key_risks', [])
    if isinstance(raw_risks, list):
        risks = [str(r) for r in raw_risks]
    else:
        risks = [str(raw_risks)] if raw_risks else []
    if thesis:
        for line in thesis.split('\n'):
            ls = line.strip()
            if any((kw in ls.lower() for kw in ['catalyst', 'upside', 'positive', 'tailwind'])):
                catalysts.append(ls)
    if not catalysts:
        catalysts = [f'Research supports {signal}']
    dq = decision.get('data_quality', {})
    data_quality_map = {}
    if isinstance(dq, dict):
        for k, v in dq.items():
            data_quality_map[k] = str(v)
    quant_metrics = {}
    qr = decision.get('quant_risk_report', {})
    if isinstance(qr, dict) and qr.get('metrics'):
        for k, v in qr['metrics'].items():
            try:
                quant_metrics[k] = float(v)
            except (TypeError, ValueError):
                pass
    return LayeredResearchReport(ticker=ticker, as_of_date=trade_date, investment_thesis=thesis, conviction_level=conviction, recommendation=signal, catalysts=catalysts, key_risks=risks if risks else ['Standard market risks apply'], analyst_reports=analyst_reports, data_quality=data_quality_map, quant_metrics=quant_metrics)

def _trading_interpret_research(research_report: LayeredResearchReport, config: Dict[str, Any]) -> TradingPlan:
    layered_cfg = config.get('layered_decision', {})
    enable_interpretation = layered_cfg.get('enable_trading_interpretation', True)
    rec = research_report.recommendation
    conviction = research_report.conviction_level
    disagreement_reasons = []
    constraints = []
    if enable_interpretation:
        if conviction in ('VERY_HIGH', 'HIGH'):
            trading_conviction = _downgrade_conviction(conviction, steps=1)
            disagreement_reasons.append(f'Conviction downgraded from {conviction} to {trading_conviction} due to liquidity assessment')
            constraints.append('liquidity_discount')
        else:
            trading_conviction = conviction
        missing = [k for k, v in research_report.data_quality.items() if str(v).lower() in ('missing', 'fail', 'none', '')]
        if missing:
            trading_conviction = _downgrade_conviction(trading_conviction, steps=1)
            disagreement_reasons.append(f"Further downgrade due to data quality issues: {', '.join(missing)}")
            constraints.append('data_quality_penalty')
    else:
        trading_conviction = conviction
    position_pct = _POSITION_MAP.get(rec, _POSITION_MAP['HOLD']).get(trading_conviction, 0.0)
    return TradingPlan(research_report_id=research_report.report_id, trading_recommendation=rec, trading_conviction=trading_conviction, position_target_pct=position_pct, execution_strategy='TWAP' if abs(position_pct) > 15 else 'market', urgency='HIGH' if conviction == 'VERY_HIGH' else 'NORMAL', disagrees_with_research=len(disagreement_reasons) > 0, disagreement_reasons=disagreement_reasons, constraints_applied=constraints)

def _risk_management_review(trading_plan: TradingPlan, research_report: LayeredResearchReport, config: Dict[str, Any]) -> RiskReview:
    layered_cfg = config.get('layered_decision', {})
    enable_veto = layered_cfg.get('enable_risk_veto', True)
    risk_flags = []
    risk_score = 0.0
    final_position = trading_plan.position_target_pct
    final_rec = trading_plan.trading_recommendation
    position_reduced = False
    reduction_reason = ''
    vetoed = False
    veto_reason = ''
    pit_passed = True
    if enable_veto:
        missing_reports = [k for k, v in research_report.data_quality.items() if str(v).lower() in ('missing', 'fail', 'none', '')]
        if missing_reports:
            risk_score += 0.2 * len(missing_reports)
            risk_flags.append(f"Data quality issues: {', '.join(missing_reports)}")
        if abs(final_position) > 25:
            risk_flags.append('Position exceeds 25% limit')
            final_position = 25.0 if final_position > 0 else -25.0
            position_reduced = True
            reduction_reason = 'Position capped at 25% per risk policy'
            risk_score += 0.3
        if trading_plan.trading_conviction == 'LOW' and abs(final_position) > 5:
            risk_flags.append('LOW conviction with large position')
            final_position = 5.0 if final_position > 0 else -5.0
            position_reduced = True
            reduction_reason = 'Position reduced due to LOW conviction'
            risk_score += 0.2
        try:
            report_date = datetime.strptime(research_report.as_of_date, '%Y-%m-%d')
            if report_date.date() > datetime.now().date():
                pit_passed = False
                risk_flags.append('PIT VIOLATION: Report date is in the future')
                risk_score += 0.5
        except (ValueError, TypeError):
            pass
        if risk_score >= 0.8:
            vetoed = True
            veto_reason = f'Risk score {risk_score:.2f} exceeds veto threshold (0.8)'
            final_position = 0.0
            final_rec = 'HOLD'
        risk_score = min(risk_score, 1.0)
    return RiskReview(trading_plan_id=trading_plan.research_report_id, approved=not vetoed, final_recommendation=final_rec, final_position_pct=final_position, position_reduced=position_reduced, reduction_reason=reduction_reason, vetoed=vetoed, veto_reason=veto_reason, risk_score=risk_score, risk_flags=risk_flags, pit_validation_passed=pit_passed)

def run_layered_decision(decision: Dict[str, Any], state: Dict[str, Any], ticker: str, trade_date: str, config: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    config = config or DEFAULT_CONFIG.copy()
    research_report = _generate_research_report(decision, state, ticker, trade_date)
    trading_plan = _trading_interpret_research(research_report, config)
    risk_review = _risk_management_review(trading_plan, research_report, config)
    layered_cfg = config.get('layered_decision', {})
    if layered_cfg.get('track_disagreements', True):
        disagreements = track_disagreements(research_report, trading_plan, risk_review)
    else:
        disagreements = {'research_to_trading': {}, 'trading_to_risk': {}, 'total_disagreement_score': 0.0}
    audit = LayerAudit(ticker=ticker, as_of_date=trade_date, research_report=research_report, trading_plan=trading_plan, risk_review=risk_review, research_to_trading_delta=disagreements.get('research_to_trading', {}), trading_to_risk_delta=disagreements.get('trading_to_risk', {}), total_disagreement_score=disagreements.get('total_disagreement_score', 0.0))
    return {'_research_report': research_report.to_dict(), '_trading_plan': trading_plan.to_dict(), '_risk_review': risk_review.to_dict(), '_layer_audit': audit.to_dict()}
