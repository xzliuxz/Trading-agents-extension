import json
import re
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime as _dt
from typing import Any, Dict, List, Optional
_DECISIONS = ('BUY', 'SELL', 'HOLD')
_SCHEMA_VERSION = 'research_decision_v1'

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    dj_match = re.search('DECISION_JSON:\\s*(\\{[\\s\\S]*?\\})\\s*$', stripped, flags=re.IGNORECASE | re.MULTILINE)
    if dj_match:
        try:
            parsed = json.loads(dj_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    fenced = re.findall('```(?:json)?\\s*(\\{[\\s\\S]*?\\})\\s*```', stripped, flags=re.IGNORECASE)
    for candidate in fenced:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    start = stripped.find('{')
    end = stripped.rfind('}')
    if start >= 0 and end > start:
        candidate = stripped[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None

def _extract_signal(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = ['\\"signal\\"\\s*:\\s*\\"(BUY|SELL|HOLD)\\"', "'signal'\\s*:\\s*'(BUY|SELL|HOLD)'", 'FINAL\\s+TRANSACTION\\s+PROPOSAL\\s*:\\s*(BUY|SELL|HOLD)', 'TRANSACTION\\s+PROPOSAL\\s*:\\s*(BUY|SELL|HOLD)', 'RECOMMENDATION\\s*:\\s*(BUY|SELL|HOLD)', 'FINAL\\s+DECISION\\s*:\\s*(BUY|SELL|HOLD)', 'DECISION\\s*:\\s*(BUY|SELL|HOLD)', '\\*\\*(BUY|SELL|HOLD)\\*\\*']
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    text_upper = text.upper()
    present = {d for d in _DECISIONS if d in text_upper}
    if len(present) == 1:
        return next(iter(present))
    return None

def extract_signal_from_text(text: str) -> Optional[str]:
    return _extract_signal(text)

def _normalize_conviction(value: Any) -> float:
    try:
        conviction = float(value)
    except Exception:
        return 0.5
    if conviction > 1 and conviction <= 100:
        conviction = conviction / 100.0
    conviction = max(0.0, min(1.0, conviction))
    return conviction

def _normalize_position_pct(value: Any) -> float:
    try:
        pct = float(value)
    except Exception:
        return 0.0
    if pct <= 1.0:
        pct = pct * 100.0
    return max(0.0, min(100.0, pct))

def _normalize_key_risks(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    out = []
    for item in items:
        text = str(item).strip()
        if text:
            out.append(text)
    return out[:8]

def _normalize_free_text(value: Any, *, max_chars: int, max_sentence_repeats: int=2) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    text = re.sub('[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]+', ' ', text)
    text = re.sub('\\s+', ' ', text).strip()
    if not text:
        return ''
    sentences = [s.strip() for s in re.split('(?<=[.!?;。！？；])\\s+', text) if s.strip()]
    if len(sentences) >= 4:
        seen: Dict[str, int] = {}
        compact: List[str] = []
        for sentence in sentences:
            key = re.sub('\\s+', ' ', sentence).strip().lower()
            if len(key) >= 18:
                count = seen.get(key, 0)
                if count >= max(1, int(max_sentence_repeats)):
                    continue
                seen[key] = count + 1
            compact.append(sentence)
        if compact:
            text = ' '.join(compact)
    limit = max(1, int(max_chars))
    if len(text) > limit:
        clipped = text[:limit].rstrip()
        if ' ' in clipped:
            clipped = clipped.rsplit(' ', 1)[0].rstrip()
        text = clipped.rstrip(' ,;:') + '...'
    return text

def _normalize_horizon_days(value: Any) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        return 5
    return max(1, min(252, parsed))

def _validate_structured_payload(parsed: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    errors = []
    signal = str(parsed.get('signal', '') or parsed.get('decision', '')).upper()
    if signal not in _DECISIONS:
        errors.append('signal_missing_or_invalid')
        signal = 'HOLD'
    conviction = _normalize_conviction(parsed.get('conviction', 0.5))
    if parsed.get('conviction', None) is None:
        errors.append('conviction_missing')
    position_target_pct = _normalize_position_pct(parsed.get('position_target_pct', parsed.get('position_target', 0.0)))
    rationale = _normalize_free_text(parsed.get('rationale', parsed.get('reasoning', '')), max_chars=1200)
    if not rationale:
        errors.append('rationale_missing')
    key_risks = _normalize_key_risks(parsed.get('key_risks', parsed.get('risks', [])))
    if not key_risks:
        errors.append('key_risks_missing')
    normalized = {'signal': signal, 'conviction': conviction, 'position_target_pct': position_target_pct, 'entry_zone': _normalize_free_text(parsed.get('entry_zone', parsed.get('entry', '')), max_chars=220), 'stop_loss': _normalize_free_text(parsed.get('stop_loss', parsed.get('stop', '')), max_chars=220), 'take_profit': _normalize_free_text(parsed.get('take_profit', parsed.get('take_profit_pct', '')), max_chars=420), 'horizon_trading_days': _normalize_horizon_days(parsed.get('horizon_trading_days', parsed.get('horizon_days', 5))), 'invalidation': _normalize_free_text(parsed.get('invalidation', ''), max_chars=320), 'rationale': rationale, 'key_risks': key_risks}
    return (normalized, errors)

def _fallback_payload_from_text(raw_text: str, fallback_signal: Optional[str]) -> Dict[str, Any]:
    signal = (fallback_signal or _extract_signal(raw_text) or 'HOLD').upper()
    if signal not in _DECISIONS:
        signal = 'HOLD'
    return {'signal': signal, 'conviction': 0.5, 'position_target_pct': 0.0, 'entry_zone': '', 'stop_loss': '', 'take_profit': '', 'horizon_trading_days': 5, 'invalidation': '', 'rationale': '', 'key_risks': []}

def build_research_decision(raw_text: str, ticker: str, as_of_date: str, fallback_signal: Optional[str]=None, data_quality: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    parsed = _extract_json(raw_text) or {}
    normalized = {}
    schema_errors = []
    schema_valid = False
    parse_mode = 'fallback'
    if parsed:
        normalized, schema_errors = _validate_structured_payload(parsed)
        schema_valid = len(schema_errors) == 0
        if schema_valid:
            parse_mode = 'structured'
    if schema_valid:
        decision_core = normalized
    else:
        decision_core = _fallback_payload_from_text(raw_text, fallback_signal)
        if normalized:
            for key in ['entry_zone', 'stop_loss', 'take_profit', 'invalidation', 'rationale', 'key_risks']:
                val = normalized.get(key)
                if val:
                    decision_core[key] = val
    model_quality = parsed.get('data_quality') if isinstance(parsed, dict) else {}
    if not isinstance(model_quality, dict):
        model_quality = {}
    computed_quality = data_quality or {}
    decision_data_quality = {**model_quality, **computed_quality}
    return {'ticker': ticker, 'as_of_date': as_of_date, 'signal': decision_core['signal'], 'conviction': decision_core['conviction'], 'position_target_pct': decision_core['position_target_pct'], 'entry_zone': decision_core['entry_zone'], 'stop_loss': decision_core['stop_loss'], 'take_profit': decision_core['take_profit'], 'horizon_trading_days': decision_core['horizon_trading_days'], 'invalidation': decision_core['invalidation'], 'rationale': decision_core['rationale'], 'key_risks': decision_core['key_risks'], 'schema_version': _SCHEMA_VERSION, 'schema_valid': schema_valid, 'schema_errors': schema_errors, 'parse_mode': parse_mode, 'data_quality': decision_data_quality, 'raw_decision': raw_text}

@dataclass
class LayeredResearchReport:
    ticker: str = ''
    as_of_date: str = ''
    report_id: str = field(default_factory=lambda: str(_uuid.uuid4())[:8])
    investment_thesis: str = ''
    conviction_level: str = 'MEDIUM'
    recommendation: str = 'HOLD'
    price_targets: Dict[str, float] = field(default_factory=lambda: {'bull': 0.0, 'base': 0.0, 'bear': 0.0})
    valuation_methodology: str = ''
    catalysts: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    analyst_reports: Dict[str, str] = field(default_factory=dict)
    data_quality: Dict[str, str] = field(default_factory=dict)
    quant_metrics: Dict[str, float] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: _dt.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {'ticker': self.ticker, 'as_of_date': self.as_of_date, 'report_id': self.report_id, 'investment_thesis': self.investment_thesis, 'conviction_level': self.conviction_level, 'recommendation': self.recommendation, 'price_targets': self.price_targets, 'valuation_methodology': self.valuation_methodology, 'catalysts': self.catalysts, 'key_risks': self.key_risks, 'analyst_reports': self.analyst_reports, 'data_quality': self.data_quality, 'quant_metrics': self.quant_metrics, 'generated_at': self.generated_at}

@dataclass
class TradingPlan:
    research_report_id: str = ''
    trading_recommendation: str = 'HOLD'
    trading_conviction: str = 'MEDIUM'
    position_target_pct: float = 0.0
    execution_strategy: str = ''
    urgency: str = 'NORMAL'
    disagrees_with_research: bool = False
    disagreement_reasons: List[str] = field(default_factory=list)
    constraints_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {'research_report_id': self.research_report_id, 'trading_recommendation': self.trading_recommendation, 'trading_conviction': self.trading_conviction, 'position_target_pct': self.position_target_pct, 'execution_strategy': self.execution_strategy, 'urgency': self.urgency, 'disagrees_with_research': self.disagrees_with_research, 'disagreement_reasons': self.disagreement_reasons, 'constraints_applied': self.constraints_applied}

@dataclass
class RiskReview:
    trading_plan_id: str = ''
    approved: bool = True
    final_recommendation: str = 'HOLD'
    final_position_pct: float = 0.0
    position_reduced: bool = False
    reduction_reason: str = ''
    vetoed: bool = False
    veto_reason: str = ''
    risk_score: float = 0.0
    risk_flags: List[str] = field(default_factory=list)
    pit_validation_passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {'trading_plan_id': self.trading_plan_id, 'approved': self.approved, 'final_recommendation': self.final_recommendation, 'final_position_pct': self.final_position_pct, 'position_reduced': self.position_reduced, 'reduction_reason': self.reduction_reason, 'vetoed': self.vetoed, 'veto_reason': self.veto_reason, 'risk_score': self.risk_score, 'risk_flags': self.risk_flags, 'pit_validation_passed': self.pit_validation_passed}

@dataclass
class LayerAudit:
    ticker: str = ''
    as_of_date: str = ''
    research_report: Optional[LayeredResearchReport] = None
    trading_plan: Optional[TradingPlan] = None
    risk_review: Optional[RiskReview] = None
    research_to_trading_delta: Dict[str, Any] = field(default_factory=dict)
    trading_to_risk_delta: Dict[str, Any] = field(default_factory=dict)
    total_disagreement_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {'ticker': self.ticker, 'as_of_date': self.as_of_date, 'research_report': self.research_report.to_dict() if self.research_report else None, 'trading_plan': self.trading_plan.to_dict() if self.trading_plan else None, 'risk_review': self.risk_review.to_dict() if self.risk_review else None, 'research_to_trading_delta': self.research_to_trading_delta, 'trading_to_risk_delta': self.trading_to_risk_delta, 'total_disagreement_score': self.total_disagreement_score}

    def summary(self) -> str:
        lines = []
        if self.research_report:
            r = self.research_report
            lines.append(f'Research Department: {r.recommendation}, {r.conviction_level} conviction')
        if self.trading_plan:
            t = self.trading_plan
            lines.append(f'Trading Desk: {t.trading_recommendation}, {t.trading_conviction} conviction, {t.position_target_pct:.1f}% position')
            if t.disagrees_with_research:
                for reason in t.disagreement_reasons:
                    lines.append(f'  ↳ Disagreement: {reason}')
        if self.risk_review:
            rr = self.risk_review
            status = 'VETOED' if rr.vetoed else 'APPROVED'
            lines.append(f'Risk Management: {status}, {rr.final_recommendation}, {rr.final_position_pct:.1f}% final position')
            if rr.position_reduced:
                lines.append(f'  ↳ Reduced: {rr.reduction_reason}')
            if rr.vetoed:
                lines.append(f'  ↳ Veto: {rr.veto_reason}')
        return '\n'.join(lines)
