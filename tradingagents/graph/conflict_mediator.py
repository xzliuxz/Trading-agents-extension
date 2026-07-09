from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List
from tradingagents.agents.utils.agent_states import AgentState
logger = logging.getLogger(__name__)
_SENTIMENT_EXTRACTION_PROMPT = 'You are a financial sentiment classifier. Read the analyst report below and output ONLY a JSON object with the sentiment assessment.\n\nReport to analyze:\n---\n{report_text}\n---\n\nOutput a single JSON object (no markdown, no extra text):\n{{\n  "sentiment_score": <float between -1.0 (very bearish) and 1.0 (very bullish)>,\n  "direction": "bullish" | "bearish" | "neutral",\n  "confidence": <float between 0.0 and 1.0>,\n  "key_signal": "<one sentence explaining the dominant signal>"\n}}'

def _extract_sentiment_via_llm(llm, report_text: str, analyst_label: str) -> Dict[str, Any]:
    if not report_text or len(report_text.strip()) < 50:
        return {'sentiment_score': 0.0, 'direction': 'neutral', 'confidence': 0.0, 'key_signal': 'Empty report'}
    truncated = report_text[:3000]
    prompt = _SENTIMENT_EXTRACTION_PROMPT.format(report_text=truncated)
    try:
        from tradingagents.utils.llm_retry import invoke_with_retry
        response = invoke_with_retry(llm, prompt, max_retries=1, caller_name=f'Conflict Mediator / {analyst_label}', fallback_content='')
        raw = response.content if hasattr(response, 'content') else str(response)
        parsed = _parse_json_from_text(raw)
        if parsed and 'sentiment_score' in parsed:
            score = float(parsed['sentiment_score'])
            score = max(-1.0, min(1.0, score))
            return {'sentiment_score': score, 'direction': parsed.get('direction', 'neutral'), 'confidence': float(parsed.get('confidence', 0.5)), 'key_signal': parsed.get('key_signal', '')}
    except Exception as e:
        logger.debug('[Conflict Mediator] LLM sentiment failed for %s: %s', analyst_label, e)
    score = _extract_sentiment_heuristic(report_text)
    direction = 'bullish' if score > 0.15 else 'bearish' if score < -0.15 else 'neutral'
    return {'sentiment_score': score, 'direction': direction, 'confidence': 0.3, 'key_signal': 'Heuristic-based (LLM unavailable)'}

def _parse_json_from_text(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    m = re.search('```(?:json)?\\s*\\n?({[\\s\\S]*?})\\s*\\n?```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search('\\{[^{}]*\\"sentiment_score\\"[^{}]*\\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    text = text.strip()
    if text.startswith('{'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return None
_BULL_WORDS = {'buy', 'bullish', 'upgrade', 'overweight', 'outperform', 'positive', 'upside', 'strong buy', 'accumulate', 'breakout', 'recovery', 'accelerating', 'improving', 'opportunity', 'tailwind', 'catalyst', 'momentum', 'growth'}
_BEAR_WORDS = {'sell', 'bearish', 'downgrade', 'underweight', 'underperform', 'negative', 'downside', 'strong sell', 'reduce', 'breakdown', 'deteriorating', 'weakening', 'headwind', 'risk-off', 'recession', 'contraction', 'decline', 'warning'}
_NEGATION_PREFIXES = {'not ', 'no ', "don't ", "doesn't ", 'limited ', 'unlikely '}

def _extract_sentiment_heuristic(text: str) -> float:
    if not text:
        return 0.0
    lower = text.lower()
    bull_count = 0
    bear_count = 0
    for word in _BULL_WORDS:
        count = lower.count(word)
        if count > 0:
            negated = any((lower.count(neg + word) > 0 for neg in _NEGATION_PREFIXES))
            if negated:
                bear_count += count
            else:
                bull_count += count
    for word in _BEAR_WORDS:
        count = lower.count(word)
        if count > 0:
            negated = any((lower.count(neg + word) > 0 for neg in _NEGATION_PREFIXES))
            if negated:
                bull_count += count
            else:
                bear_count += count
    total = bull_count + bear_count
    if total == 0:
        return 0.0
    return (bull_count - bear_count) / total

def _macro_is_bearish(macro_analysis: Dict[str, Any]) -> bool:
    risk = (macro_analysis.get('risk_level') or '').lower()
    growth = (macro_analysis.get('growth_signal') or '').lower()
    return risk == 'high' or growth == 'falling'

def _macro_is_bullish(macro_analysis: Dict[str, Any]) -> bool:
    risk = (macro_analysis.get('risk_level') or '').lower()
    growth = (macro_analysis.get('growth_signal') or '').lower()
    return risk == 'low' and growth == 'rising'

class ConflictReport:

    def __init__(self):
        self.conflicts: List[Dict[str, Any]] = []
        self.overall_severity: str = 'none'
        self.confidence_adjustment: float = 1.0
        self.extra_debate_rounds: int = 0
        self.summary: str = ''
        self.analyst_sentiments: Dict[str, Dict[str, Any]] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {'conflicts': self.conflicts, 'overall_severity': self.overall_severity, 'confidence_adjustment': self.confidence_adjustment, 'extra_debate_rounds': self.extra_debate_rounds, 'summary': self.summary, 'analyst_sentiments': self.analyst_sentiments}

def detect_conflicts(state: AgentState, llm=None) -> ConflictReport:
    report = ConflictReport()
    macro_analysis = state.get('macro_analysis') or {}
    macro_guidance = state.get('macro_guidance') or ''
    if not macro_guidance:
        report.summary = 'No macro guidance available — skipping mediation.'
        return report
    analyst_sentiments: Dict[str, Dict[str, Any]] = {}
    for label, key in [('Market', 'market_report'), ('Sentiment', 'sentiment_report'), ('News', 'news_report'), ('Fundamentals', 'fundamentals_report')]:
        text = state.get(key, '')
        if text and text.strip():
            if llm is not None:
                sentiment = _extract_sentiment_via_llm(llm, text, label)
            else:
                score = _extract_sentiment_heuristic(text)
                sentiment = {'sentiment_score': score, 'direction': 'bullish' if score > 0.15 else 'bearish' if score < -0.15 else 'neutral', 'confidence': 0.3, 'key_signal': 'Heuristic'}
            analyst_sentiments[label] = sentiment
    report.analyst_sentiments = analyst_sentiments
    macro_bearish = _macro_is_bearish(macro_analysis)
    macro_bullish = _macro_is_bullish(macro_analysis)
    for label, sent in analyst_sentiments.items():
        score = sent['sentiment_score']
        conf = sent.get('confidence', 0.5)
        if conf < 0.3:
            continue
        if macro_bearish and score > 0.25:
            report.conflicts.append({'type': 'macro_bearish_analyst_bullish', 'analyst': label, 'analyst_sentiment': round(score, 2), 'analyst_confidence': round(conf, 2), 'macro_signal': 'bearish', 'key_signal': sent.get('key_signal', ''), 'detail': f'{label} analyst is bullish (score={score:.2f}, conf={conf:.0%}) while macro signals falling growth / high risk.'})
        elif macro_bullish and score < -0.25:
            report.conflicts.append({'type': 'macro_bullish_analyst_bearish', 'analyst': label, 'analyst_sentiment': round(score, 2), 'analyst_confidence': round(conf, 2), 'macro_signal': 'bullish', 'key_signal': sent.get('key_signal', ''), 'detail': f'{label} analyst is bearish (score={score:.2f}, conf={conf:.0%}) while macro signals rising growth / low risk.'})
    n = len(report.conflicts)
    if n == 0:
        report.overall_severity = 'none'
        report.confidence_adjustment = 1.0
        report.extra_debate_rounds = 0
        report.summary = 'No macro-micro conflicts detected.'
    elif n == 1:
        report.overall_severity = 'low'
        report.confidence_adjustment = 0.9
        report.extra_debate_rounds = 0
        report.summary = f"Minor conflict: {report.conflicts[0]['detail']}"
    elif n <= 2:
        report.overall_severity = 'medium'
        report.confidence_adjustment = 0.75
        report.extra_debate_rounds = 1
        details = '; '.join((c['detail'] for c in report.conflicts))
        report.summary = f'Moderate conflict ({n} analysts contradict macro): {details}'
    else:
        report.overall_severity = 'high'
        report.confidence_adjustment = 0.6
        report.extra_debate_rounds = 1
        details = '; '.join((c['detail'] for c in report.conflicts))
        report.summary = f'Severe conflict ({n} analysts contradict macro). Recommend caution. Details: {details}'
    return report

def create_conflict_mediator_node(llm=None):

    def conflict_mediator(state: AgentState) -> Dict[str, Any]:
        report = detect_conflicts(state, llm=llm)
        print(f'[Conflict Mediator] Severity: {report.overall_severity}')
        if report.analyst_sentiments:
            for name, sent in report.analyst_sentiments.items():
                print(f"  {name}: {sent['direction']} (score={sent['sentiment_score']:.2f}, conf={sent.get('confidence', 0):.0%})")
        if report.conflicts:
            for c in report.conflicts:
                print(f"  ⚠️ CONFLICT: {c['detail']}")
        return {'macro_fund_debate': report.to_dict()}
    return conflict_mediator
