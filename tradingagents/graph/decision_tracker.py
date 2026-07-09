from __future__ import annotations
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

class DecisionNode:

    def __init__(self, node_name: str, timestamp: str='', input_summary: str='', output_summary: str='', decision: str='', confidence: float=0.0, key_factors: Optional[List[str]]=None):
        self.node_name = node_name
        self.timestamp = timestamp or datetime.now().isoformat()
        self.input_summary = input_summary
        self.output_summary = output_summary
        self.decision = decision
        self.confidence = confidence
        self.key_factors = key_factors or []

    def to_dict(self) -> Dict[str, Any]:
        return {'node_name': self.node_name, 'timestamp': self.timestamp, 'input_summary': self.input_summary[:500], 'output_summary': self.output_summary[:500], 'decision': self.decision, 'confidence': self.confidence, 'key_factors': self.key_factors}

class DecisionTracker:

    def __init__(self, ticker: str='', trade_date: str=''):
        self.ticker = ticker
        self.trade_date = trade_date
        self.nodes: List[DecisionNode] = []
        self._analyst_sentiments: Dict[str, float] = {}

    def record(self, node_name: str, input_summary: str='', output_summary: str='', decision: str='', confidence: float=0.0, key_factors: Optional[List[str]]=None):
        self.nodes.append(DecisionNode(node_name=node_name, input_summary=input_summary, output_summary=output_summary, decision=decision, confidence=confidence, key_factors=key_factors))

    def record_analyst_sentiment(self, analyst_name: str, sentiment: float):
        self._analyst_sentiments[analyst_name] = sentiment

    def generate_mermaid(self) -> str:
        lines = ['graph TD']
        for i, node in enumerate(self.nodes):
            safe_name = re.sub('[^a-zA-Z0-9]', '_', node.node_name)
            label = node.node_name
            if node.decision:
                label += f'\\n[{node.decision}]'
            if node.confidence:
                label += f'\\nConf: {node.confidence:.0%}'
            lines.append(f'    {safe_name}["{label}"]')
            if i > 0:
                prev_safe = re.sub('[^a-zA-Z0-9]', '_', self.nodes[i - 1].node_name)
                lines.append(f'    {prev_safe} --> {safe_name}')
        for node in self.nodes:
            safe_name = re.sub('[^a-zA-Z0-9]', '_', node.node_name)
            if node.decision.upper() == 'BUY':
                lines.append(f'    style {safe_name} fill:#c8e6c9')
            elif node.decision.upper() == 'SELL':
                lines.append(f'    style {safe_name} fill:#ffcdd2')
            elif node.decision.upper() == 'HOLD':
                lines.append(f'    style {safe_name} fill:#fff9c4')
        return '\n'.join(lines)

    def generate_html(self) -> str:
        mermaid = self.generate_mermaid()
        rows = ''
        for n in self.nodes:
            rows += f"""\n            <tr>\n                <td>{n.node_name}</td>\n                <td>{n.decision or '-'}</td>\n                <td>{n.confidence:.0%}</td>\n                <td style="font-size:0.85em">{(', '.join(n.key_factors) if n.key_factors else '-')}</td>\n                <td style="font-size:0.8em">{n.output_summary[:200]}</td>\n            </tr>"""
        analyst_table = ''
        for name, score in self._analyst_sentiments.items():
            direction = 'Bullish' if score > 0.1 else 'Bearish' if score < -0.1 else 'Neutral'
            analyst_table += f'<tr><td>{name}</td><td>{score:.2f}</td><td>{direction}</td></tr>'
        html = f'<!DOCTYPE html>\n<html><head>\n<meta charset="utf-8">\n<title>Decision Audit: {self.ticker} @ {self.trade_date}</title>\n<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>\n<style>\nbody {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1000px; margin: auto; padding: 20px; }}\ntable {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}\nth, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}\nth {{ background: #f5f5f5; }}\nh1 {{ color: #1a237e; }} h2 {{ color: #283593; }}\n.mermaid {{ text-align: center; margin: 24px 0; }}\n</style>\n</head><body>\n<h1>Decision Audit Report</h1>\n<p><strong>Ticker:</strong> {self.ticker} | <strong>Date:</strong> {self.trade_date}</p>\n\n<h2>Decision Flow</h2>\n<div class="mermaid">{mermaid}</div>\n\n<h2>Decision Chain Detail</h2>\n<table>\n<tr><th>Node</th><th>Decision</th><th>Confidence</th><th>Key Factors</th><th>Summary</th></tr>\n{rows}\n</table>\n\n<h2>Analyst Sentiment Matrix</h2>\n<table>\n<tr><th>Analyst</th><th>Sentiment Score</th><th>Direction</th></tr>\n{analyst_table}\n</table>\n\n<script>mermaid.initialize({{startOnLoad:true}});</script>\n</body></html>'
        return html

    def to_dict(self) -> Dict[str, Any]:
        return {'ticker': self.ticker, 'trade_date': self.trade_date, 'nodes': [n.to_dict() for n in self.nodes], 'analyst_sentiments': self._analyst_sentiments}

    def save(self, directory: str='results/decision_audits'):
        os.makedirs(directory, exist_ok=True)
        base = f'{self.ticker}_{self.trade_date}'
        with open(os.path.join(directory, f'{base}.json'), 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        with open(os.path.join(directory, f'{base}.html'), 'w') as f:
            f.write(self.generate_html())
        print(f'[DecisionTracker] Saved audit to {directory}/{base}.*')

class ConfidenceCalibrator:

    def __init__(self, history_path: str='results/calibration_history.json'):
        self.history_path = history_path
        self.predictions: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    self.predictions = json.load(f)
            except Exception:
                self.predictions = []

    def _save(self):
        os.makedirs(os.path.dirname(self.history_path) or '.', exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(self.predictions, f, indent=2, ensure_ascii=False)

    def record_prediction(self, ticker: str, trade_date: str, signal: str, conviction: float, analyst_signals: Optional[Dict[str, str]]=None):
        self.predictions.append({'ticker': ticker, 'trade_date': trade_date, 'signal': signal.upper(), 'conviction': conviction, 'analyst_signals': analyst_signals or {}, 'actual_return': None, 'resolved': False})
        self._save()

    def resolve_outcome(self, ticker: str, trade_date: str, actual_return: float):
        for pred in self.predictions:
            if pred['ticker'] == ticker and pred['trade_date'] == trade_date and (not pred['resolved']):
                pred['actual_return'] = actual_return
                pred['resolved'] = True
                break
        self._save()

    def get_calibration_stats(self) -> Dict[str, Any]:
        resolved = [p for p in self.predictions if p['resolved']]
        if not resolved:
            return {'total': 0, 'message': 'No resolved predictions yet'}
        hits = 0
        brier_sum = 0.0
        total = len(resolved)
        for p in resolved:
            actual = p['actual_return']
            signal = p['signal']
            conviction = p['conviction']
            correct = signal == 'BUY' and actual > 0 or (signal == 'SELL' and actual < 0) or (signal == 'HOLD' and abs(actual) < 0.02)
            if correct:
                hits += 1
            outcome = 1.0 if correct else 0.0
            brier_sum += (conviction - outcome) ** 2
        hit_rate = hits / total if total else 0
        brier = brier_sum / total if total else 0
        return {'total': total, 'hits': hits, 'hit_rate': round(hit_rate, 4), 'brier_score': round(brier, 4), 'overconfident': brier > 0.25}

    def get_analyst_accuracy(self) -> Dict[str, Dict[str, Any]]:
        resolved = [p for p in self.predictions if p['resolved'] and p.get('analyst_signals')]
        if not resolved:
            return {}
        analyst_stats: Dict[str, Dict[str, int]] = {}
        for p in resolved:
            actual = p['actual_return']
            for analyst, signal in p['analyst_signals'].items():
                if analyst not in analyst_stats:
                    analyst_stats[analyst] = {'total': 0, 'correct': 0}
                analyst_stats[analyst]['total'] += 1
                sig_upper = signal.upper() if isinstance(signal, str) else ''
                correct = sig_upper in ('BUY', 'BULLISH') and actual > 0 or (sig_upper in ('SELL', 'BEARISH') and actual < 0) or (sig_upper in ('HOLD', 'NEUTRAL') and abs(actual) < 0.02)
                if correct:
                    analyst_stats[analyst]['correct'] += 1
        result = {}
        for analyst, stats in analyst_stats.items():
            acc = stats['correct'] / stats['total'] if stats['total'] else 0
            result[analyst] = {'total': stats['total'], 'correct': stats['correct'], 'accuracy': round(acc, 4)}
        return dict(sorted(result.items(), key=lambda x: x[1]['accuracy'], reverse=True))
