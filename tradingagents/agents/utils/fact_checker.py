from __future__ import annotations
import re
from typing import NamedTuple
_CLAIM_PATTERNS = [re.compile('\\$[\\d,.]+\\s*(?:[BMKTbmkt](?:illion|rillion)?)?'), re.compile('[\\d,.]+\\s*%'), re.compile('[\\d,.]+\\s*(?:basis\\s+points?|bps)', re.IGNORECASE), re.compile('[\\d,.]+\\s*x\\b'), re.compile('(?:P/E|EPS|EBITDA|FCF|ROE|ROA|ROIC|EV/EBITDA)\\s*(?:of\\s+)?[\\d,.]+', re.IGNORECASE), re.compile('(?:revenue|earnings|profit|sales|margin|growth|decline|increase|decrease|rose|fell|dropped|gained|surged|plunged)\\s+(?:of\\s+|by\\s+|to\\s+)?[\\d,.]+\\s*%?', re.IGNORECASE)]
_SOURCE_PATTERN = re.compile('\\[Source:\\s*[^\\]]+\\]', re.IGNORECASE)

class ClaimCheckResult(NamedTuple):
    checked_text: str
    total_claims: int
    sourced_claims: int
    unsourced_claims: int
    claim_ratio: float

def _sentence_has_source(sentence: str) -> bool:
    return bool(_SOURCE_PATTERN.search(sentence))

def _sentence_has_claim(sentence: str) -> bool:
    for pat in _CLAIM_PATTERNS:
        if pat.search(sentence):
            return True
    return False

def _split_sentences(text: str) -> list[str]:
    parts = re.split('(?<=[.!?])\\s+(?=[A-Z\\[\\(])', text)
    return parts

def enforce_claim_citations(report_text: str, tag: str='(Unverified)') -> ClaimCheckResult:
    if not report_text or not report_text.strip():
        return ClaimCheckResult(checked_text=report_text, total_claims=0, sourced_claims=0, unsourced_claims=0, claim_ratio=1.0)
    lines = report_text.split('\n')
    output_lines = []
    total_claims = 0
    sourced = 0
    unsourced = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('---') or stripped.startswith('|'):
            output_lines.append(line)
            continue
        sentences = _split_sentences(line)
        processed_sentences = []
        for sentence in sentences:
            if _sentence_has_claim(sentence):
                total_claims += 1
                if _sentence_has_source(sentence):
                    sourced += 1
                    processed_sentences.append(sentence)
                else:
                    unsourced += 1
                    s = sentence.rstrip()
                    if s.endswith('.'):
                        s = s[:-1] + f' {tag}.'
                    else:
                        s = s + f' {tag}'
                    processed_sentences.append(s)
            else:
                processed_sentences.append(sentence)
        output_lines.append(' '.join(processed_sentences))
    ratio = sourced / total_claims if total_claims > 0 else 1.0
    return ClaimCheckResult(checked_text='\n'.join(output_lines), total_claims=total_claims, sourced_claims=sourced, unsourced_claims=unsourced, claim_ratio=ratio)

def format_claim_audit(result: ClaimCheckResult) -> str:
    pct = result.claim_ratio * 100
    status = 'PASS' if pct >= 80 else 'WARN' if pct >= 50 else 'FAIL'
    return f'[Fact-Checker] {status} — {result.sourced_claims}/{result.total_claims} claims sourced ({pct:.0f}%), {result.unsourced_claims} marked (Unverified)'
