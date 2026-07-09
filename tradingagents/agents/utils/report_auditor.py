from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set
_INLINE_SOURCE_PATTERN = re.compile('\\[Source:\\s*[^\\]]+\\]', flags=re.IGNORECASE)
_INLINE_ALPHA_LABEL_PATTERN = re.compile('\\[([A-Za-z]{1,8}\\d{1,4})\\]')
_INLINE_NUMERIC_LABEL_PATTERN = re.compile('\\[(\\d{1,4}(?:\\s*,\\s*\\d{1,4})*)\\]')
_URL_PATTERN = re.compile('https?://\\S+', flags=re.IGNORECASE)
_REFERENCE_ALPHA_LABEL_LINE = re.compile('^\\s*[-*]?\\s*\\[([A-Za-z]{1,8}\\d{1,4})\\]\\s+')
_REFERENCE_NUMERIC_LABEL_LINE = re.compile('^\\s*[-*]?\\s*\\[(\\d{1,4})\\]\\s+')
_REFERENCE_LABEL_TOKEN = re.compile('^\\s*[-*]?\\s*\\[[^\\]]+\\]\\s+')
_DIGIT_PATTERN = re.compile('\\d')
_ORDERED_LIST_PREFIX_PATTERN = re.compile('^\\s*(\\d+)\\.\\s+')
_ALREADY_UNVERIFIED_PATTERN = re.compile('\\((?:estimate|unverified|estimate/unverified)[^)]*\\)', flags=re.IGNORECASE)

def _extract_inline_labels(line: str) -> Set[str]:
    labels: Set[str] = set()
    text = str(line or '')
    for match in _INLINE_ALPHA_LABEL_PATTERN.finditer(text):
        labels.add(match.group(1).upper())
    for match in _INLINE_NUMERIC_LABEL_PATTERN.finditer(text):
        for token in str(match.group(1) or '').split(','):
            stripped = token.strip()
            if stripped:
                labels.add(stripped)
    return labels

def _collect_reference_labels(lines: Sequence[str]) -> Set[str]:
    labels: Set[str] = set()
    for raw_line in lines:
        line = str(raw_line or '').strip()
        if not line:
            continue
        alpha_match = _REFERENCE_ALPHA_LABEL_LINE.match(line)
        if alpha_match:
            labels.add(alpha_match.group(1).upper())
            continue
        num_match = _REFERENCE_NUMERIC_LABEL_LINE.match(line)
        if num_match:
            labels.add(num_match.group(1))
    return labels

def _append_tag(line: str, tag: str) -> str:
    raw = str(line or '')
    stripped = raw.rstrip()
    if not stripped:
        return raw
    lowered = stripped.lower()
    if str(tag or '').strip().lower() in lowered:
        return raw
    if _ALREADY_UNVERIFIED_PATTERN.search(stripped):
        return raw
    if stripped[-1] in '.!?':
        return f'{stripped[:-1]} {tag}{stripped[-1]}'
    return f'{stripped} {tag}'

def _is_skipped_line(line: str) -> bool:
    stripped = str(line or '').strip()
    if not stripped:
        return True
    if stripped.startswith('#'):
        return True
    if stripped.startswith('```'):
        return True
    if stripped.startswith('|'):
        return True
    if stripped.startswith('!['):
        return True
    if re.fullmatch('-{3,}', stripped):
        return True
    return False

def _is_numeric_claim_line(line: str, strict_numeric: bool) -> bool:
    stripped = str(line or '').strip()
    if not stripped:
        return False
    if _REFERENCE_LABEL_TOKEN.match(stripped):
        return False
    if not _DIGIT_PATTERN.search(stripped):
        return False
    ordered_list = _ORDERED_LIST_PREFIX_PATTERN.match(stripped)
    if ordered_list:
        tail = stripped[ordered_list.end():]
        if not _DIGIT_PATTERN.search(tail):
            return False
    if strict_numeric:
        return True
    if re.search('\\$[\\d,.]+', stripped):
        return True
    if re.search('[\\d,.]+\\s*%', stripped):
        return True
    if re.search('[\\d,.]+\\s*(?:bps|basis\\s+points?)', stripped, flags=re.IGNORECASE):
        return True
    if re.search('\\b[\\d,.]+\\s*x\\b', stripped, flags=re.IGNORECASE):
        return True
    return False

def _line_is_sourced(line: str, known_reference_labels: Set[str]) -> bool:
    text = str(line or '')
    if not text.strip():
        return True
    if _URL_PATTERN.search(text):
        return True
    if _INLINE_SOURCE_PATTERN.search(text):
        return True
    labels = _extract_inline_labels(text)
    if not labels:
        return False
    if not known_reference_labels:
        return False
    return bool(labels & known_reference_labels)

@dataclass
class ReportAuditResult:
    audited_text: str
    total_numeric_lines: int
    sourced_numeric_lines: int
    unsourced_numeric_lines: int
    sourced_ratio: float
    tag: str
    known_reference_labels: List[str] = field(default_factory=list)
    flagged_examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {'total_numeric_lines': int(self.total_numeric_lines), 'sourced_numeric_lines': int(self.sourced_numeric_lines), 'unsourced_numeric_lines': int(self.unsourced_numeric_lines), 'sourced_ratio': round(float(self.sourced_ratio), 4), 'tag': str(self.tag), 'known_reference_label_count': int(len(self.known_reference_labels)), 'known_reference_labels': list(self.known_reference_labels), 'flagged_examples': list(self.flagged_examples)}

class AuditorAgent:

    def __init__(self, *, tag: str='(Estimate/Unverified)', strict_numeric: bool=True, max_flagged_examples: int=12) -> None:
        cleaned_tag = str(tag or '').strip()
        self.tag = cleaned_tag or '(Estimate/Unverified)'
        self.strict_numeric = bool(strict_numeric)
        self.max_flagged_examples = max(1, int(max_flagged_examples))

    def audit_markdown(self, markdown_text: str) -> ReportAuditResult:
        source = str(markdown_text or '')
        lines = source.splitlines()
        known_reference_labels = _collect_reference_labels(lines)
        output_lines: List[str] = []
        total_numeric_lines = 0
        sourced_numeric_lines = 0
        unsourced_numeric_lines = 0
        flagged_examples: List[str] = []
        for raw_line in lines:
            line = str(raw_line or '')
            if _is_skipped_line(line):
                output_lines.append(line)
                continue
            if not _is_numeric_claim_line(line, strict_numeric=self.strict_numeric):
                output_lines.append(line)
                continue
            total_numeric_lines += 1
            if _line_is_sourced(line, known_reference_labels=known_reference_labels):
                sourced_numeric_lines += 1
                output_lines.append(line)
                continue
            unsourced_numeric_lines += 1
            audited_line = _append_tag(line, self.tag)
            output_lines.append(audited_line)
            if len(flagged_examples) < self.max_flagged_examples:
                example = re.sub('\\s+', ' ', line).strip()
                if example:
                    flagged_examples.append(example[:220])
        sourced_ratio = float(sourced_numeric_lines) / float(total_numeric_lines) if total_numeric_lines > 0 else 1.0
        return ReportAuditResult(audited_text='\n'.join(output_lines), total_numeric_lines=total_numeric_lines, sourced_numeric_lines=sourced_numeric_lines, unsourced_numeric_lines=unsourced_numeric_lines, sourced_ratio=sourced_ratio, tag=self.tag, known_reference_labels=sorted(known_reference_labels), flagged_examples=flagged_examples)

def run_rigorous_report_audit(markdown_text: str, *, tag: str='(Estimate/Unverified)', strict_numeric: bool=True, max_flagged_examples: int=12) -> ReportAuditResult:
    agent = AuditorAgent(tag=tag, strict_numeric=strict_numeric, max_flagged_examples=max_flagged_examples)
    return agent.audit_markdown(markdown_text)

def format_report_auditor_summary(result: ReportAuditResult) -> str:
    pct = round(float(result.sourced_ratio) * 100.0, 1)
    return f'sourced {result.sourced_numeric_lines}/{result.total_numeric_lines} numeric lines ({pct}%), flagged {result.unsourced_numeric_lines} with {result.tag}'
