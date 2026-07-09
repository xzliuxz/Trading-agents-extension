import re
from typing import Any, Dict, Tuple

def word_count(text: str) -> int:
    content = str(text or '')
    return len(re.findall("\\b[\\w'-]+\\b", content))

def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default

def resolve_longform_requirements(config: Dict[str, Any] | None, role_key: str, default_words: int, default_sections: int) -> Tuple[int, int, int]:
    cfg = config or {}
    base_words = max(200, _safe_int(cfg.get('research_min_words_default', default_words), default_words))
    base_sections = max(3, _safe_int(cfg.get('research_min_sections_default', default_sections), default_sections))
    min_words = max(200, _safe_int(cfg.get(f'research_min_words_{role_key}', base_words), base_words))
    min_sections = max(3, _safe_int(cfg.get(f'research_min_sections_{role_key}', base_sections), base_sections))
    max_rounds = max(1, _safe_int(cfg.get('research_longform_max_rounds', 1), 1))
    return (min_words, min_sections, max_rounds)

def ensure_long_form_report(llm: Any, report_text: str, role_name: str, min_words: int, min_sections: int, citation_style_hint: str, max_expansion_rounds: int=1) -> str:
    draft = str(report_text or '').strip()
    if not draft:
        return draft
    if llm is None:
        return draft
    target_words = max(200, int(min_words))
    target_sections = max(3, int(min_sections))
    rounds = max(1, int(max_expansion_rounds))
    best_text = draft
    best_words = word_count(draft)
    if best_words >= target_words:
        return best_text
    for _ in range(rounds):
        prompt = f'You are editing a professional equity research memo written by the {role_name}.\nExpand the draft into a deeper long-form memo without changing the factual basis.\n\nHard requirements:\n- English only.\n- At least {target_words} words and at least {target_sections} titled sections.\n- Preserve all existing numeric values and factual claims unless explicitly flagged as uncertain.\n- Preserve and reuse existing inline citations. When adding support, keep the same citation marker style (example: {citation_style_hint}).\n- Every factual statement must include an inline citation in the same sentence using source tag style like [Source: X] (or preserve an existing equivalent marker).\n- Remove unsourced factual statements instead of leaving them uncited.\n- Include explicit sections for: thesis, evidence breakdown, counterarguments, scenario analysis, risk triggers, and monitoring checklist.\n- End with a concise action recommendation.\n\nDraft:\n{best_text}\n\nReturn only the revised memo in Markdown.\n'
        try:
            response = llm.invoke(prompt)
        except Exception:
            break
        candidate = getattr(response, 'content', response)
        candidate_text = str(candidate or '').strip()
        if not candidate_text:
            continue
        candidate_words = word_count(candidate_text)
        if candidate_words > best_words:
            best_text = candidate_text
            best_words = candidate_words
        if best_words >= target_words:
            break
    return best_text
