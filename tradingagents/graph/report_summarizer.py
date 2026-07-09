from __future__ import annotations
import re
from typing import Any, Dict
from tradingagents.agents.utils.agent_states import AgentState

def word_count(text: str) -> int:
    return len(re.findall("\\\\b[\\\\w'-]+\\\\b", str(text or '')))
_SUMMARISE_PROMPT = 'You are a senior research editor. Compress the following analyst report into a concise executive summary.\n\nHard rules:\n- Maximum 350 words.\n- Preserve ALL numeric values, key findings, and [Source: X] citations.\n- Keep the recommendation (BUY/SELL/HOLD) and conviction level if mentioned.\n- Use bullet points for key takeaways.\n- Remove boilerplate, caveats, and repeated information.\n\nOriginal report:\n{report}\n\nOutput the compressed summary only.'

def create_report_summarizer(llm, max_words: int=500):

    def report_summarizer_node(state: AgentState) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        report_keys = ['market_report', 'sentiment_report', 'news_report', 'fundamentals_report']
        for key in report_keys:
            report = state.get(key, '')
            if not report or word_count(report) <= max_words:
                continue
            try:
                prompt = _SUMMARISE_PROMPT.format(report=report)
                response = llm.invoke(prompt)
                compressed = response.content if hasattr(response, 'content') else str(response)
                if compressed and word_count(compressed) < word_count(report):
                    updates[key] = compressed
                    print(f'[Summarizer] {key}: {word_count(report)} → {word_count(compressed)} words')
            except Exception as exc:
                print(f'[Summarizer] Failed to compress {key}: {exc}')
        return updates
    return report_summarizer_node
