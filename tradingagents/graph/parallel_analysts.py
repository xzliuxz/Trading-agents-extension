from __future__ import annotations
import concurrent.futures
import traceback
from typing import Any, Callable, Dict, List
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from tradingagents.agents.utils.agent_states import AgentState

def _generic_should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'Tools'
    return 'Clear'

def _create_clear_node():

    def clear(state):
        messages = state['messages']
        removal_ops = [RemoveMessage(id=m.id) for m in messages]
        placeholder = HumanMessage(content='Continue')
        return {'messages': removal_ops + [placeholder]}
    return clear

def compile_analyst_subgraph(analyst_node: Callable, tool_node: ToolNode) -> Any:
    sub = StateGraph(AgentState)
    sub.add_node('Analyst', analyst_node)
    sub.add_node('Tools', tool_node)
    sub.add_node('Clear', _create_clear_node())
    sub.add_edge(START, 'Analyst')
    sub.add_conditional_edges('Analyst', _generic_should_continue, ['Tools', 'Clear'])
    sub.add_edge('Tools', 'Analyst')
    sub.add_edge('Clear', END)
    return sub.compile()
REPORT_KEY_MAP: Dict[str, str] = {'market': 'market_report', 'social': 'sentiment_report', 'news': 'news_report', 'fundamentals': 'fundamentals_report'}

def create_parallel_analysts_node(analyst_configs: Dict[str, Dict[str, Any]], max_workers: int=4, recursion_limit: int=100) -> Callable:
    subgraphs: Dict[str, Any] = {}
    for atype, cfg in analyst_configs.items():
        subgraphs[atype] = compile_analyst_subgraph(analyst_node=cfg['node'], tool_node=cfg['tools'])

    def _run_single_analyst(atype: str, subgraph: Any, state_snapshot: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        init_state: Dict[str, Any] = {'messages': [('human', state_snapshot['company_of_interest'])], 'company_of_interest': state_snapshot['company_of_interest'], 'trade_date': state_snapshot['trade_date'], 'macro_guidance': state_snapshot.get('macro_guidance', ''), 'macro_analysis': state_snapshot.get('macro_analysis', {}), 'target_allocations': state_snapshot.get('target_allocations', '{}'), 'market_report': '', 'sentiment_report': '', 'news_report': '', 'fundamentals_report': '', 'investment_debate_state': {'history': '', 'current_response': '', 'count': 0, 'bull_history': '', 'bear_history': '', 'judge_decision': ''}, 'risk_debate_state': {'history': '', 'current_risky_response': '', 'current_safe_response': '', 'current_neutral_response': '', 'count': 0, 'latest_speaker': '', 'judge_decision': ''}}
        result = subgraph.invoke(init_state, config={'recursion_limit': recursion_limit})
        return (atype, result)

    def parallel_analysts_node(state: AgentState) -> Dict[str, Any]:
        state_snapshot = dict(state)
        updates: Dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_single_analyst, atype, sg, state_snapshot): atype for atype, sg in subgraphs.items()}
            for future in concurrent.futures.as_completed(futures):
                atype = futures[future]
                try:
                    _, result = future.result()
                    report_key = REPORT_KEY_MAP.get(atype)
                    if report_key and result.get(report_key):
                        updates[report_key] = result[report_key]
                        print(f'[Parallel] ✅ {atype} analyst completed')
                except Exception as exc:
                    print(f'[Parallel] ❌ {atype} analyst failed: {exc}')
                    traceback.print_exc()
        messages = state.get('messages', [])
        removal_ops = [RemoveMessage(id=m.id) for m in messages]
        updates['messages'] = removal_ops + [HumanMessage(content='Continue')]
        return updates
    return parallel_analysts_node
