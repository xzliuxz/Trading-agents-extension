from __future__ import annotations
from typing import Annotated, Any, Dict
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict
try:
    from langgraph.graph import MessagesState

    class AgentState(MessagesState):
        company_of_interest: Annotated[str, 'Company that we are interested in trading']
        trade_date: Annotated[str, 'What date we are trading at']
        sender: Annotated[str, 'Agent that sent this message']
        macro_analysis: Annotated[Dict[str, Any], 'Macro analysis payload']
        target_allocations: Annotated[str, 'Target allocations JSON string']
        macro_guidance: Annotated[str, 'Macro guidance text']
        market_report: Annotated[str, 'Market analyst report']
        sentiment_report: Annotated[str, 'Sentiment analyst report']
        news_report: Annotated[str, 'News analyst report']
        fundamentals_report: Annotated[str, 'Fundamentals analyst report']
        investment_debate_state: Annotated[Dict[str, Any], 'Bull/bear debate state']
        investment_plan: Annotated[str, 'Investment plan']
        trader_investment_plan: Annotated[str, 'Trader plan']
        risk_debate_state: Annotated[Dict[str, Any], 'Risk debate state']
        final_trade_decision: Annotated[str, 'Final trade decision']
except Exception:

    class AgentState(TypedDict, total=False):
        messages: list
        company_of_interest: str
        trade_date: str
        sender: str
        macro_analysis: Dict[str, Any]
        target_allocations: str
        macro_guidance: str
        market_report: str
        sentiment_report: str
        news_report: str
        fundamentals_report: str
        investment_debate_state: Dict[str, Any]
        investment_plan: str
        trader_investment_plan: str
        risk_debate_state: Dict[str, Any]
        final_trade_decision: str

class MacroAnalysisState(TypedDict, total=False):
    quadrant: str
    growth_signal: str
    inflation_signal: str
    risk_level: str
    analysis_summary: str
__all__ = ['AgentState', 'MacroAnalysisState']
