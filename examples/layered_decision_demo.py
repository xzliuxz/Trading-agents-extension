import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.modes.schemas import LayeredResearchReport, TradingPlan, RiskReview
from tradingagents.modes.layered_decision import run_layered_decision, _generate_research_report, _trading_interpret_research, _risk_management_review
from tradingagents.modes.disagreement_tracker import track_disagreements
from tradingagents.modes.horizon_optimizer import calculate_optimal_horizon, get_research_update_schedule, explain_horizon_choice

def demo_layered_with_mock():
    print('=' * 70)
    print('  TradingAgents Extension - Layered Decision Demo (Mock Data)')
    print('=' * 70)
    mock_decision = {'signal': 'BUY', 'conviction': 0.72, 'position_target_pct': 18.0, 'rationale': 'Strong fundamentals with positive catalyst pipeline', 'key_risks': ['Earnings miss risk', 'Sector rotation', 'Valuation stretched'], 'data_quality': {'market_report': 'OK', 'news_report': 'OK', 'fundamentals_report': 'OK', 'sentiment_report': 'OK'}, 'quant_risk_report': {'status': 'OK', 'metrics': {'annual_vol': 35.0, 'sharpe': 1.2}}}
    mock_state = {'market_report': 'Bullish technical setup with breakout above resistance', 'sentiment_report': 'Positive social sentiment', 'news_report': 'Favorable regulatory developments', 'fundamentals_report': 'Revenue growth exceeding expectations', 'investment_debate_state': {'judge_decision': 'BUY with HIGH conviction. Strong upside catalyst from product launch.'}, 'final_trade_decision': 'BUY - HIGH conviction.'}
    config = DEFAULT_CONFIG.copy()
    config['decision_mode'] = 'layered'
    result = run_layered_decision(decision=mock_decision, state=mock_state, ticker='NVDA', trade_date='2025-01-10', config=config)
    report = result['_research_report']
    plan = result['_trading_plan']
    review = result['_risk_review']
    audit = result['_layer_audit']
    print(f'\n  Ticker: NVDA   Date: 2025-01-10')
    print()
    print(f'  [Stage 1: Research Department]')
    print(f"    Recommendation: {report['recommendation']}")
    print(f"    Conviction:     {report['conviction_level']}")
    print(f'\n  [Stage 2: Trading Desk]')
    print(f"    Recommendation: {plan['trading_recommendation']}")
    print(f"    Conviction:     {plan['trading_conviction']}")
    print(f"    Position:       {plan['position_target_pct']:.1f}%")
    if plan['disagrees_with_research']:
        for reason in plan['disagreement_reasons']:
            print(f'    DISAGREE: {reason}')
    print(f'\n  [Stage 3: Risk Management]')
    print(f"    Status:         {('VETOED' if review['vetoed'] else 'APPROVED')}")
    print(f"    Final Rec:      {review['final_recommendation']}")
    print(f"    Final Position: {review['final_position_pct']:.1f}%")
    print(f"    Risk Score:     {review['risk_score']:.2f}")
    print(f"\n  Disagreement Score: {audit['total_disagreement_score']:.4f}")

def demo_horizon_optimizer():
    print()
    print('=' * 70)
    print('  Horizon Optimizer Demo')
    print('=' * 70)
    print()
    test_cases = [4, 8, 12, 20, 24, 30, 36, 48, 52, 100]
    for weeks in test_cases:
        horizon, explanation = explain_horizon_choice(weeks)
        schedule = get_research_update_schedule(weeks, horizon)
        print(f'  {weeks:3d} weeks -> {horizon:3s} horizon -> schedule: {schedule}')
        print(f'    {explanation}')
        print()

def demo_compare_modes():
    print()
    print('=' * 70)
    print('  Mode Comparison (config only)')
    print('=' * 70)
    print()
    config = DEFAULT_CONFIG.copy()
    print(f"  decision_mode default: {config['decision_mode']}")
    print(f"  llm_provider default:  {config['llm_provider']}")
    print(f"  layered_decision:      {config.get('layered_decision', {})}")
    print()
    print('  Legacy: python3 main.py research NVDA --date 2025-01-10')
    print('  Layered: python3 main.py research NVDA --date 2025-01-10 --decision-mode layered')
    print()
    print('  In legacy mode, run_research_mode() returns:')
    print('    (decision, state)')
    print()
    print('  In layered mode, decision additionally contains:')
    print('    _research_report, _trading_plan, _risk_review, _layer_audit')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TradingAgents Extension - Layered Decision Demo')
    parser.add_argument('--demo', choices=['full', 'horizon', 'compare'], default='full')
    args = parser.parse_args()
    if args.demo == 'full':
        demo_layered_with_mock()
    elif args.demo == 'horizon':
        demo_horizon_optimizer()
    else:
        demo_compare_modes()
        demo_horizon_optimizer()
