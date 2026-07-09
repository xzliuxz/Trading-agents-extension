from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_package_version():
    import tradingagents
    assert tradingagents.__version__

def test_default_config():
    from tradingagents.default_config import DEFAULT_CONFIG
    assert 'macro_data_source' in DEFAULT_CONFIG
    assert DEFAULT_CONFIG['use_enhanced_regime_detector'] is True

def test_fact_checker_marks_unsourced_claim():
    from tradingagents.agents.utils.fact_checker import enforce_claim_citations
    result = enforce_claim_citations('Revenue grew 25% year over year.')
    assert result.total_claims >= 1
    assert '25%' in result.checked_text
    assert result.unsourced_claims >= 1
    assert 'Unverified' in result.checked_text

def test_risk_framework_imports():
    from tradingagents.utils.risk_framework import DynamicRiskManager, RiskAttributor, RiskBudgetManager, StressTester
    assert RiskBudgetManager is not None
    assert RiskAttributor is not None
    assert StressTester is not None
    assert DynamicRiskManager is not None

def test_decision_layers_import():
    from tradingagents.graph.decision_layers import DecisionCoordinator, ExecutionLayer, StrategicLayer, TacticalLayer
    assert DecisionCoordinator is not None
    assert StrategicLayer is not None
    assert TacticalLayer is not None
    assert ExecutionLayer is not None

def test_factor_analyst_import():
    from tradingagents.agents.analysts.factor_analyst import FactorAnalyst
    fa = FactorAnalyst()
    assert hasattr(fa, 'FACTORS')
    assert 'momentum' in fa.FACTORS

def test_institutional_framework_constructs():
    from tradingagents.institutional_framework import InstitutionalTradingFramework
    fw = InstitutionalTradingFramework(config={'macro_data_source': 'yfinance'})
    assert fw is not None
    assert hasattr(fw, 'analyze_macro_environment')

def test_layered_decision_helpers_exist():
    from tradingagents.modes.layered_decision import _generate_research_report, _risk_management_review, _trading_interpret_research, run_layered_decision
    assert callable(run_layered_decision)
    assert callable(_generate_research_report)
    assert callable(_trading_interpret_research)
    assert callable(_risk_management_review)

def test_time_machine():
    from tradingagents.utils.time_machine import TimeMachine, get_current_date
    with TimeMachine('2024-06-15'):
        assert get_current_date() == '2024-06-15'
