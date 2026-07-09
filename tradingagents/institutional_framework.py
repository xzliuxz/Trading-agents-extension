import logging
from datetime import datetime
from typing import Dict, Any, Optional
from tradingagents.agents.macro.indicators import YieldCurveAnalyzer, CreditSpreadAnalyzer, VolatilityAnalyzer, FundFlowAnalyzer, EconomicIndicators, RegimeDetector
from tradingagents.utils.risk_framework import RiskBudgetManager, RiskAttributor, StressTester, DynamicRiskManager
from tradingagents.agents.analysts.factor_analyst import FactorAnalyst
from tradingagents.graph.decision_layers import StrategicLayer, TacticalLayer, ExecutionLayer, DecisionCoordinator
from tradingagents.utils.dashboard import PortfolioMonitor, PerformanceTracker, RiskDashboard, AlertSystem
from tradingagents.utils.enhanced_backtest import EnhancedBacktestEngine
logger = logging.getLogger(__name__)

class InstitutionalTradingFramework:

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}
        logger.info('Initializing extension framework...')
        logger.info('  - 初始化宏观指标模块')
        self.yield_curve = YieldCurveAnalyzer(data_source=self.config.get('macro_data_source', 'fred'))
        self.credit_spreads = CreditSpreadAnalyzer(data_source=self.config.get('macro_data_source', 'fred'))
        self.volatility = VolatilityAnalyzer()
        self.fund_flows = FundFlowAnalyzer()
        self.economic = EconomicIndicators(data_source=self.config.get('macro_data_source', 'fred'))
        self.regime_detector = RegimeDetector(config=self.config)
        logger.info('  - 初始化风险框架')
        self.risk_budget = RiskBudgetManager(config=self.config)
        self.risk_attributor = RiskAttributor(config=self.config)
        self.stress_tester = StressTester(config=self.config)
        self.dynamic_risk = DynamicRiskManager(config=self.config)
        logger.info('  - 初始化因子分析')
        self.factor_analyst = FactorAnalyst(config=self.config)
        logger.info('  - 初始化决策层')
        self.decision_coordinator = DecisionCoordinator(config=self.config, regime_detector=self.regime_detector, factor_analyst=self.factor_analyst, risk_budget_manager=self.risk_budget, dynamic_risk_manager=self.dynamic_risk)
        logger.info('  - 初始化监控仪表盘')
        self.portfolio_monitor = PortfolioMonitor(config=self.config)
        self.performance_tracker = PerformanceTracker(config=self.config)
        self.risk_dashboard = RiskDashboard(config=self.config)
        self.alert_system = AlertSystem(config=self.config)
        logger.info('  - 初始化增强回测引擎')
        self.backtest_engine = EnhancedBacktestEngine(config=self.config)
        logger.info('Extension framework ready.')

    def analyze_macro_environment(self, as_of_date: str=None) -> Dict[str, Any]:
        logger.info(f"分析宏观环境: {as_of_date or 'today'}")
        return {'regime': self.regime_detector.detect_regime(as_of_date), 'yield_curve': self.yield_curve.get_growth_signal(as_of_date), 'credit': self.credit_spreads.get_risk_appetite_signal(as_of_date), 'volatility': self.volatility.get_volatility_regime(as_of_date), 'fund_flows': self.fund_flows.get_flow_signal(as_of_date), 'economic_growth': self.economic.get_growth_signal(as_of_date), 'economic_inflation': self.economic.get_inflation_signal(as_of_date)}

    def make_investment_decision(self, market_data: Dict=None, current_positions: Dict[str, float]=None, current_prices: Dict[str, float]=None, total_value: float=100000, volume_data: Dict[str, float]=None, as_of_date: str=None, force_all: bool=False) -> Dict[str, Any]:
        return self.decision_coordinator.run_full_cycle(market_data=market_data, current_positions=current_positions, current_prices=current_prices, total_value=total_value, volume_data=volume_data, as_of_date=as_of_date, force_all=force_all)

    def run_stress_test(self, weights: Dict[str, float], scenarios: str='all') -> Dict[str, Any]:
        if scenarios == 'all':
            return self.stress_tester.run_all_stress_tests(weights)
        elif scenarios == 'historical':
            results = {}
            for name in self.stress_tester.HISTORICAL_SCENARIOS:
                results[name] = self.stress_tester.run_historical_stress_test(weights, name)
            return results
        elif scenarios == 'hypothetical':
            results = {}
            for name in self.stress_tester.HYPOTHETICAL_SCENARIOS:
                results[name] = self.stress_tester.run_hypothetical_stress_test(weights, name)
            return results
        elif scenarios in self.stress_tester.HISTORICAL_SCENARIOS:
            return self.stress_tester.run_historical_stress_test(weights, scenarios)
        elif scenarios in self.stress_tester.HYPOTHETICAL_SCENARIOS:
            return self.stress_tester.run_hypothetical_stress_test(weights, scenarios)
        else:
            raise ValueError(f'Unknown scenario: {scenarios}')

    def analyze_risk(self, weights: Dict[str, float], returns_data, covariance_matrix=None) -> Dict[str, Any]:
        return self.risk_dashboard.calculate_current_risk(weights, returns_data, covariance_matrix)

    def get_factor_exposure(self, ticker: str, price_data, fundamental_data: Dict=None, market_data=None) -> Dict[str, Any]:
        return self.factor_analyst.calculate_stock_factors(ticker, price_data, fundamental_data, market_data)

    def get_dashboard_summary(self) -> Dict[str, Any]:
        return {'portfolio': self.portfolio_monitor.get_summary(), 'performance': self.performance_tracker.get_performance_metrics(), 'risk': self.risk_dashboard.get_risk_summary(), 'alerts': self.alert_system.get_alert_summary(), 'decision_state': self.decision_coordinator.get_current_state()}

    def update_portfolio(self, positions: Dict[str, float], prices: Dict[str, float], timestamp: datetime=None) -> None:
        self.portfolio_monitor.update_positions(positions, prices, timestamp)
        value = self.portfolio_monitor.get_portfolio_value()
        self.performance_tracker.add_daily_return(timestamp or datetime.now(), value)
        self.dynamic_risk.update_nav(value)

    def check_alerts(self, current_drawdown: float=None, var_value: float=None, vix_level: float=None, regime_change: bool=False, regime_name: str=None) -> None:
        self.alert_system.check_risk_alerts(current_drawdown=current_drawdown, var_value=var_value)
        self.alert_system.check_market_alerts(vix_level=vix_level, regime_change=regime_change, regime_name=regime_name)

    def generate_report(self) -> str:
        summary = self.get_dashboard_summary()
        risk_report = self.risk_dashboard.generate_risk_report()
        report = f"\n╔══════════════════════════════════════════════════════════════╗\n║           TradingAgents Extension — Summary Report           ║\n╚══════════════════════════════════════════════════════════════╝\n\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n1. 组合概览\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n总价值: ${summary['portfolio'].get('total_value', 0):,.2f}\n总收益: {summary['portfolio'].get('total_return', 0):.2%}\n持仓数: {summary['portfolio'].get('num_positions', 0)}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n2. 绩效指标\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n年化收益: {summary['performance'].get('annual_return', 0):.2%}\n年化波动: {summary['performance'].get('annual_volatility', 0):.2%}\n夏普比率: {summary['performance'].get('sharpe_ratio', 0):.2f}\n最大回撤: {summary['performance'].get('max_drawdown', 0):.2%}\n胜率: {summary['performance'].get('win_rate', 0):.2%}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n3. 风险状态\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{risk_report}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n4. 警报摘要\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n活跃警报: {summary['alerts'].get('active_alerts', 0)} 条\n  - 严重: {summary['alerts'].get('by_level', {}).get('CRITICAL', 0)}\n  - 警告: {summary['alerts'].get('by_level', {}).get('WARNING', 0)}\n  - 信息: {summary['alerts'].get('by_level', {}).get('INFO', 0)}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        return report

def create_institutional_framework(config: Dict[str, Any]=None) -> InstitutionalTradingFramework:
    return InstitutionalTradingFramework(config)
