#!/usr/bin/env python
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from datetime import datetime
import pandas as pd
import numpy as np
from tradingagents.institutional_framework import InstitutionalTradingFramework
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
console = Console()

def demo_macro_analysis():
    console.print(Panel('[bold cyan]1. 宏观环境分析[/bold cyan]'))
    framework = InstitutionalTradingFramework()
    macro = framework.analyze_macro_environment()
    table = Table(title='宏观环境判断', box=box.ROUNDED)
    table.add_column('指标', style='cyan')
    table.add_column('信号', style='yellow')
    table.add_column('置信度', style='green')
    regime = macro['regime']
    table.add_row('宏观象限', regime.get('quadrant_name', 'N/A'), f"{regime.get('confidence', 0):.0%}")
    table.add_row('增长信号', regime.get('growth_signal', 'N/A'), '')
    table.add_row('通胀信号', regime.get('inflation_signal', 'N/A'), '')
    table.add_row('风险等级', regime.get('risk_level', 'N/A'), '')
    console.print(table)
    if 'recommendations' in regime:
        console.print(f'\n[bold]配置建议:[/bold]')
        console.print(regime['recommendations'].get('text', ''))
    return framework

def demo_stress_test(framework):
    console.print(Panel('[bold cyan]2. 压力测试[/bold cyan]'))
    weights = {'SPY': 0.3, 'QQQ': 0.15, 'TLT': 0.25, 'GLD': 0.15, 'DBC': 0.1, 'CASH': 0.05}
    results = framework.run_stress_test(weights, scenarios='all')
    table = Table(title='压力测试结果', box=box.ROUNDED)
    table.add_column('情景', style='cyan')
    table.add_column('组合影响', style='red')
    table.add_column('严重程度', style='yellow')
    for name, result in results.get('historical', {}).items():
        table.add_row(name, f"{result['portfolio_impact']:.1%}", result['severity'])
    for name, result in results.get('hypothetical', {}).items():
        table.add_row(name, f"{result['portfolio_impact']:.1%}", result['severity'])
    console.print(table)
    worst = results.get('worst_case', {})
    console.print(f"\n[bold red]最坏情况:[/bold red] {worst.get('scenario_name', 'N/A')} ({worst.get('portfolio_impact', 0):.1%})")

def demo_decision_cycle(framework):
    console.print(Panel('[bold cyan]3. 三层决策演示[/bold cyan]'))
    market_data = {'realized_vol': 0.18, 'current_drawdown': 0.05}
    current_positions = {'SPY': 300, 'TLT': 200, 'GLD': 100}
    current_prices = {'SPY': 500.0, 'TLT': 90.0, 'GLD': 200.0, 'QQQ': 450.0, 'DBC': 25.0}
    total_value = sum((current_positions.get(t, 0) * current_prices.get(t, 0) for t in current_positions))
    decision = framework.make_investment_decision(market_data=market_data, current_positions=current_positions, current_prices=current_prices, total_value=total_value, force_all=True)
    strategic = decision.get('strategic', {})
    console.print('\n[bold]战略层决策:[/bold]')
    if 'allocation' in strategic:
        alloc_table = Table(box=box.SIMPLE)
        alloc_table.add_column('资产类别')
        alloc_table.add_column('目标权重')
        for asset, weight in strategic['allocation'].items():
            alloc_table.add_row(asset, f'{weight:.1%}')
        console.print(alloc_table)
    execution = decision.get('execution', {})
    console.print('\n[bold]执行层决策:[/bold]')
    console.print(f"  订单数量: {execution.get('order_count', 0)}")
    console.print(f"  预估成本: ${execution.get('estimated_cost', 0):.2f}")
    summary = decision.get('summary', {})
    console.print(f"\n[bold]行动摘要:[/bold] {summary.get('action_summary', 'N/A')}")

def demo_risk_monitoring(framework):
    console.print(Panel('[bold cyan]4. 风险监控[/bold cyan]'))
    weights = {'SPY': 0.3, 'QQQ': 0.2, 'TLT': 0.25, 'GLD': 0.15, 'DBC': 0.1}
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
    returns = pd.DataFrame(np.random.randn(252, 5) * 0.01, index=dates, columns=['SPY', 'QQQ', 'TLT', 'GLD', 'DBC'])
    risk = framework.analyze_risk(weights, returns)
    table = Table(title='风险指标', box=box.ROUNDED)
    table.add_column('指标', style='cyan')
    table.add_column('数值', style='yellow')
    table.add_row('风险等级', risk.get('risk_level', 'N/A'))
    table.add_row('VaR (95%)', f"{risk['var_cvar'].get('var_95', 0):.2%}")
    table.add_row('CVaR (95%)', f"{risk['var_cvar'].get('cvar_95', 0):.2%}")
    table.add_row('组合波动率', f"{risk.get('portfolio_volatility', 0):.2%}")
    table.add_row('最大权重', f"{risk['concentration'].get('max_weight', 0):.2%}")
    table.add_row('平均相关性', f"{risk['correlation_risk'].get('avg_correlation', 0):.2f}")
    console.print(table)
    breaches = risk.get('breaches', [])
    if breaches:
        console.print(f'\n[bold red]风险突破: {len(breaches)} 项[/bold red]')
        for breach in breaches:
            console.print(f"  - {breach['type']}: {breach['actual']:.2%} > {breach['limit']:.2%}")
    else:
        console.print('\n[bold green]无风险突破[/bold green]')

def demo_report(framework):
    console.print(Panel('[bold cyan]5. 综合报告[/bold cyan]'))
    framework.update_portfolio(positions={'SPY': 300, 'TLT': 200, 'GLD': 100}, prices={'SPY': 500, 'TLT': 90, 'GLD': 200})
    report = framework.generate_report()
    console.print(report)

def main():
    console.print(Panel('[bold white on blue]  TradingAgents Extension Demo  [/bold white on blue]', border_style='blue'))
    console.print()
    try:
        framework = demo_macro_analysis()
        console.print()
        demo_stress_test(framework)
        console.print()
        demo_decision_cycle(framework)
        console.print()
        demo_risk_monitoring(framework)
        console.print()
        demo_report(framework)
    except Exception as e:
        console.print(f'[bold red]错误: {e}[/bold red]')
        import traceback
        traceback.print_exc()
if __name__ == '__main__':
    main()
