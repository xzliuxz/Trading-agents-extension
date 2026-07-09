import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class StressTester:
    HISTORICAL_SCENARIOS = {'2008_financial_crisis': {'name': '2008金融危机', 'start': '2008-09-01', 'end': '2008-11-30', 'description': '雷曼兄弟破产引发的全球金融危机', 'asset_shocks': {'equities': -0.4, 'bonds': 0.05, 'gold': 0.1, 'commodities': -0.35}}, '2020_covid_crash': {'name': '2020新冠崩盘', 'start': '2020-02-20', 'end': '2020-03-23', 'description': '新冠疫情引发的快速崩盘', 'asset_shocks': {'equities': -0.34, 'bonds': 0.02, 'gold': -0.05, 'commodities': -0.3}}, '2022_rate_shock': {'name': '2022加息冲击', 'start': '2022-01-01', 'end': '2022-10-31', 'description': '美联储激进加息导致的股债双杀', 'asset_shocks': {'equities': -0.25, 'bonds': -0.2, 'gold': -0.05, 'commodities': 0.15}}, '2011_eurozone_crisis': {'name': '2011欧债危机', 'start': '2011-07-01', 'end': '2011-10-31', 'description': '欧洲主权债务危机', 'asset_shocks': {'equities': -0.2, 'bonds': 0.08, 'gold': 0.15, 'commodities': -0.15}}, '2018_q4_selloff': {'name': '2018Q4抛售', 'start': '2018-10-01', 'end': '2018-12-24', 'description': '美联储鹰派导致的股市下跌', 'asset_shocks': {'equities': -0.2, 'bonds': 0.02, 'gold': 0.05, 'commodities': -0.12}}, '1987_black_monday': {'name': '1987黑色星期一', 'start': '1987-10-01', 'end': '1987-10-31', 'description': '单日跌幅超过20%的崩盘', 'asset_shocks': {'equities': -0.23, 'bonds': 0.03, 'gold': 0.02, 'commodities': -0.1}}}
    HYPOTHETICAL_SCENARIOS = {'rate_shock_200bp': {'name': '利率上升200bp', 'description': '10年期国债收益率上升200个基点', 'asset_shocks': {'equities': -0.15, 'bonds': -0.15, 'gold': -0.03, 'commodities': 0.05}}, 'stagflation': {'name': '滞胀情景', 'description': '经济停滞 + 高通胀', 'asset_shocks': {'equities': -0.25, 'bonds': -0.1, 'gold': 0.2, 'commodities': 0.25}}, 'deflation': {'name': '通缩情景', 'description': '全球通缩 + 经济衰退', 'asset_shocks': {'equities': -0.3, 'bonds': 0.15, 'gold': 0.1, 'commodities': -0.25}}, 'risk_off_panic': {'name': '避险恐慌', 'description': 'VIX飙升到50+的极端避险', 'asset_shocks': {'equities': -0.25, 'bonds': 0.08, 'gold': 0.15, 'commodities': -0.2}}, 'correlation_breakdown': {'name': '相关性崩溃', 'description': '所有资产同时下跌', 'asset_shocks': {'equities': -0.2, 'bonds': -0.05, 'gold': -0.08, 'commodities': -0.15}}, 'usd_crash': {'name': '美元崩盘', 'description': '美元指数暴跌20%', 'asset_shocks': {'equities': -0.05, 'bonds': -0.08, 'gold': 0.3, 'commodities': 0.2}}}

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}

    def run_historical_stress_test(self, weights: Dict[str, float], scenario_name: str, asset_mapping: Dict[str, str]=None) -> Dict[str, Any]:
        if scenario_name not in self.HISTORICAL_SCENARIOS:
            raise ValueError(f'Unknown scenario: {scenario_name}')
        scenario = self.HISTORICAL_SCENARIOS[scenario_name]
        return self._calculate_scenario_impact(weights, scenario, asset_mapping)

    def run_hypothetical_stress_test(self, weights: Dict[str, float], scenario_name: str, asset_mapping: Dict[str, str]=None) -> Dict[str, Any]:
        if scenario_name not in self.HYPOTHETICAL_SCENARIOS:
            raise ValueError(f'Unknown scenario: {scenario_name}')
        scenario = self.HYPOTHETICAL_SCENARIOS[scenario_name]
        return self._calculate_scenario_impact(weights, scenario, asset_mapping)

    def run_custom_stress_test(self, weights: Dict[str, float], asset_shocks: Dict[str, float], scenario_name: str='Custom', asset_mapping: Dict[str, str]=None) -> Dict[str, Any]:
        scenario = {'name': scenario_name, 'description': 'Custom scenario', 'asset_shocks': asset_shocks}
        return self._calculate_scenario_impact(weights, scenario, asset_mapping)

    def _calculate_scenario_impact(self, weights: Dict[str, float], scenario: Dict, asset_mapping: Dict[str, str]=None) -> Dict[str, Any]:
        asset_shocks = scenario['asset_shocks']
        if asset_mapping is None:
            asset_mapping = self._default_asset_mapping()
        asset_impacts = {}
        portfolio_impact = 0.0
        for asset, weight in weights.items():
            asset_class = asset_mapping.get(asset, 'equities')
            shock = asset_shocks.get(asset_class, 0.0)
            impact = weight * shock
            asset_impacts[asset] = {'weight': weight, 'asset_class': asset_class, 'shock': shock, 'impact': impact}
            portfolio_impact += impact
        var_threshold = self.config.get('var_threshold', -0.1)
        var_breach = portfolio_impact < var_threshold
        return {'scenario_name': scenario['name'], 'scenario_description': scenario.get('description', ''), 'portfolio_impact': float(portfolio_impact), 'asset_impacts': asset_impacts, 'var_breach': var_breach, 'var_threshold': var_threshold, 'severity': self._classify_severity(portfolio_impact)}

    def _classify_severity(self, impact: float) -> str:
        if impact < -0.25:
            return 'CATASTROPHIC'
        elif impact < -0.15:
            return 'SEVERE'
        elif impact < -0.08:
            return 'MODERATE'
        elif impact < -0.03:
            return 'MILD'
        else:
            return 'MINIMAL'

    def run_all_stress_tests(self, weights: Dict[str, float], asset_mapping: Dict[str, str]=None) -> Dict[str, Any]:
        results = {'historical': {}, 'hypothetical': {}}
        for name in self.HISTORICAL_SCENARIOS:
            results['historical'][name] = self.run_historical_stress_test(weights, name, asset_mapping)
        for name in self.HYPOTHETICAL_SCENARIOS:
            results['hypothetical'][name] = self.run_hypothetical_stress_test(weights, name, asset_mapping)
        all_scenarios = list(results['historical'].values()) + list(results['hypothetical'].values())
        worst = min(all_scenarios, key=lambda x: x['portfolio_impact'])
        results['worst_case'] = worst
        impacts = [s['portfolio_impact'] for s in all_scenarios]
        results['summary'] = {'scenarios_tested': len(all_scenarios), 'average_impact': float(np.mean(impacts)), 'median_impact': float(np.median(impacts)), 'worst_impact': float(min(impacts)), 'best_impact': float(max(impacts)), 'var_breaches': sum((1 for s in all_scenarios if s['var_breach']))}
        return results

    def run_monte_carlo_stress(self, weights: Dict[str, float], covariance_matrix: pd.DataFrame, n_simulations: int=10000, horizon_days: int=21, confidence_levels: List[float]=[0.95, 0.99]) -> Dict[str, Any]:
        available = [k for k in weights.keys() if k in covariance_matrix.columns]
        if not available:
            return {'var': {}, 'cvar': {}, 'percentiles': {}, 'worst_simulation': 0.0, 'distribution': np.array([])}
        w = np.array([weights[k] for k in available])
        cov = covariance_matrix.loc[available, available].values
        mean_daily = np.zeros(len(available))
        cov_horizon = cov * horizon_days
        np.random.seed(42)
        try:
            L = np.linalg.cholesky(cov_horizon)
            z = np.random.randn(n_simulations, len(available))
            asset_returns = z @ L.T
        except np.linalg.LinAlgError:
            vols = np.sqrt(np.diag(cov_horizon))
            asset_returns = np.random.randn(n_simulations, len(available)) * vols
        portfolio_returns = asset_returns @ w
        results = {'var': {}, 'cvar': {}, 'percentiles': {}}
        for conf in confidence_levels:
            var_percentile = 100 * (1 - conf)
            var = -np.percentile(portfolio_returns, var_percentile)
            losses = -portfolio_returns
            cvar = losses[losses >= var].mean() if (losses >= var).sum() > 0 else var
            results['var'][f'{int(conf * 100)}%'] = float(var)
            results['cvar'][f'{int(conf * 100)}%'] = float(cvar)
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            results['percentiles'][f'{p}%'] = float(np.percentile(portfolio_returns, p))
        results['worst_simulation'] = float(portfolio_returns.min())
        results['best_simulation'] = float(portfolio_returns.max())
        results['mean'] = float(portfolio_returns.mean())
        results['std'] = float(portfolio_returns.std())
        results['distribution'] = portfolio_returns
        results['horizon_days'] = horizon_days
        results['n_simulations'] = n_simulations
        return results

    def reverse_stress_test(self, weights: Dict[str, float], target_loss: float, asset_volatilities: Dict[str, float]) -> Dict[str, Any]:
        if target_loss <= 0:
            return {'required_shocks': {}, 'plausibility': 'N/A', 'scenarios': []}
        total_weight = sum(weights.values())
        if total_weight > 0:
            uniform_shock = -target_loss / total_weight
        else:
            uniform_shock = -target_loss
        required_shocks = {asset: uniform_shock for asset in weights}
        plausibility_scores = []
        for asset, shock in required_shocks.items():
            vol = asset_volatilities.get(asset, 0.02)
            monthly_vol = vol * np.sqrt(21)
            z_score = abs(shock) / monthly_vol if monthly_vol > 0 else 10
            plausibility_scores.append(z_score)
        avg_z = np.mean(plausibility_scores)
        if avg_z < 2:
            plausibility = 'VERY_LIKELY'
        elif avg_z < 3:
            plausibility = 'PLAUSIBLE'
        elif avg_z < 4:
            plausibility = 'UNLIKELY'
        else:
            plausibility = 'VERY_UNLIKELY'
        matching_scenarios = []
        for name, scenario in {**self.HISTORICAL_SCENARIOS, **self.HYPOTHETICAL_SCENARIOS}.items():
            scenario_loss = sum((weights.get(asset, 0) * scenario['asset_shocks'].get(self._get_asset_class(asset), 0) for asset in weights))
            if abs(scenario_loss) >= target_loss * 0.7:
                matching_scenarios.append({'name': name, 'expected_loss': abs(scenario_loss), 'match_ratio': abs(scenario_loss) / target_loss})
        return {'target_loss': target_loss, 'required_shocks': required_shocks, 'plausibility': plausibility, 'avg_z_score': float(avg_z), 'matching_scenarios': sorted(matching_scenarios, key=lambda x: x['match_ratio'], reverse=True)[:5]}

    def _default_asset_mapping(self) -> Dict[str, str]:
        return {'SPY': 'equities', 'QQQ': 'equities', 'IWM': 'equities', 'AAPL': 'equities', 'MSFT': 'equities', 'NVDA': 'equities', 'EFA': 'equities', 'EEM': 'equities', 'VWO': 'equities', 'TLT': 'bonds', 'IEF': 'bonds', 'LQD': 'bonds', 'HYG': 'bonds', 'AGG': 'bonds', 'BND': 'bonds', 'GLD': 'gold', 'IAU': 'gold', 'SGLN.L': 'gold', 'DBC': 'commodities', 'GSG': 'commodities', 'USO': 'commodities', 'VUSA.L': 'equities', 'ISF.L': 'equities', 'EQQQ.L': 'equities', 'IUHC.L': 'equities', 'IUKD.L': 'equities', 'VHYL.L': 'equities', 'VWRL.L': 'equities'}

    def _get_asset_class(self, asset: str) -> str:
        mapping = self._default_asset_mapping()
        return mapping.get(asset, 'equities')
