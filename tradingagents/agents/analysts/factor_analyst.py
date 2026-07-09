import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class FactorAnalyst:
    FACTORS = ['market', 'size', 'value', 'momentum', 'quality', 'low_vol']
    QUADRANT_FACTOR_TILT = {'rising_growth_rising_inflation': {'description': '再通胀环境', 'favored_factors': ['momentum', 'value'], 'unfavored_factors': ['quality', 'low_vol'], 'tilts': {'market': 0.1, 'size': 0.05, 'value': 0.15, 'momentum': 0.15, 'quality': -0.1, 'low_vol': -0.15}}, 'rising_growth_falling_inflation': {'description': '金发女郎环境', 'favored_factors': ['momentum', 'quality'], 'unfavored_factors': ['low_vol'], 'tilts': {'market': 0.15, 'size': 0.1, 'value': 0.0, 'momentum': 0.2, 'quality': 0.1, 'low_vol': -0.1}}, 'falling_growth_rising_inflation': {'description': '滞胀环境', 'favored_factors': ['quality', 'low_vol'], 'unfavored_factors': ['size', 'momentum'], 'tilts': {'market': -0.15, 'size': -0.15, 'value': 0.05, 'momentum': -0.1, 'quality': 0.2, 'low_vol': 0.2}}, 'falling_growth_falling_inflation': {'description': '通缩/衰退环境', 'favored_factors': ['quality', 'low_vol', 'value'], 'unfavored_factors': ['size', 'market'], 'tilts': {'market': -0.1, 'size': -0.1, 'value': 0.1, 'momentum': -0.05, 'quality': 0.15, 'low_vol': 0.15}}}

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}
        self._cache = {}

    def calculate_stock_factors(self, ticker: str, price_data: pd.DataFrame, fundamental_data: Dict[str, Any]=None, market_data: pd.DataFrame=None, as_of_date: str=None) -> Dict[str, Any]:
        if price_data.empty:
            return self._empty_factor_result(ticker)
        factors = {}
        percentiles = {}
        beta = self._calculate_beta(price_data, market_data)
        factors['market'] = beta
        percentiles['market'] = self._beta_to_percentile(beta)
        size_score = self._calculate_size_score(price_data, fundamental_data)
        factors['size'] = size_score
        percentiles['size'] = (size_score + 1) / 2
        value_score = self._calculate_value_score(fundamental_data)
        factors['value'] = value_score
        percentiles['value'] = (value_score + 1) / 2
        momentum_score = self._calculate_momentum_score(price_data)
        factors['momentum'] = momentum_score
        percentiles['momentum'] = (momentum_score + 1) / 2
        quality_score = self._calculate_quality_score(fundamental_data)
        factors['quality'] = quality_score
        percentiles['quality'] = (quality_score + 1) / 2
        low_vol_score = self._calculate_low_vol_score(price_data)
        factors['low_vol'] = low_vol_score
        percentiles['low_vol'] = (low_vol_score + 1) / 2
        composite = np.mean([factors['value'] * 0.2, factors['momentum'] * 0.25, factors['quality'] * 0.25, factors['low_vol'] * 0.15, factors['size'] * 0.15])
        style = self._classify_style(factors)
        return {'ticker': ticker, 'as_of_date': as_of_date or datetime.now().strftime('%Y-%m-%d'), 'factors': factors, 'factor_percentiles': percentiles, 'composite_score': float(composite), 'style_classification': style}

    def _calculate_beta(self, price_data: pd.DataFrame, market_data: pd.DataFrame=None, window: int=252) -> float:
        if price_data.empty:
            return 1.0
        stock_returns = price_data['Close'].pct_change().dropna()
        if market_data is not None and (not market_data.empty):
            market_returns = market_data['Close'].pct_change().dropna()
            common_idx = stock_returns.index.intersection(market_returns.index)
            if len(common_idx) < 60:
                return 1.0
            s_ret = stock_returns.loc[common_idx].iloc[-window:]
            m_ret = market_returns.loc[common_idx].iloc[-window:]
            cov = np.cov(s_ret, m_ret)[0, 1]
            var_m = np.var(m_ret)
            if var_m > 0:
                beta = cov / var_m
            else:
                beta = 1.0
        else:
            vol = stock_returns.iloc[-min(window, len(stock_returns)):].std() * np.sqrt(252)
            beta = vol / 0.16
        return float(np.clip(beta, 0.0, 3.0))

    def _beta_to_percentile(self, beta: float) -> float:
        if beta < 0.5:
            return 0.1
        elif beta < 0.7:
            return 0.25
        elif beta < 0.9:
            return 0.4
        elif beta < 1.1:
            return 0.5
        elif beta < 1.3:
            return 0.65
        elif beta < 1.5:
            return 0.8
        else:
            return 0.95

    def _calculate_size_score(self, price_data: pd.DataFrame, fundamental_data: Dict=None) -> float:
        if fundamental_data and 'market_cap' in fundamental_data:
            market_cap = fundamental_data['market_cap']
            if market_cap > 100000000000.0:
                return -0.8
            elif market_cap > 50000000000.0:
                return -0.5
            elif market_cap > 10000000000.0:
                return -0.2
            elif market_cap > 2000000000.0:
                return 0.2
            elif market_cap > 500000000.0:
                return 0.5
            else:
                return 0.8
        if not price_data.empty and 'Volume' in price_data.columns:
            avg_volume = price_data['Volume'].iloc[-20:].mean()
            avg_price = price_data['Close'].iloc[-20:].mean()
            daily_turnover = avg_volume * avg_price
            if daily_turnover > 1000000000.0:
                return -0.6
            elif daily_turnover > 100000000.0:
                return -0.2
            elif daily_turnover > 10000000.0:
                return 0.2
            else:
                return 0.6
        return 0.0

    def _calculate_value_score(self, fundamental_data: Dict=None) -> float:
        if not fundamental_data:
            return 0.0
        scores = []
        pe = fundamental_data.get('pe_ratio')
        if pe and pe > 0:
            if pe < 10:
                scores.append(0.8)
            elif pe < 15:
                scores.append(0.4)
            elif pe < 20:
                scores.append(0.0)
            elif pe < 30:
                scores.append(-0.4)
            else:
                scores.append(-0.8)
        pb = fundamental_data.get('pb_ratio')
        if pb and pb > 0:
            if pb < 1:
                scores.append(0.8)
            elif pb < 2:
                scores.append(0.4)
            elif pb < 3:
                scores.append(0.0)
            elif pb < 5:
                scores.append(-0.4)
            else:
                scores.append(-0.8)
        ps = fundamental_data.get('ps_ratio')
        if ps and ps > 0:
            if ps < 1:
                scores.append(0.6)
            elif ps < 3:
                scores.append(0.2)
            elif ps < 5:
                scores.append(-0.2)
            else:
                scores.append(-0.6)
        div_yield = fundamental_data.get('dividend_yield')
        if div_yield and div_yield > 0:
            if div_yield > 0.04:
                scores.append(0.6)
            elif div_yield > 0.02:
                scores.append(0.3)
            else:
                scores.append(0.0)
        if scores:
            return float(np.mean(scores))
        return 0.0

    def _calculate_momentum_score(self, price_data: pd.DataFrame, long_period: int=252, short_period: int=21) -> float:
        if price_data.empty or len(price_data) < long_period:
            return 0.0
        close = price_data['Close']
        if len(close) > long_period:
            ret_12m = close.iloc[-short_period] / close.iloc[-long_period] - 1
        else:
            ret_12m = close.iloc[-1] / close.iloc[0] - 1
        score = np.clip(ret_12m / 0.5, -1, 1)
        return float(score)

    def _calculate_quality_score(self, fundamental_data: Dict=None) -> float:
        if not fundamental_data:
            return 0.0
        scores = []
        roe = fundamental_data.get('roe')
        if roe is not None:
            if roe > 0.25:
                scores.append(0.8)
            elif roe > 0.15:
                scores.append(0.4)
            elif roe > 0.08:
                scores.append(0.0)
            elif roe > 0:
                scores.append(-0.4)
            else:
                scores.append(-0.8)
        debt_equity = fundamental_data.get('debt_to_equity')
        if debt_equity is not None:
            if debt_equity < 0.3:
                scores.append(0.6)
            elif debt_equity < 0.5:
                scores.append(0.3)
            elif debt_equity < 1.0:
                scores.append(0.0)
            elif debt_equity < 2.0:
                scores.append(-0.3)
            else:
                scores.append(-0.6)
        margin = fundamental_data.get('profit_margin')
        if margin is not None:
            if margin > 0.2:
                scores.append(0.7)
            elif margin > 0.1:
                scores.append(0.3)
            elif margin > 0.05:
                scores.append(0.0)
            elif margin > 0:
                scores.append(-0.3)
            else:
                scores.append(-0.7)
        earnings_growth = fundamental_data.get('earnings_growth')
        if earnings_growth is not None:
            if earnings_growth > 0.15:
                scores.append(0.4)
            elif earnings_growth > 0.05:
                scores.append(0.2)
            elif earnings_growth > -0.05:
                scores.append(0.0)
            else:
                scores.append(-0.4)
        if scores:
            return float(np.mean(scores))
        return 0.0

    def _calculate_low_vol_score(self, price_data: pd.DataFrame, window: int=252) -> float:
        if price_data.empty or len(price_data) < 60:
            return 0.0
        returns = price_data['Close'].pct_change().dropna()
        vol = returns.iloc[-min(window, len(returns)):].std() * np.sqrt(252)
        if vol < 0.1:
            return 0.8
        elif vol < 0.15:
            return 0.4
        elif vol < 0.2:
            return 0.0
        elif vol < 0.3:
            return -0.4
        else:
            return -0.8

    def _classify_style(self, factors: Dict[str, float]) -> str:
        value = factors.get('value', 0)
        momentum = factors.get('momentum', 0)
        quality = factors.get('quality', 0)
        size = factors.get('size', 0)
        low_vol = factors.get('low_vol', 0)
        styles = []
        if value > 0.3:
            styles.append('Value')
        elif value < -0.3:
            styles.append('Growth')
        if momentum > 0.3:
            styles.append('Momentum')
        if quality > 0.3:
            styles.append('Quality')
        if size > 0.3:
            styles.append('SmallCap')
        elif size < -0.3:
            styles.append('LargeCap')
        if low_vol > 0.3:
            styles.append('LowVol')
        elif low_vol < -0.3:
            styles.append('HighBeta')
        if not styles:
            return 'Blend'
        return '/'.join(styles)

    def calculate_portfolio_factor_exposure(self, holdings: Dict[str, float], stock_factors: Dict[str, Dict]) -> Dict[str, Any]:
        portfolio_factors = {f: 0.0 for f in self.FACTORS}
        style_counts = {}
        total_weight = sum(holdings.values())
        if total_weight == 0:
            return {'portfolio_factors': portfolio_factors, 'factor_tilts': {}, 'dominant_factors': [], 'style_breakdown': {}}
        for ticker, weight in holdings.items():
            normalized_weight = weight / total_weight
            if ticker in stock_factors:
                factors = stock_factors[ticker].get('factors', {})
                for factor in self.FACTORS:
                    portfolio_factors[factor] += normalized_weight * factors.get(factor, 0)
                style = stock_factors[ticker].get('style_classification', 'Blend')
                style_counts[style] = style_counts.get(style, 0) + normalized_weight
        sorted_factors = sorted(portfolio_factors.items(), key=lambda x: abs(x[1]), reverse=True)
        dominant_factors = [f[0] for f in sorted_factors[:3] if abs(f[1]) > 0.15]
        return {'portfolio_factors': portfolio_factors, 'factor_tilts': {f: v for f, v in sorted_factors}, 'dominant_factors': dominant_factors, 'style_breakdown': style_counts}

    def get_factor_tilt_recommendation(self, current_quadrant: str, current_factors: Dict[str, float]=None) -> Dict[str, Any]:
        if current_quadrant not in self.QUADRANT_FACTOR_TILT:
            return {'recommended_tilts': {f: 0 for f in self.FACTORS}, 'favored_factors': [], 'unfavored_factors': [], 'adjustment_needed': {}, 'reasoning': '未知象限，维持中性因子配置'}
        tilt_config = self.QUADRANT_FACTOR_TILT[current_quadrant]
        result = {'recommended_tilts': tilt_config['tilts'], 'favored_factors': tilt_config['favored_factors'], 'unfavored_factors': tilt_config['unfavored_factors'], 'adjustment_needed': {}, 'reasoning': f"在{tilt_config['description']}下，建议超配{', '.join(tilt_config['favored_factors'])}，低配{', '.join(tilt_config['unfavored_factors'])}"}
        if current_factors:
            for factor, target_tilt in tilt_config['tilts'].items():
                current_exposure = current_factors.get(factor, 0)
                diff = target_tilt - current_exposure
                if abs(diff) > 0.1:
                    result['adjustment_needed'][factor] = {'current': current_exposure, 'target': target_tilt, 'change': diff}
        return result

    def screen_stocks_by_factors(self, stock_factors_list: List[Dict], target_factors: Dict[str, float]=None, min_score: float=0.0) -> List[Dict]:
        if target_factors is None:
            target_factors = {f: 1.0 / len(self.FACTORS) for f in self.FACTORS}
        scored_stocks = []
        for stock in stock_factors_list:
            factors = stock.get('factors', {})
            weighted_score = sum((factors.get(f, 0) * target_factors.get(f, 0) for f in self.FACTORS))
            if weighted_score >= min_score:
                scored_stocks.append({**stock, 'weighted_score': weighted_score})
        return sorted(scored_stocks, key=lambda x: x['weighted_score'], reverse=True)

    def _empty_factor_result(self, ticker: str) -> Dict[str, Any]:
        return {'ticker': ticker, 'as_of_date': datetime.now().strftime('%Y-%m-%d'), 'factors': {f: 0.0 for f in self.FACTORS}, 'factor_percentiles': {f: 0.5 for f in self.FACTORS}, 'composite_score': 0.0, 'style_classification': 'Unknown'}
