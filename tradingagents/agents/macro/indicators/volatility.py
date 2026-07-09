import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class VolatilityAnalyzer:
    VOLATILITY_TICKERS = {'vix': '^VIX', 'vix_3m': '^VIX3M', 'vvix': '^VVIX', 'gvz': '^GVZ', 'ovx': '^OVX'}
    ASSET_TICKERS = {'spy': 'SPY', 'tlt': 'TLT', 'gld': 'GLD', 'eem': 'EEM'}

    def __init__(self):
        self._cache = {}
        self._cache_expiry = {}
        self._cache_duration = timedelta(hours=1)

    def get_volatility_data(self, as_of_date: str=None, lookback_days: int=252) -> pd.DataFrame:
        if as_of_date:
            end_date = datetime.strptime(as_of_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)
        cache_key = f'vol_{start_date.date()}_{end_date.date()}'
        if cache_key in self._cache:
            if datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
                return self._cache[cache_key]
        df = self._fetch_data(start_date, end_date)
        if not df.empty:
            self._cache[cache_key] = df
            self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
        return df

    def _fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            import yfinance as yf
            all_tickers = list(self.VOLATILITY_TICKERS.values()) + list(self.ASSET_TICKERS.values())
            data = yf.download(all_tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False, threads=False, auto_adjust=True)
            if data.empty:
                return pd.DataFrame()
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close'].copy()
            else:
                prices = data[['Close']].copy()
            vol_rename = {v: k for k, v in self.VOLATILITY_TICKERS.items()}
            asset_rename = {v: k for k, v in self.ASSET_TICKERS.items()}
            rename_map = {**vol_rename, **asset_rename}
            prices = prices.rename(columns=rename_map)
            return prices.ffill()
        except Exception as e:
            logger.error(f'Volatility data fetch error: {e}')
            return pd.DataFrame()

    def calculate_vix_metrics(self, as_of_date: str=None) -> Dict[str, Any]:
        df = self.get_volatility_data(as_of_date)
        if df.empty or 'vix' not in df.columns:
            return self._empty_vix_result()
        vix = df['vix'].dropna()
        if len(vix) < 20:
            return self._empty_vix_result()
        current_vix = vix.iloc[-1]
        percentile = (vix < current_vix).mean()
        ma_20 = vix.iloc[-20:].mean()
        std = vix.std()
        mean = vix.mean()
        z_score = (current_vix - mean) / std if std > 0 else 0
        if 'vix_3m' in df.columns:
            vix_3m = df['vix_3m'].dropna()
            if len(vix_3m) > 0:
                current_vix_3m = vix_3m.iloc[-1]
                term_spread = current_vix_3m - current_vix
                term_structure = 'CONTANGO' if term_spread > 0 else 'BACKWARDATION'
            else:
                term_spread = 0
                term_structure = 'UNKNOWN'
        else:
            term_spread = 0
            term_structure = 'UNKNOWN'
        signal = self._interpret_vix_signal(current_vix, percentile, z_score, term_structure)
        return {'vix': float(current_vix), 'vix_percentile': float(percentile), 'vix_ma_20': float(ma_20), 'vix_zscore': float(z_score), 'term_structure': term_structure, 'term_spread': float(term_spread), 'signal': signal}

    def _interpret_vix_signal(self, vix: float, percentile: float, z_score: float, term_structure: str) -> str:
        if vix > 35:
            return 'PANIC'
        elif vix > 25:
            return 'FEAR'
        elif vix > 20:
            return 'ELEVATED'
        elif vix < 12 and term_structure == 'CONTANGO':
            return 'COMPLACENT'
        elif vix < 15 and percentile < 0.2:
            return 'CALM'
        else:
            return 'NORMAL'

    def calculate_realized_volatility(self, ticker: str='spy', as_of_date: str=None, windows: List[int]=[10, 21, 63]) -> Dict[str, Any]:
        df = self.get_volatility_data(as_of_date)
        if df.empty or ticker not in df.columns:
            return {f'realized_vol_{w}d': 0.0 for w in windows}
        prices = df[ticker].dropna()
        if len(prices) < max(windows) + 5:
            return {f'realized_vol_{w}d': 0.0 for w in windows}
        returns = prices.pct_change().dropna()
        result = {}
        vol_values = []
        for window in windows:
            vol = returns.iloc[-window:].std() * np.sqrt(252)
            result[f'realized_vol_{window}d'] = float(vol)
            vol_values.append(vol)
        if len(vol_values) >= 2:
            if vol_values[0] > vol_values[-1] * 1.2:
                result['vol_trend'] = 'INCREASING'
            elif vol_values[0] < vol_values[-1] * 0.8:
                result['vol_trend'] = 'DECREASING'
            else:
                result['vol_trend'] = 'STABLE'
        else:
            result['vol_trend'] = 'UNKNOWN'
        return result

    def calculate_vol_risk_premium(self, as_of_date: str=None) -> Dict[str, Any]:
        vix_metrics = self.calculate_vix_metrics(as_of_date)
        realized = self.calculate_realized_volatility('spy', as_of_date, [21])
        implied_vol = vix_metrics['vix']
        realized_vol = realized.get('realized_vol_21d', 0) * 100
        if realized_vol > 0:
            vrp = implied_vol - realized_vol
            vrp_ratio = implied_vol / realized_vol
        else:
            vrp = 0
            vrp_ratio = 1
        if vrp > 5:
            signal = 'HIGH_RISK_PREMIUM'
        elif vrp > 2:
            signal = 'NORMAL_PREMIUM'
        elif vrp < -2:
            signal = 'NEGATIVE_PREMIUM'
        else:
            signal = 'NEUTRAL'
        return {'implied_vol': float(implied_vol), 'realized_vol': float(realized_vol), 'vrp': float(vrp), 'vrp_ratio': float(vrp_ratio), 'signal': signal}

    def calculate_cross_asset_vol(self, as_of_date: str=None) -> Dict[str, Any]:
        df = self.get_volatility_data(as_of_date)
        result = {}
        for asset in ['spy', 'tlt', 'gld', 'eem']:
            if asset in df.columns:
                vol_data = self.calculate_realized_volatility(asset, as_of_date, [21])
                result[f'{asset}_vol'] = vol_data.get('realized_vol_21d', 0)
        vol_values = [v for k, v in result.items() if v > 0]
        if vol_values:
            result['avg_vol'] = np.mean(vol_values)
            result['vol_dispersion'] = np.std(vol_values)
        else:
            result['avg_vol'] = 0
            result['vol_dispersion'] = 0
        return result

    def get_volatility_regime(self, as_of_date: str=None) -> Dict[str, Any]:
        vix_metrics = self.calculate_vix_metrics(as_of_date)
        vrp = self.calculate_vol_risk_premium(as_of_date)
        cross_vol = self.calculate_cross_asset_vol(as_of_date)
        score = 0
        if vix_metrics['signal'] in ['PANIC', 'FEAR']:
            score += 3
        elif vix_metrics['signal'] == 'ELEVATED':
            score += 1.5
        elif vix_metrics['signal'] == 'COMPLACENT':
            score -= 0.5
        elif vix_metrics['signal'] == 'CALM':
            score -= 1
        if vix_metrics['term_structure'] == 'BACKWARDATION':
            score += 1
        if vrp['signal'] == 'NEGATIVE_PREMIUM':
            score += 0.5
        elif vrp['signal'] == 'HIGH_RISK_PREMIUM':
            score += 0.5
        if score >= 3:
            regime = 'CRISIS'
            recommendation = '大幅降低风险敞口，增加现金/债券'
        elif score >= 1.5:
            regime = 'HIGH_VOL'
            recommendation = '降低股票仓位，增加对冲'
        elif score >= 0:
            regime = 'NORMAL'
            recommendation = '维持正常配置'
        else:
            regime = 'LOW_VOL'
            recommendation = '可适度增加风险敞口，但警惕自满'
        confidence = min(0.9, 0.5 + abs(score) * 0.1)
        return {'regime': regime, 'confidence': confidence, 'components': {'vix': vix_metrics, 'vrp': vrp, 'cross_asset': cross_vol}, 'recommendation': recommendation}

    def _empty_vix_result(self) -> Dict[str, Any]:
        return {'vix': 20.0, 'vix_percentile': 0.5, 'vix_ma_20': 20.0, 'vix_zscore': 0.0, 'term_structure': 'UNKNOWN', 'term_spread': 0.0, 'signal': 'NO_DATA'}
