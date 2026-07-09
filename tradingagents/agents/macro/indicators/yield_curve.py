import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from .fred_ssl import bootstrap_fred_ssl, is_cert_verify_error
logger = logging.getLogger(__name__)

class YieldCurveAnalyzer:
    FRED_SERIES = {'us_3m': 'DGS3MO', 'us_2y': 'DGS2', 'us_5y': 'DGS5', 'us_10y': 'DGS10', 'us_30y': 'DGS30', 'tips_10y': 'DFII10', 'breakeven_10y': 'T10YIE'}
    YFINANCE_PROXIES = {'us_2y': '^IRX', 'us_10y': '^TNX', 'us_30y': '^TYX'}

    def __init__(self, data_source: str='fred'):
        self.data_source = data_source
        self._cache = {}
        self._cache_expiry = {}
        self._cache_duration = timedelta(hours=6)
        self._fred_ssl_warned = False
        bootstrap_fred_ssl()

    def _warn_fred_ssl_once(self):
        if self._fred_ssl_warned:
            return
        logger.warning('FRED SSL certificate verification failed for yield-curve data; falling back to yfinance proxies.')
        self._fred_ssl_warned = True

    def get_yield_curve_data(self, as_of_date: str=None, lookback_days: int=252) -> pd.DataFrame:
        if as_of_date:
            end_date = datetime.strptime(as_of_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)
        cache_key = f'yield_curve_{start_date.date()}_{end_date.date()}'
        if cache_key in self._cache:
            if datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
                return self._cache[cache_key]
        if self.data_source == 'fred':
            df = self._fetch_from_fred(start_date, end_date)
        else:
            df = self._fetch_from_yfinance(start_date, end_date)
        if not df.empty:
            self._cache[cache_key] = df
            self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
        return df

    def _fetch_from_fred(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            from fredapi import Fred
            api_key = os.getenv('FRED_API_KEY')
            if not api_key:
                logger.warning('FRED_API_KEY not set, falling back to yfinance')
                return self._fetch_from_yfinance(start_date, end_date)
            fred = Fred(api_key=api_key)
            data = {}
            for name, series_id in self.FRED_SERIES.items():
                try:
                    series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                    data[name] = series
                except Exception as e:
                    if is_cert_verify_error(e):
                        self._warn_fred_ssl_once()
                        return self._fetch_from_yfinance(start_date, end_date)
                    logger.warning(f'Failed to fetch {series_id}: {e}')
            if not data:
                return self._fetch_from_yfinance(start_date, end_date)
            df = pd.DataFrame(data)
            df.index.name = 'date'
            return df.ffill().dropna()
        except ImportError:
            logger.warning('fredapi not installed, falling back to yfinance')
            return self._fetch_from_yfinance(start_date, end_date)
        except Exception as e:
            if is_cert_verify_error(e):
                self._warn_fred_ssl_once()
                return self._fetch_from_yfinance(start_date, end_date)
            logger.error(f'FRED fetch error: {e}')
            return self._fetch_from_yfinance(start_date, end_date)

    def _fetch_from_yfinance(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            import yfinance as yf
            tickers = list(self.YFINANCE_PROXIES.values())
            data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False, threads=False, auto_adjust=True)
            if data.empty:
                return pd.DataFrame()
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data[['Close']]
            rename_map = {v: k for k, v in self.YFINANCE_PROXIES.items()}
            prices = prices.rename(columns=rename_map)
            return prices.ffill().dropna()
        except Exception as e:
            logger.error(f'yfinance fetch error: {e}')
            return pd.DataFrame()

    def calculate_term_spread(self, as_of_date: str=None, short_tenor: str='2y', long_tenor: str='10y') -> Dict[str, Any]:
        df = self.get_yield_curve_data(as_of_date, lookback_days=504)
        if df.empty:
            return self._empty_spread_result()
        short_col = f'us_{short_tenor}'
        long_col = f'us_{long_tenor}'
        if short_col not in df.columns or long_col not in df.columns:
            available = [c for c in df.columns if 'us_' in c]
            if len(available) < 2:
                return self._empty_spread_result()
            short_col, long_col = (available[0], available[-1])
        spread = df[long_col] - df[short_col]
        current_spread = spread.iloc[-1]
        percentile = (spread < current_spread).mean()
        change_1m = current_spread - spread.iloc[-21] if len(spread) > 21 else 0
        change_3m = current_spread - spread.iloc[-63] if len(spread) > 63 else 0
        is_inverted = current_spread < 0
        inversion_days = 0
        if is_inverted:
            for i in range(len(spread) - 1, -1, -1):
                if spread.iloc[i] < 0:
                    inversion_days += 1
                else:
                    break
        signal = self._interpret_spread_signal(current_spread, percentile, change_3m, is_inverted, inversion_days)
        return {'spread': float(current_spread), 'spread_percentile': float(percentile), 'spread_change_1m': float(change_1m), 'spread_change_3m': float(change_3m), 'is_inverted': is_inverted, 'inversion_days': inversion_days, 'signal': signal, 'as_of_date': as_of_date or datetime.now().strftime('%Y-%m-%d')}

    def _interpret_spread_signal(self, spread: float, percentile: float, change_3m: float, is_inverted: bool, inversion_days: int) -> str:
        if is_inverted and inversion_days > 20:
            return 'RECESSION_WARNING'
        elif is_inverted:
            return 'CAUTION'
        elif percentile < 0.1:
            return 'VERY_FLAT'
        elif percentile < 0.3 and change_3m < -0.2:
            return 'FLATTENING'
        elif percentile > 0.7 and change_3m > 0.2:
            return 'STEEPENING'
        elif percentile > 0.5:
            return 'NORMAL'
        else:
            return 'NEUTRAL'

    def calculate_real_rate(self, as_of_date: str=None) -> Dict[str, Any]:
        df = self.get_yield_curve_data(as_of_date, lookback_days=504)
        if df.empty or 'tips_10y' not in df.columns:
            if 'us_10y' in df.columns and 'breakeven_10y' in df.columns:
                nominal = df['us_10y'].iloc[-1]
                breakeven = df['breakeven_10y'].iloc[-1]
                real_rate = nominal - breakeven
            else:
                return {'real_rate': 0.0, 'nominal_rate': 0.0, 'breakeven_inflation': 0.0, 'percentile': 0.5, 'signal': 'NO_DATA'}
        else:
            real_rate = df['tips_10y'].iloc[-1]
            nominal = df.get('us_10y', pd.Series([0])).iloc[-1]
            breakeven = nominal - real_rate if nominal else 0
        if 'tips_10y' in df.columns:
            real_series = df['tips_10y']
            percentile = (real_series < real_rate).mean()
        else:
            percentile = 0.5
        if real_rate > 2.0:
            signal = 'RESTRICTIVE'
        elif real_rate > 0.5:
            signal = 'NEUTRAL'
        elif real_rate > -0.5:
            signal = 'ACCOMMODATIVE'
        else:
            signal = 'VERY_ACCOMMODATIVE'
        return {'real_rate': float(real_rate), 'nominal_rate': float(nominal), 'breakeven_inflation': float(breakeven), 'percentile': float(percentile), 'signal': signal}

    def get_growth_signal(self, as_of_date: str=None) -> Dict[str, Any]:
        term_spread = self.calculate_term_spread(as_of_date, '2y', '10y')
        term_spread_3m10y = self.calculate_term_spread(as_of_date, '3m', '10y') if self.data_source == 'fred' else term_spread
        real_rate = self.calculate_real_rate(as_of_date)
        score = 0
        reasons = []
        if term_spread['signal'] in ['RECESSION_WARNING']:
            score -= 2
            reasons.append('收益率曲线持续倒挂，衰退风险高')
        elif term_spread['signal'] in ['CAUTION', 'VERY_FLAT']:
            score -= 1
            reasons.append('收益率曲线平坦或倒挂')
        elif term_spread['signal'] in ['STEEPENING']:
            score += 1
            reasons.append('收益率曲线变陡，增长预期上升')
        elif term_spread['signal'] in ['NORMAL']:
            score += 0.5
            reasons.append('收益率曲线正常')
        if real_rate['signal'] == 'RESTRICTIVE':
            score -= 1
            reasons.append('实际利率过高，货币政策紧缩')
        elif real_rate['signal'] in ['ACCOMMODATIVE', 'VERY_ACCOMMODATIVE']:
            score += 1
            reasons.append('实际利率低，货币环境宽松')
        if score >= 1.5:
            signal = 'BULLISH'
            confidence = min(0.9, 0.5 + score * 0.1)
        elif score <= -1.5:
            signal = 'BEARISH'
            confidence = min(0.9, 0.5 + abs(score) * 0.1)
        else:
            signal = 'NEUTRAL'
            confidence = 0.5
        return {'signal': signal, 'confidence': confidence, 'components': {'term_spread': term_spread, 'real_rate': real_rate}, 'reasoning': '; '.join(reasons) if reasons else '信号混合，保持中性'}

    def _empty_spread_result(self) -> Dict[str, Any]:
        return {'spread': 0.0, 'spread_percentile': 0.5, 'spread_change_1m': 0.0, 'spread_change_3m': 0.0, 'is_inverted': False, 'inversion_days': 0, 'signal': 'NO_DATA', 'as_of_date': datetime.now().strftime('%Y-%m-%d')}
