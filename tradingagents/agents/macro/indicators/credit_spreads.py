import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import os
from .fred_ssl import bootstrap_fred_ssl, is_cert_verify_error
logger = logging.getLogger(__name__)

class CreditSpreadAnalyzer:
    FRED_SERIES = {'baa_yield': 'DBAA', 'aaa_yield': 'DAAA', 'ig_oas': 'BAMLC0A0CM', 'hy_oas': 'BAMLH0A0HYM2', 'ted_spread': 'TEDRATE', 'financial_stress': 'STLFSI4'}

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
        logger.warning('FRED SSL certificate verification failed for credit-spread data; continuing without live FRED credit series.')
        self._fred_ssl_warned = True

    def get_credit_data(self, as_of_date: str=None, lookback_days: int=504) -> pd.DataFrame:
        if as_of_date:
            end_date = datetime.strptime(as_of_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)
        cache_key = f'credit_{start_date.date()}_{end_date.date()}'
        if cache_key in self._cache:
            if datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
                return self._cache[cache_key]
        df = self._fetch_from_fred(start_date, end_date)
        if not df.empty:
            self._cache[cache_key] = df
            self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
        return df

    def _fetch_from_fred(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        allow_synthetic = os.getenv('TRADINGAGENTS_ALLOW_SYNTHETIC_DATA', '0') == '1'
        try:
            from fredapi import Fred
            api_key = os.getenv('FRED_API_KEY')
            if not api_key:
                if allow_synthetic:
                    logger.warning('FRED_API_KEY not set; using synthetic macro data (TRADINGAGENTS_ALLOW_SYNTHETIC_DATA=1).')
                    return self._generate_synthetic_data(start_date, end_date)
                logger.warning('FRED_API_KEY not set; credit spread data unavailable.')
                return pd.DataFrame()
            fred = Fred(api_key=api_key)
            data = {}
            for name, series_id in self.FRED_SERIES.items():
                try:
                    series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                    data[name] = series
                except Exception as e:
                    if is_cert_verify_error(e):
                        self._warn_fred_ssl_once()
                        if allow_synthetic:
                            return self._generate_synthetic_data(start_date, end_date)
                        return pd.DataFrame()
                    logger.debug(f'Failed to fetch {series_id}: {e}')
            if not data:
                if allow_synthetic:
                    return self._generate_synthetic_data(start_date, end_date)
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df.index.name = 'date'
            if 'baa_yield' in df.columns and 'aaa_yield' in df.columns:
                df['baa_aaa_spread'] = df['baa_yield'] - df['aaa_yield']
            return df.infer_objects(copy=False).ffill().dropna(how='all')
        except ImportError:
            logger.warning('fredapi not installed')
            if allow_synthetic:
                return self._generate_synthetic_data(start_date, end_date)
            return pd.DataFrame()
        except Exception as e:
            if is_cert_verify_error(e):
                self._warn_fred_ssl_once()
                if allow_synthetic:
                    return self._generate_synthetic_data(start_date, end_date)
                return pd.DataFrame()
            logger.error(f'FRED fetch error: {e}')
            if allow_synthetic:
                return self._generate_synthetic_data(start_date, end_date)
            return pd.DataFrame()

    def _generate_synthetic_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        dates = pd.date_range(start_date, end_date, freq='B')
        n = len(dates)
        np.random.seed(42)
        t = np.arange(n)
        cycle = np.sin(2 * np.pi * t / 252)
        noise = np.random.randn(n) * 0.1
        df = pd.DataFrame({'baa_aaa_spread': 1.0 + 0.5 * cycle + noise + 0.3, 'ig_oas': 1.2 + 0.6 * cycle + noise * 1.2 + 0.3, 'hy_oas': 4.0 + 2.0 * cycle + noise * 3 + 1, 'financial_stress': 0 + 0.5 * cycle + noise}, index=dates)
        return df

    def calculate_ig_spread(self, as_of_date: str=None) -> Dict[str, Any]:
        df = self.get_credit_data(as_of_date)
        if df.empty:
            return self._empty_spread_result('ig')
        if 'ig_oas' in df.columns:
            spread_series = df['ig_oas'].dropna()
        elif 'baa_aaa_spread' in df.columns:
            spread_series = df['baa_aaa_spread'].dropna()
        else:
            return self._empty_spread_result('ig')
        if len(spread_series) < 20:
            return self._empty_spread_result('ig')
        current = spread_series.iloc[-1]
        percentile = (spread_series < current).mean()
        mean = spread_series.mean()
        std = spread_series.std()
        z_score = (current - mean) / std if std > 0 else 0
        change_1m = current - spread_series.iloc[-21] if len(spread_series) > 21 else 0
        change_3m = current - spread_series.iloc[-63] if len(spread_series) > 63 else 0
        signal = self._interpret_credit_signal(percentile, z_score, change_3m)
        return {'spread': float(current), 'percentile': float(percentile), 'z_score': float(z_score), 'change_1m': float(change_1m), 'change_3m': float(change_3m), 'signal': signal}

    def calculate_hy_spread(self, as_of_date: str=None) -> Dict[str, Any]:
        df = self.get_credit_data(as_of_date)
        if df.empty or 'hy_oas' not in df.columns:
            return self._empty_spread_result('hy')
        spread_series = df['hy_oas'].dropna()
        if len(spread_series) < 20:
            return self._empty_spread_result('hy')
        current = spread_series.iloc[-1]
        percentile = (spread_series < current).mean()
        mean = spread_series.mean()
        std = spread_series.std()
        z_score = (current - mean) / std if std > 0 else 0
        change_1m = current - spread_series.iloc[-21] if len(spread_series) > 21 else 0
        change_3m = current - spread_series.iloc[-63] if len(spread_series) > 63 else 0
        signal = self._interpret_credit_signal(percentile, z_score, change_3m, is_hy=True)
        return {'spread': float(current), 'percentile': float(percentile), 'z_score': float(z_score), 'change_1m': float(change_1m), 'change_3m': float(change_3m), 'signal': signal}

    def calculate_financial_stress(self, as_of_date: str=None) -> Dict[str, Any]:
        df = self.get_credit_data(as_of_date)
        if df.empty or 'financial_stress' not in df.columns:
            return {'stress_index': 0.0, 'percentile': 0.5, 'signal': 'NO_DATA'}
        stress_series = df['financial_stress'].dropna()
        if len(stress_series) < 20:
            return {'stress_index': 0.0, 'percentile': 0.5, 'signal': 'NO_DATA'}
        current = stress_series.iloc[-1]
        percentile = (stress_series < current).mean()
        if current > 1.5:
            signal = 'CRISIS'
        elif current > 0.5:
            signal = 'ELEVATED'
        elif current > 0:
            signal = 'ABOVE_NORMAL'
        elif current > -0.5:
            signal = 'NORMAL'
        else:
            signal = 'CALM'
        return {'stress_index': float(current), 'percentile': float(percentile), 'signal': signal}

    def _interpret_credit_signal(self, percentile: float, z_score: float, change_3m: float, is_hy: bool=False) -> str:
        z_threshold = 1.5 if is_hy else 1.0
        if z_score > 2:
            return 'CRISIS'
        elif z_score > z_threshold:
            return 'STRESSED'
        elif percentile > 0.8:
            return 'WIDE'
        elif percentile > 0.6 and change_3m > 0.3:
            return 'WIDENING'
        elif percentile < 0.2:
            return 'TIGHT'
        elif percentile < 0.4 and change_3m < -0.3:
            return 'TIGHTENING'
        else:
            return 'NORMAL'

    def get_risk_appetite_signal(self, as_of_date: str=None) -> Dict[str, Any]:
        ig_spread = self.calculate_ig_spread(as_of_date)
        hy_spread = self.calculate_hy_spread(as_of_date)
        stress = self.calculate_financial_stress(as_of_date)
        score = 0
        reasons = []
        if ig_spread['signal'] in ['CRISIS', 'STRESSED']:
            score -= 2
            reasons.append('投资级信用利差显著扩大')
        elif ig_spread['signal'] in ['WIDE', 'WIDENING']:
            score -= 1
            reasons.append('投资级利差偏宽或扩大中')
        elif ig_spread['signal'] in ['TIGHT', 'TIGHTENING']:
            score += 1
            reasons.append('投资级利差收窄，风险偏好上升')
        if hy_spread['signal'] in ['CRISIS', 'STRESSED']:
            score -= 2
            reasons.append('高收益债利差显著扩大')
        elif hy_spread['signal'] in ['WIDE', 'WIDENING']:
            score -= 1
            reasons.append('高收益利差偏宽')
        elif hy_spread['signal'] in ['TIGHT', 'TIGHTENING']:
            score += 1
            reasons.append('高收益利差收窄')
        if stress['signal'] in ['CRISIS', 'ELEVATED']:
            score -= 1.5
            reasons.append('金融压力指数升高')
        elif stress['signal'] == 'CALM':
            score += 0.5
            reasons.append('金融压力处于低位')
        if score >= 1.5:
            signal = 'RISK_ON'
            confidence = min(0.9, 0.5 + score * 0.1)
        elif score <= -2:
            signal = 'RISK_OFF'
            confidence = min(0.9, 0.5 + abs(score) * 0.1)
        else:
            signal = 'NEUTRAL'
            confidence = 0.5
        return {'signal': signal, 'confidence': confidence, 'components': {'ig_spread': ig_spread, 'hy_spread': hy_spread, 'financial_stress': stress}, 'reasoning': '; '.join(reasons) if reasons else '信用市场信号混合'}

    def _empty_spread_result(self, spread_type: str) -> Dict[str, Any]:
        return {'spread': 0.0, 'percentile': 0.5, 'z_score': 0.0, 'change_1m': 0.0, 'change_3m': 0.0, 'signal': 'NO_DATA'}
