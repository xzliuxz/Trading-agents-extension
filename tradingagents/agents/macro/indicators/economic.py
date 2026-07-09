import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import os
from .fred_ssl import bootstrap_fred_ssl, is_cert_verify_error
logger = logging.getLogger(__name__)

class EconomicIndicators:
    FRED_SERIES = {'lei': 'USSLIND', 'consumer_confidence': 'UMCSENT', 'building_permits': 'PERMIT', 'industrial_production': 'INDPRO', 'initial_claims': 'ICSA', 'continued_claims': 'CCSA', 'nonfarm_payrolls': 'PAYEMS', 'cpi_yoy': 'CPIAUCSL', 'pce_yoy': 'PCEPI', 'breakeven_5y': 'T5YIE', 'ism_manufacturing': 'MANEMP', 'retail_sales': 'RSAFS'}
    PMI_PROXIES = {'manufacturing': ['INDPRO', 'MANEMP', 'NEWORDER'], 'services': ['PAYEMS', 'RSAFS']}

    def __init__(self, data_source: str='fred'):
        self.data_source = data_source
        self._cache = {}
        self._cache_expiry = {}
        self._cache_duration = timedelta(hours=24)
        self._fred_ssl_warned = False
        bootstrap_fred_ssl()

    def _warn_fred_ssl_once(self):
        if self._fred_ssl_warned:
            return
        logger.warning('FRED SSL certificate verification failed for economic indicators; continuing without live FRED economic series.')
        self._fred_ssl_warned = True

    def get_economic_data(self, as_of_date: str=None, lookback_months: int=24) -> pd.DataFrame:
        if as_of_date:
            end_date = datetime.strptime(as_of_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_months * 31)
        cache_key = f'econ_{start_date.date()}_{end_date.date()}'
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
                logger.warning('FRED_API_KEY not set; economic indicator data unavailable.')
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
            return df.infer_objects(copy=False).ffill()
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
        dates = pd.date_range(start_date, end_date, freq='M')
        n = len(dates)
        np.random.seed(42)
        t = np.arange(n)
        cycle = np.sin(2 * np.pi * t / 48)
        noise = np.random.randn(n) * 0.1
        df = pd.DataFrame({'lei': 100 + 5 * cycle + noise * 2, 'consumer_confidence': 80 + 20 * cycle + noise * 5, 'initial_claims': 250000 - 50000 * cycle + noise * 20000, 'industrial_production': 100 + 10 * cycle + noise * 2, 'cpi_yoy': 2.5 + 1.5 * np.sin(2 * np.pi * t / 24) + noise, 'breakeven_5y': 2.0 + 0.5 * np.sin(2 * np.pi * t / 24) + noise * 0.2}, index=dates)
        return df

    def calculate_lei_signal(self, as_of_date: str=None) -> Dict[str, Any]:
        df = self.get_economic_data(as_of_date)
        if df.empty or 'lei' not in df.columns:
            return self._empty_indicator_result('lei')
        lei = df['lei'].dropna()
        if len(lei) < 7:
            return self._empty_indicator_result('lei')
        current = lei.iloc[-1]
        mom_3m = (current / lei.iloc[-4] - 1) * 100 if len(lei) > 4 else 0
        mom_6m = (current / lei.iloc[-7] - 1) * 100 if len(lei) > 7 else 0
        percentile = (lei < current).mean()
        if mom_3m > 1 and mom_6m > 2:
            signal = 'EXPANSION'
        elif mom_3m > 0 and mom_6m > 0:
            signal = 'GROWTH'
        elif mom_3m < -1 and mom_6m < -2:
            signal = 'CONTRACTION'
        elif mom_3m < 0 or mom_6m < 0:
            signal = 'SLOWING'
        else:
            signal = 'STABLE'
        return {'value': float(current), 'mom_3m': float(mom_3m), 'mom_6m': float(mom_6m), 'percentile': float(percentile), 'signal': signal}

    def calculate_consumer_confidence(self, as_of_date: str=None) -> Dict[str, Any]:
        df = self.get_economic_data(as_of_date)
        if df.empty or 'consumer_confidence' not in df.columns:
            return self._empty_indicator_result('consumer_confidence')
        cc = df['consumer_confidence'].dropna()
        if len(cc) < 4:
            return self._empty_indicator_result('consumer_confidence')
        current = cc.iloc[-1]
        prev = cc.iloc[-2]
        mom_3m = current - cc.iloc[-4] if len(cc) > 4 else 0
        percentile = (cc < current).mean()
        if current > 100 and mom_3m > 5:
            signal = 'VERY_BULLISH'
        elif current > 80 and mom_3m > 0:
            signal = 'BULLISH'
        elif current < 60 or mom_3m < -10:
            signal = 'BEARISH'
        elif current < 70:
            signal = 'CAUTIOUS'
        else:
            signal = 'NEUTRAL'
        return {'value': float(current), 'change_1m': float(current - prev), 'change_3m': float(mom_3m), 'percentile': float(percentile), 'signal': signal}

    def calculate_labor_market(self, as_of_date: str=None) -> Dict[str, Any]:
        df = self.get_economic_data(as_of_date)
        result = {'initial_claims': None, 'claims_trend': 'UNKNOWN', 'signal': 'NO_DATA'}
        if df.empty or 'initial_claims' not in df.columns:
            return result
        claims = df['initial_claims'].dropna()
        if len(claims) < 5:
            return result
        current = claims.iloc[-1]
        ma_4w = claims.iloc[-4:].mean()
        ma_13w = claims.iloc[-13:].mean() if len(claims) > 13 else ma_4w
        if ma_4w < ma_13w * 0.95:
            claims_trend = 'IMPROVING'
        elif ma_4w > ma_13w * 1.05:
            claims_trend = 'DETERIORATING'
        else:
            claims_trend = 'STABLE'
        if current < 220000:
            signal = 'STRONG'
        elif current < 280000:
            signal = 'HEALTHY'
        elif current < 350000:
            signal = 'WEAKENING'
        else:
            signal = 'RECESSIONARY'
        return {'initial_claims': float(current), 'ma_4w': float(ma_4w), 'ma_13w': float(ma_13w), 'claims_trend': claims_trend, 'signal': signal}

    def calculate_inflation_expectations(self, as_of_date: str=None) -> Dict[str, Any]:
        df = self.get_economic_data(as_of_date)
        result = {'breakeven_5y': 2.0, 'cpi_trend': 'UNKNOWN', 'signal': 'NEUTRAL'}
        if df.empty:
            return result
        if 'breakeven_5y' in df.columns:
            be = df['breakeven_5y'].dropna()
            if len(be) > 0:
                result['breakeven_5y'] = float(be.iloc[-1])
                if len(be) > 3:
                    change_3m = be.iloc[-1] - be.iloc[-4]
                    if change_3m > 0.3:
                        result['inflation_trend'] = 'RISING'
                    elif change_3m < -0.3:
                        result['inflation_trend'] = 'FALLING'
                    else:
                        result['inflation_trend'] = 'STABLE'
        be_val = result['breakeven_5y']
        if be_val > 3.0:
            result['signal'] = 'HIGH_INFLATION'
        elif be_val > 2.5:
            result['signal'] = 'ABOVE_TARGET'
        elif be_val < 1.5:
            result['signal'] = 'DEFLATION_RISK'
        else:
            result['signal'] = 'ON_TARGET'
        return result

    def get_growth_signal(self, as_of_date: str=None) -> Dict[str, Any]:
        lei = self.calculate_lei_signal(as_of_date)
        consumer = self.calculate_consumer_confidence(as_of_date)
        labor = self.calculate_labor_market(as_of_date)
        score = 0
        reasons = []
        if lei['signal'] in ['EXPANSION', 'GROWTH']:
            score += 1.5
            reasons.append(f"领先指标显示增长 (MoM 3M: {lei['mom_3m']:.1f}%)")
        elif lei['signal'] in ['CONTRACTION']:
            score -= 2
            reasons.append('领先指标显示收缩')
        elif lei['signal'] == 'SLOWING':
            score -= 0.5
            reasons.append('领先指标放缓')
        if consumer['signal'] in ['VERY_BULLISH', 'BULLISH']:
            score += 1
            reasons.append('消费者信心强劲')
        elif consumer['signal'] == 'BEARISH':
            score -= 1
            reasons.append('消费者信心疲软')
        if labor['signal'] in ['STRONG', 'HEALTHY']:
            score += 1
            reasons.append('劳动力市场健康')
        elif labor['signal'] == 'RECESSIONARY':
            score -= 1.5
            reasons.append('劳动力市场恶化')
        elif labor['signal'] == 'WEAKENING':
            score -= 0.5
            reasons.append('劳动力市场走弱')
        if score >= 2:
            signal = 'BULLISH'
            confidence = min(0.9, 0.6 + score * 0.1)
        elif score <= -1.5:
            signal = 'BEARISH'
            confidence = min(0.9, 0.6 + abs(score) * 0.1)
        else:
            signal = 'NEUTRAL'
            confidence = 0.5
        return {'signal': signal, 'confidence': confidence, 'score': float(score), 'components': {'lei': lei, 'consumer_confidence': consumer, 'labor_market': labor}, 'reasoning': '; '.join(reasons) if reasons else '经济信号混合'}

    def get_inflation_signal(self, as_of_date: str=None) -> Dict[str, Any]:
        inflation = self.calculate_inflation_expectations(as_of_date)
        signal = inflation['signal']
        if signal == 'HIGH_INFLATION':
            macro_signal = 'RISING'
            confidence = 0.8
            recommendation = '增加通胀对冲资产（商品、黄金）'
        elif signal == 'ABOVE_TARGET':
            macro_signal = 'ELEVATED'
            confidence = 0.6
            recommendation = '适度配置通胀保护'
        elif signal == 'DEFLATION_RISK':
            macro_signal = 'FALLING'
            confidence = 0.7
            recommendation = '增加债券配置'
        else:
            macro_signal = 'STABLE'
            confidence = 0.5
            recommendation = '维持正常配置'
        return {'signal': macro_signal, 'confidence': confidence, 'components': inflation, 'recommendation': recommendation}

    def _empty_indicator_result(self, indicator: str) -> Dict[str, Any]:
        return {'value': 0.0, 'mom_3m': 0.0, 'mom_6m': 0.0, 'percentile': 0.5, 'signal': 'NO_DATA'}
