import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class FundFlowAnalyzer:
    ASSET_CLASS_ETFS = {'us_equity': ['SPY', 'QQQ', 'IWM'], 'intl_equity': ['EFA', 'EEM', 'VWO'], 'fixed_income': ['TLT', 'IEF', 'LQD', 'HYG'], 'commodities': ['GLD', 'SLV', 'DBC', 'USO'], 'sectors': ['XLF', 'XLE', 'XLK', 'XLV', 'XLU']}

    def __init__(self):
        self._cache = {}
        self._cache_expiry = {}
        self._cache_duration = timedelta(hours=6)

    def get_etf_data(self, tickers: List[str], as_of_date: str=None, lookback_days: int=126) -> pd.DataFrame:
        if as_of_date:
            end_date = datetime.strptime(as_of_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 30)
        cache_key = f"etf_{'_'.join(sorted(tickers))}_{start_date.date()}_{end_date.date()}"
        if cache_key in self._cache:
            if datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
                return self._cache[cache_key]
        try:
            import yfinance as yf
            data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False, threads=False, auto_adjust=True)
            if data.empty:
                return pd.DataFrame()
            if not isinstance(data.columns, pd.MultiIndex):
                data = pd.DataFrame({('Close', tickers[0]): data['Close'], ('Volume', tickers[0]): data['Volume']})
            self._cache[cache_key] = data
            self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
            return data
        except Exception as e:
            logger.error(f'ETF data fetch error: {e}')
            return pd.DataFrame()

    def calculate_volume_flow(self, ticker: str, as_of_date: str=None, short_window: int=5, long_window: int=20) -> Dict[str, Any]:
        data = self.get_etf_data([ticker], as_of_date)
        if data.empty:
            return self._empty_volume_result()
        try:
            close = data['Close'][ticker].dropna()
            volume = data['Volume'][ticker].dropna()
        except:
            return self._empty_volume_result()
        if len(volume) < long_window + 5:
            return self._empty_volume_result()
        short_vol = volume.iloc[-short_window:].mean()
        long_vol = volume.iloc[-long_window:].mean()
        volume_ratio = short_vol / long_vol if long_vol > 0 else 1.0
        vol_change = volume.iloc[-5:].mean() / volume.iloc[-20:-5].mean() - 1 if volume.iloc[-20:-5].mean() > 0 else 0
        if vol_change > 0.3:
            volume_trend = 'INCREASING'
        elif vol_change < -0.3:
            volume_trend = 'DECREASING'
        else:
            volume_trend = 'STABLE'
        price_change = close.iloc[-5:].mean() / close.iloc[-20:-5].mean() - 1 if close.iloc[-20:-5].mean() > 0 else 0
        if price_change > 0.02 and vol_change < -0.2:
            divergence = 'BEARISH'
        elif price_change < -0.02 and vol_change < -0.2:
            divergence = 'BULLISH'
        elif price_change > 0.02 and vol_change > 0.2:
            divergence = 'BULLISH_CONFIRMATION'
        elif price_change < -0.02 and vol_change > 0.2:
            divergence = 'BEARISH_CONFIRMATION'
        else:
            divergence = 'NONE'
        if volume_ratio > 1.5 and price_change > 0:
            signal = 'STRONG_INFLOW'
        elif volume_ratio > 1.2 and price_change > 0:
            signal = 'MODERATE_INFLOW'
        elif volume_ratio > 1.5 and price_change < 0:
            signal = 'STRONG_OUTFLOW'
        elif volume_ratio > 1.2 and price_change < 0:
            signal = 'MODERATE_OUTFLOW'
        else:
            signal = 'NEUTRAL'
        return {'volume_ratio': float(volume_ratio), 'volume_trend': volume_trend, 'price_volume_divergence': divergence, 'signal': signal}

    def calculate_relative_strength(self, as_of_date: str=None, lookback: int=21) -> Dict[str, Any]:
        all_tickers = []
        for etfs in self.ASSET_CLASS_ETFS.values():
            all_tickers.extend(etfs[:2])
        data = self.get_etf_data(all_tickers, as_of_date, lookback + 30)
        if data.empty:
            return {'rankings': {}, 'momentum': {}, 'rotation_signal': 'NO_DATA'}
        class_momentum = {}
        for asset_class, etfs in self.ASSET_CLASS_ETFS.items():
            class_returns = []
            for ticker in etfs[:2]:
                try:
                    prices = data['Close'][ticker].dropna()
                    if len(prices) > lookback:
                        ret = prices.iloc[-1] / prices.iloc[-lookback] - 1
                        class_returns.append(ret)
                except:
                    pass
            if class_returns:
                class_momentum[asset_class] = np.mean(class_returns)
        sorted_classes = sorted(class_momentum.items(), key=lambda x: x[1], reverse=True)
        rankings = {item[0]: i + 1 for i, item in enumerate(sorted_classes)}
        if sorted_classes:
            top_class = sorted_classes[0][0]
            top_momentum = sorted_classes[0][1]
            if top_class in ['commodities'] and top_momentum > 0.05:
                rotation_signal = 'INFLATION_HEDGE'
            elif top_class in ['fixed_income'] and top_momentum > 0.02:
                rotation_signal = 'RISK_OFF'
            elif top_class in ['us_equity', 'intl_equity'] and top_momentum > 0.05:
                rotation_signal = 'RISK_ON'
            else:
                rotation_signal = 'NEUTRAL'
        else:
            rotation_signal = 'NO_DATA'
        return {'rankings': rankings, 'momentum': class_momentum, 'rotation_signal': rotation_signal}

    def calculate_sector_rotation(self, as_of_date: str=None) -> Dict[str, Any]:
        sector_etfs = self.ASSET_CLASS_ETFS['sectors']
        data = self.get_etf_data(sector_etfs, as_of_date, 42)
        if data.empty:
            return {'sector_rankings': {}, 'cycle_phase': 'UNKNOWN', 'leading_sectors': []}
        sector_momentum = {}
        for ticker in sector_etfs:
            try:
                prices = data['Close'][ticker].dropna()
                if len(prices) > 21:
                    mom = prices.iloc[-1] / prices.iloc[-21] - 1
                    sector_momentum[ticker] = mom
            except:
                pass
        sorted_sectors = sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)
        rankings = {s[0]: i + 1 for i, s in enumerate(sorted_sectors)}
        leading = [s[0] for s in sorted_sectors[:2]]
        if 'XLF' in leading or 'XLE' in leading:
            cycle_phase = 'EARLY_CYCLE'
        elif 'XLK' in leading:
            cycle_phase = 'MID_CYCLE'
        elif 'XLU' in leading or 'XLV' in leading:
            cycle_phase = 'LATE_CYCLE'
        else:
            cycle_phase = 'MIXED'
        return {'sector_rankings': rankings, 'sector_momentum': sector_momentum, 'cycle_phase': cycle_phase, 'leading_sectors': leading}

    def get_flow_signal(self, as_of_date: str=None) -> Dict[str, Any]:
        spy_flow = self.calculate_volume_flow('SPY', as_of_date)
        tlt_flow = self.calculate_volume_flow('TLT', as_of_date)
        gld_flow = self.calculate_volume_flow('GLD', as_of_date)
        relative_strength = self.calculate_relative_strength(as_of_date)
        sector_rotation = self.calculate_sector_rotation(as_of_date)
        score = 0
        reasons = []
        if spy_flow['signal'] in ['STRONG_INFLOW', 'MODERATE_INFLOW']:
            score += 1
            reasons.append('股市资金流入')
        elif spy_flow['signal'] in ['STRONG_OUTFLOW', 'MODERATE_OUTFLOW']:
            score -= 1
            reasons.append('股市资金流出')
        if tlt_flow['signal'] in ['STRONG_INFLOW', 'MODERATE_INFLOW']:
            score -= 0.5
            reasons.append('避险资金流入债券')
        if gld_flow['signal'] in ['STRONG_INFLOW', 'MODERATE_INFLOW']:
            score -= 0.5
            reasons.append('避险资金流入黄金')
        rotation = relative_strength['rotation_signal']
        if rotation == 'RISK_ON':
            score += 0.5
            reasons.append('风险资产领涨')
        elif rotation == 'RISK_OFF':
            score -= 0.5
            reasons.append('避险资产领涨')
        elif rotation == 'INFLATION_HEDGE':
            reasons.append('通胀对冲资产领涨')
        cycle = sector_rotation['cycle_phase']
        if cycle == 'EARLY_CYCLE':
            score += 0.5
            reasons.append('板块轮动显示早周期特征')
        elif cycle == 'LATE_CYCLE':
            score -= 0.5
            reasons.append('板块轮动显示晚周期特征')
        if score >= 1:
            signal = 'RISK_ON'
        elif score <= -1:
            signal = 'RISK_OFF'
        elif abs(score) < 0.5 and rotation != 'NEUTRAL':
            signal = 'ROTATION'
        else:
            signal = 'NEUTRAL'
        confidence = min(0.9, 0.5 + abs(score) * 0.15)
        return {'signal': signal, 'confidence': confidence, 'components': {'spy_flow': spy_flow, 'tlt_flow': tlt_flow, 'gld_flow': gld_flow, 'relative_strength': relative_strength, 'sector_rotation': sector_rotation}, 'reasoning': '; '.join(reasons) if reasons else '资金流向信号混合'}

    def _empty_volume_result(self) -> Dict[str, Any]:
        return {'volume_ratio': 1.0, 'volume_trend': 'UNKNOWN', 'price_volume_divergence': 'NONE', 'signal': 'NO_DATA'}
