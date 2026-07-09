from __future__ import annotations
import contextlib
import io
import os
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
import pandas as pd

class MacroDataCache:
    _instance: Optional['MacroDataCache'] = None
    _lock = threading.Lock()
    MACRO_TICKERS = ['^TNX', 'HYG', 'LQD', '^VIX', '^VIX3M', 'UUP', 'SPY']
    FALLBACK_MAP = {'^VIX3M': '^VXV'}

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._cache: Dict[str, pd.DataFrame] = {}
                cls._instance._cache_dates: Dict[str, Tuple[str, str]] = {}
                cls._instance._initialized = False
            return cls._instance

    def get_macro_data(self, ticker: str, cache_dir: str='') -> pd.DataFrame:
        key = ticker.upper()
        with self._lock:
            if key in self._cache and (not self._cache[key].empty):
                return self._cache[key].copy()
        df = self._download(key, cache_dir)
        with self._lock:
            self._cache[key] = df
        return df.copy()

    def _download(self, ticker: str, cache_dir: str) -> pd.DataFrame:
        import yfinance as yf
        today = pd.Timestamp.today()
        start_date = (today - pd.DateOffset(years=15)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{ticker.replace('^', '_')}-macro-cache-{start_date}-{end_date}.csv")
            if os.path.exists(cache_file):
                try:
                    data = pd.read_csv(cache_file)
                    if 'Date' in data.columns:
                        data['Date'] = pd.to_datetime(data['Date'])
                    if not data.empty:
                        return data
                except Exception:
                    pass
        candidates = [ticker]
        if ticker in self.FALLBACK_MAP:
            candidates.append(self.FALLBACK_MAP[ticker])
        for candidate in candidates:
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    raw = yf.download(candidate, start=start_date, end=end_date, multi_level_index=False, progress=False, auto_adjust=True, threads=False)
                if raw is not None and (not raw.empty):
                    raw = raw.reset_index()
                    if cache_dir:
                        try:
                            raw.to_csv(cache_file, index=False)
                        except Exception:
                            pass
                    return raw
            except Exception:
                continue
        return pd.DataFrame()

    def preload_all(self, cache_dir: str=''):
        import concurrent.futures

        def _load(t):
            return (t, self.get_macro_data(t, cache_dir))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_load, t) for t in self.MACRO_TICKERS]
            for f in concurrent.futures.as_completed(futures):
                try:
                    ticker, data = f.result()
                    status = 'OK' if not data.empty else 'EMPTY'
                    print(f'[MacroCache] {ticker}: {status} ({len(data)} rows)')
                except Exception as e:
                    print(f'[MacroCache] Error: {e}')

    def is_macro_ticker(self, ticker: str) -> bool:
        return ticker.upper() in [t.upper() for t in self.MACRO_TICKERS]

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._cache_dates.clear()
