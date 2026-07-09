import threading
from datetime import datetime, date
from typing import Optional
from contextlib import contextmanager
_thread_local = threading.local()

def get_current_date() -> str:
    virtual_date = getattr(_thread_local, 'virtual_date', None)
    if virtual_date is not None:
        return virtual_date
    return datetime.now().strftime('%Y-%m-%d')

def set_virtual_date(date_str: Optional[str]):
    if date_str is None:
        if hasattr(_thread_local, 'virtual_date'):
            delattr(_thread_local, 'virtual_date')
    else:
        _thread_local.virtual_date = date_str

def is_backtesting() -> bool:
    return hasattr(_thread_local, 'virtual_date') and _thread_local.virtual_date is not None

class TimeMachine:

    def __init__(self, target_date: str):
        self.target_date = target_date
        self._previous_date = None

    def __enter__(self):
        self._previous_date = getattr(_thread_local, 'virtual_date', None)
        set_virtual_date(self.target_date)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_virtual_date(self._previous_date)
        return False

@contextmanager
def time_travel(target_date: str):
    previous = getattr(_thread_local, 'virtual_date', None)
    try:
        set_virtual_date(target_date)
        yield
    finally:
        set_virtual_date(previous)

class BacktestRunner:

    def __init__(self, start_date: str, end_date: str, frequency: str='daily'):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.frequency = frequency
        self.results = []

    def dates(self):
        import pandas as pd
        if self.frequency == 'daily':
            date_range = pd.bdate_range(start=self.start_date, end=self.end_date)
        elif self.frequency == 'weekly':
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='W-MON')
        elif self.frequency == 'monthly':
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')
        else:
            raise ValueError(f'Unknown frequency: {self.frequency}')
        for dt in date_range:
            yield dt.strftime('%Y-%m-%d')

    def at(self, target_date: str) -> TimeMachine:
        return TimeMachine(target_date)

    def record(self, date: str, result: dict):
        self.results.append({'date': date, **result})

    def get_results(self):
        return self.results

    def get_summary(self):
        if not self.results:
            return {'error': 'No results recorded'}
        import pandas as pd
        df = pd.DataFrame(self.results)
        return {'total_days': len(self.results), 'start_date': self.results[0]['date'], 'end_date': self.results[-1]['date'], 'results': df}
