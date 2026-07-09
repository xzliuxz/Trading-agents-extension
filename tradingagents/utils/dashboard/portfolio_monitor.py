import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

class PortfolioMonitor:

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', 100000)
        self._positions: Dict[str, float] = {}
        self._prices: Dict[str, float] = {}
        self._cost_basis: Dict[str, float] = {}
        self._snapshots: List[Dict] = []
        self._trade_log: List[Dict] = []
        self._last_update: Optional[datetime] = None

    def update_positions(self, positions: Dict[str, float], prices: Dict[str, float], timestamp: datetime=None) -> None:
        self._positions = positions.copy()
        self._prices = prices.copy()
        self._last_update = timestamp or datetime.now()
        self._take_snapshot()

    def update_price(self, ticker: str, price: float) -> None:
        self._prices[ticker] = price
        self._last_update = datetime.now()

    def record_trade(self, ticker: str, side: str, quantity: float, price: float, timestamp: datetime=None) -> None:
        self._trade_log.append({'timestamp': (timestamp or datetime.now()).isoformat(), 'ticker': ticker, 'side': side, 'quantity': quantity, 'price': price, 'value': quantity * price})
        if side == 'BUY':
            old_qty = self._positions.get(ticker, 0)
            old_cost = self._cost_basis.get(ticker, 0)
            new_qty = old_qty + quantity
            if new_qty > 0:
                self._cost_basis[ticker] = (old_qty * old_cost + quantity * price) / new_qty
            self._positions[ticker] = new_qty
        else:
            self._positions[ticker] = self._positions.get(ticker, 0) - quantity
            if self._positions[ticker] <= 0:
                self._positions.pop(ticker, None)
                self._cost_basis.pop(ticker, None)

    def get_portfolio_value(self) -> float:
        return sum((self._positions.get(ticker, 0) * self._prices.get(ticker, 0) for ticker in self._positions))

    def get_weights(self) -> Dict[str, float]:
        total = self.get_portfolio_value()
        if total <= 0:
            return {}
        return {ticker: qty * self._prices.get(ticker, 0) / total for ticker, qty in self._positions.items() if qty > 0}

    def get_pnl(self) -> Dict[str, Any]:
        total_value = self.get_portfolio_value()
        total_cost = sum((self._cost_basis.get(ticker, 0) * qty for ticker, qty in self._positions.items()))
        position_pnl = {}
        for ticker, qty in self._positions.items():
            if qty > 0:
                current_price = self._prices.get(ticker, 0)
                avg_cost = self._cost_basis.get(ticker, current_price)
                unrealized = (current_price - avg_cost) * qty
                pct_return = current_price / avg_cost - 1 if avg_cost > 0 else 0
                position_pnl[ticker] = {'quantity': qty, 'avg_cost': avg_cost, 'current_price': current_price, 'unrealized_pnl': unrealized, 'pct_return': pct_return}
        return {'total_value': total_value, 'total_cost': total_cost, 'total_pnl': total_value - total_cost, 'total_return': total_value / self.initial_capital - 1 if self.initial_capital > 0 else 0, 'positions': position_pnl}

    def get_summary(self) -> Dict[str, Any]:
        weights = self.get_weights()
        pnl = self.get_pnl()
        return {'timestamp': self._last_update.isoformat() if self._last_update else None, 'total_value': pnl['total_value'], 'total_return': pnl['total_return'], 'total_pnl': pnl['total_pnl'], 'num_positions': len(self._positions), 'top_holdings': sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5], 'concentration': max(weights.values()) if weights else 0}

    def get_allocation_breakdown(self) -> Dict[str, Any]:
        weights = self.get_weights()
        asset_classes = {'equities': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VUSA.L'], 'bonds': ['TLT', 'IEF', 'AGG', 'LQD', 'HYG'], 'gold': ['GLD', 'IAU', 'SGLN.L'], 'commodities': ['DBC', 'USO', 'GSG'], 'cash': ['SHV', 'BIL', 'CASH']}
        class_weights = {cls: 0.0 for cls in asset_classes}
        unclassified = 0.0
        for ticker, weight in weights.items():
            classified = False
            for cls, tickers in asset_classes.items():
                if ticker in tickers or ticker.upper() in [t.upper() for t in tickers]:
                    class_weights[cls] += weight
                    classified = True
                    break
            if not classified:
                unclassified += weight
        if unclassified > 0:
            class_weights['other'] = unclassified
        return {'by_asset_class': class_weights, 'by_position': weights, 'herfindahl_index': sum((w ** 2 for w in weights.values())) if weights else 0}

    def _take_snapshot(self) -> None:
        snapshot = {'timestamp': self._last_update.isoformat() if self._last_update else datetime.now().isoformat(), 'total_value': self.get_portfolio_value(), 'weights': self.get_weights().copy(), 'positions': self._positions.copy()}
        self._snapshots.append(snapshot)
        if len(self._snapshots) > 1000:
            self._snapshots = self._snapshots[-1000:]

    def get_history(self, n: int=100) -> List[Dict]:
        return self._snapshots[-n:]

    def get_trade_log(self, n: int=50) -> List[Dict]:
        return self._trade_log[-n:]

    def get_equity_curve(self) -> pd.Series:
        if not self._snapshots:
            return pd.Series()
        data = [(s['timestamp'], s['total_value']) for s in self._snapshots]
        df = pd.DataFrame(data, columns=['timestamp', 'value'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df['value']
