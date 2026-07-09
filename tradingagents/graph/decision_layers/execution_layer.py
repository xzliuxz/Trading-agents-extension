import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
logger = logging.getLogger(__name__)

@dataclass
class Order:
    ticker: str
    side: str
    quantity: float
    order_type: str
    limit_price: Optional[float] = None
    urgency: str = 'NORMAL'
    reason: str = ''

class ExecutionLayer:

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}
        self.min_trade_value = self.config.get('min_trade_value', 100)
        self.max_single_order_pct = self.config.get('max_single_order_pct', 0.05)
        self.commission_rate = self.config.get('commission_rate', 0.001)
        self.slippage_estimate = self.config.get('slippage_estimate', 0.001)
        self.rebalance_threshold = self.config.get('rebalance_threshold', 0.03)
        self._pending_orders: List[Order] = []
        self._execution_history: List[Dict] = []

    def make_decision(self, tactical_decision: Dict, current_positions: Dict[str, float], current_prices: Dict[str, float], total_value: float, volume_data: Dict[str, float]=None, as_of_date: str=None) -> Dict[str, Any]:
        current_date = datetime.strptime(as_of_date, '%Y-%m-%d') if as_of_date else datetime.now()
        logger.info(f"执行层决策: {as_of_date or 'today'}")
        target_weights = tactical_decision.get('target_weights', {})
        current_weights = self._calculate_current_weights(current_positions, current_prices, total_value)
        weight_diffs = self._calculate_weight_diffs(current_weights, target_weights)
        significant_diffs = {k: v for k, v in weight_diffs.items() if abs(v) > self.rebalance_threshold}
        if not significant_diffs:
            return {'timestamp': current_date.isoformat(), 'orders': [], 'estimated_cost': 0.0, 'execution_plan': {}, 'reasoning': '当前持仓接近目标，无需调整', 'layer': 'execution'}
        orders = self._generate_orders(significant_diffs, current_prices, total_value, volume_data)
        execution_plan = self._optimize_execution(orders, volume_data)
        estimated_cost = self._estimate_transaction_cost(orders, current_prices)
        reasoning = self._generate_reasoning(weight_diffs, orders, estimated_cost)
        decision = {'timestamp': current_date.isoformat(), 'as_of_date': as_of_date or current_date.strftime('%Y-%m-%d'), 'tactical_reference': tactical_decision.get('timestamp'), 'orders': orders, 'order_count': len(orders), 'total_buy_value': sum((o.quantity * current_prices.get(o.ticker, 0) for o in orders if o.side == 'BUY')), 'total_sell_value': sum((o.quantity * current_prices.get(o.ticker, 0) for o in orders if o.side == 'SELL')), 'estimated_cost': estimated_cost, 'execution_plan': execution_plan, 'weight_adjustments': significant_diffs, 'reasoning': reasoning, 'layer': 'execution'}
        self._pending_orders = orders
        self._execution_history.append(decision)
        if len(self._execution_history) > 200:
            self._execution_history = self._execution_history[-200:]
        return decision

    def _calculate_current_weights(self, positions: Dict[str, float], prices: Dict[str, float], total_value: float) -> Dict[str, float]:
        if total_value <= 0:
            return {}
        weights = {}
        for ticker, quantity in positions.items():
            price = prices.get(ticker, 0)
            value = quantity * price
            weights[ticker] = value / total_value
        return weights

    def _calculate_weight_diffs(self, current: Dict[str, float], target: Dict[str, float]) -> Dict[str, float]:
        all_tickers = set(current.keys()) | set(target.keys())
        diffs = {}
        for ticker in all_tickers:
            curr_weight = current.get(ticker, 0)
            tgt_weight = target.get(ticker, 0)
            diff = tgt_weight - curr_weight
            if abs(diff) > 0.001:
                diffs[ticker] = diff
        return diffs

    def _generate_orders(self, weight_diffs: Dict[str, float], prices: Dict[str, float], total_value: float, volume_data: Dict[str, float]=None) -> List[Order]:
        orders = []
        for ticker, weight_diff in weight_diffs.items():
            price = prices.get(ticker, 0)
            if price <= 0:
                continue
            trade_value = abs(weight_diff) * total_value
            if trade_value < self.min_trade_value:
                continue
            quantity = trade_value / price
            side = 'BUY' if weight_diff > 0 else 'SELL'
            order_type, urgency = self._determine_order_type(ticker, trade_value, volume_data)
            limit_price = None
            if order_type == 'LIMIT':
                if side == 'BUY':
                    limit_price = price * 0.998
                else:
                    limit_price = price * 1.002
            order = Order(ticker=ticker, side=side, quantity=round(quantity, 4), order_type=order_type, limit_price=limit_price, urgency=urgency, reason=f'权重调整 {weight_diff:+.1%}')
            orders.append(order)
        urgency_order = {'HIGH': 0, 'NORMAL': 1, 'LOW': 2}
        orders.sort(key=lambda o: urgency_order.get(o.urgency, 1))
        return orders

    def _determine_order_type(self, ticker: str, trade_value: float, volume_data: Dict[str, float]=None) -> tuple:
        order_type = 'MARKET'
        urgency = 'NORMAL'
        if volume_data and ticker in volume_data:
            daily_volume_value = volume_data[ticker]
            if daily_volume_value > 0:
                participation_rate = trade_value / daily_volume_value
                if participation_rate > 0.05:
                    order_type = 'TWAP'
                    urgency = 'LOW'
                elif participation_rate > 0.01:
                    order_type = 'LIMIT'
                    urgency = 'NORMAL'
        return (order_type, urgency)

    def _optimize_execution(self, orders: List[Order], volume_data: Dict[str, float]=None) -> Dict[str, Any]:
        plan = {'immediate': [], 'scheduled': [], 'total_orders': len(orders)}
        for order in orders:
            if order.urgency == 'HIGH' or order.order_type == 'MARKET':
                plan['immediate'].append({'ticker': order.ticker, 'side': order.side, 'quantity': order.quantity, 'type': order.order_type})
            else:
                slices = self._calculate_order_slices(order, volume_data)
                plan['scheduled'].append({'ticker': order.ticker, 'side': order.side, 'total_quantity': order.quantity, 'slices': slices, 'type': order.order_type})
        return plan

    def _calculate_order_slices(self, order: Order, volume_data: Dict[str, float]=None) -> List[Dict]:
        n_slices = 3
        if volume_data and order.ticker in volume_data:
            participation = order.quantity / volume_data[order.ticker]
            if participation > 0.1:
                n_slices = 5
            elif participation > 0.05:
                n_slices = 4
        slice_quantity = order.quantity / n_slices
        return [{'slice': i + 1, 'quantity': round(slice_quantity, 4)} for i in range(n_slices)]

    def _estimate_transaction_cost(self, orders: List[Order], prices: Dict[str, float]) -> float:
        total_cost = 0.0
        for order in orders:
            price = prices.get(order.ticker, 0)
            trade_value = order.quantity * price
            commission = trade_value * self.commission_rate
            if order.order_type == 'MARKET':
                slippage = trade_value * self.slippage_estimate
            elif order.order_type == 'TWAP':
                slippage = trade_value * self.slippage_estimate * 0.5
            else:
                slippage = trade_value * self.slippage_estimate * 0.3
            total_cost += commission + slippage
        return total_cost

    def _generate_reasoning(self, weight_diffs: Dict[str, float], orders: List[Order], estimated_cost: float) -> str:
        buy_orders = [o for o in orders if o.side == 'BUY']
        sell_orders = [o for o in orders if o.side == 'SELL']
        reasoning = f'\n执行层决策说明：\n\n1. 订单概要：\n   - 买入订单: {len(buy_orders)} 笔\n   - 卖出订单: {len(sell_orders)} 笔\n   - 预估交易成本: ${estimated_cost:.2f}\n\n2. 主要调整：\n'
        sorted_diffs = sorted(weight_diffs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for ticker, diff in sorted_diffs:
            action = '增持' if diff > 0 else '减持'
            reasoning += f'   - {ticker}: {action} {abs(diff):.1%}\n'
        return reasoning.strip()

    def get_pending_orders(self) -> List[Order]:
        return self._pending_orders

    def clear_pending_orders(self) -> None:
        self._pending_orders = []

    def get_history(self, n: int=50) -> List[Dict]:
        return self._execution_history[-n:]
