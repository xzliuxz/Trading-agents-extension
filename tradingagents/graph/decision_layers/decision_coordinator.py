import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from .strategic_layer import StrategicLayer
from .tactical_layer import TacticalLayer
from .execution_layer import ExecutionLayer
logger = logging.getLogger(__name__)

class DecisionCoordinator:

    def __init__(self, config: Dict[str, Any]=None, regime_detector=None, factor_analyst=None, risk_budget_manager=None, dynamic_risk_manager=None):
        self.config = config or {}
        self.strategic_layer = StrategicLayer(config=config, regime_detector=regime_detector, risk_budget_manager=risk_budget_manager)
        self.tactical_layer = TacticalLayer(config=config, factor_analyst=factor_analyst, dynamic_risk_manager=dynamic_risk_manager)
        self.execution_layer = ExecutionLayer(config=config)
        self.strategic_frequency = self.config.get('strategic_frequency', 'monthly')
        self.tactical_frequency = self.config.get('tactical_frequency', 'weekly')
        self.execution_frequency = self.config.get('execution_frequency', 'daily')
        self._last_strategic_date: Optional[datetime] = None
        self._last_tactical_date: Optional[datetime] = None
        self._decision_log: List[Dict] = []

    def run_full_cycle(self, market_data: Dict=None, current_positions: Dict[str, float]=None, current_prices: Dict[str, float]=None, total_value: float=100000, volume_data: Dict[str, float]=None, as_of_date: str=None, force_all: bool=False) -> Dict[str, Any]:
        current_date = datetime.strptime(as_of_date, '%Y-%m-%d') if as_of_date else datetime.now()
        logger.info(f"=== 开始决策周期: {as_of_date or 'today'} ===")
        agent_context = self._extract_agent_context(market_data)
        macro_context = agent_context.get('macro_context')
        analyst_context = agent_context.get('analyst_context')
        results = {'timestamp': current_date.isoformat(), 'as_of_date': as_of_date or current_date.strftime('%Y-%m-%d')}
        if macro_context or analyst_context:
            results['agent_context'] = {'macro_source': agent_context.get('macro_source'), 'analyst_source': agent_context.get('analyst_source'), 'macro_context_used': bool(macro_context), 'analyst_context_used': bool(analyst_context)}
        should_run_strategic = self._should_run_layer('strategic', current_date, force_all)
        if not should_run_strategic and macro_context and self.config.get('refresh_strategic_on_agent_macro', True):
            logger.info('>>> 检测到 Agent 宏观上下文，触发战略层刷新')
            should_run_strategic = True
        if should_run_strategic:
            logger.info('>>> 执行战略层决策')
            strategic_decision = self.strategic_layer.make_decision(as_of_date=as_of_date, force=force_all, macro_context=macro_context)
            self._last_strategic_date = current_date
        else:
            strategic_decision = self.strategic_layer.get_last_decision()
            if not strategic_decision:
                strategic_decision = self.strategic_layer.make_decision(as_of_date=as_of_date, macro_context=macro_context)
        results['strategic'] = strategic_decision
        should_run_tactical = self._should_run_layer('tactical', current_date, force_all)
        if not should_run_tactical and analyst_context and self.config.get('refresh_tactical_on_agent_signals', True):
            logger.info('>>> 检测到 Analyst 上下文，触发战术层刷新')
            should_run_tactical = True
        if should_run_tactical:
            logger.info('>>> 执行战术层决策')
            tactical_decision = self.tactical_layer.make_decision(strategic_decision=strategic_decision, market_data=market_data, as_of_date=as_of_date, force=force_all, analyst_context=analyst_context)
            self._last_tactical_date = current_date
        else:
            tactical_decision = self.tactical_layer.get_last_decision()
            if not tactical_decision:
                tactical_decision = self.tactical_layer.make_decision(strategic_decision=strategic_decision, market_data=market_data, as_of_date=as_of_date, analyst_context=analyst_context)
        results['tactical'] = tactical_decision
        logger.info('>>> 执行执行层决策')
        execution_decision = self.execution_layer.make_decision(tactical_decision=tactical_decision, current_positions=current_positions or {}, current_prices=current_prices or {}, total_value=total_value, volume_data=volume_data, as_of_date=as_of_date)
        results['execution'] = execution_decision
        results['summary'] = self._generate_summary(strategic_decision, tactical_decision, execution_decision)
        self._log_decision(results)
        logger.info(f"=== 决策周期完成: {results['summary']['action_summary']} ===")
        return results

    def _should_run_layer(self, layer: str, current_date: datetime, force: bool) -> bool:
        if force:
            return True
        if layer == 'strategic':
            if not self._last_strategic_date:
                return True
            days_since = (current_date - self._last_strategic_date).days
            if self.strategic_frequency == 'monthly':
                return days_since >= 20
            elif self.strategic_frequency == 'weekly':
                return days_since >= 5
            else:
                return True
        elif layer == 'tactical':
            if not self._last_tactical_date:
                return True
            days_since = (current_date - self._last_tactical_date).days
            if self.tactical_frequency == 'weekly':
                return days_since >= 5
            elif self.tactical_frequency == 'daily':
                return True
            else:
                return days_since >= 5
        return True

    def _generate_summary(self, strategic: Dict, tactical: Dict, execution: Dict) -> Dict[str, Any]:
        allocation = strategic.get('allocation', {})
        target_weights = tactical.get('target_weights', {})
        orders = execution.get('orders', [])
        total_orders = len(orders)
        buy_orders = sum((1 for o in orders if o.side == 'BUY'))
        sell_orders = sum((1 for o in orders if o.side == 'SELL'))
        if total_orders == 0:
            action_summary = '无需调整'
        elif total_orders <= 3:
            action_summary = f'小幅调整 ({total_orders} 笔订单)'
        elif total_orders <= 6:
            action_summary = f'中等调整 ({total_orders} 笔订单)'
        else:
            action_summary = f'大幅调仓 ({total_orders} 笔订单)'
        return {'regime': strategic.get('regime', {}).get('quadrant_name', '未知'), 'risk_level': strategic.get('regime', {}).get('risk_level', '未知'), 'top_allocation': max(allocation.items(), key=lambda x: x[1]) if allocation else ('N/A', 0), 'total_orders': total_orders, 'buy_orders': buy_orders, 'sell_orders': sell_orders, 'estimated_cost': execution.get('estimated_cost', 0), 'action_summary': action_summary, 'strategic_themes': strategic.get('investment_themes', []), 'factor_tilts': tactical.get('factor_tilts', {})}

    def _log_decision(self, results: Dict) -> None:
        log_entry = {'timestamp': results['timestamp'], 'as_of_date': results['as_of_date'], 'regime': results['summary'].get('regime'), 'action_summary': results['summary'].get('action_summary'), 'total_orders': results['summary'].get('total_orders')}
        self._decision_log.append(log_entry)
        if len(self._decision_log) > 500:
            self._decision_log = self._decision_log[-500:]

    def get_current_state(self) -> Dict[str, Any]:
        return {'strategic': self.strategic_layer.get_last_decision(), 'tactical': self.tactical_layer.get_last_decision(), 'pending_orders': self.execution_layer.get_pending_orders(), 'last_strategic_date': self._last_strategic_date.isoformat() if self._last_strategic_date else None, 'last_tactical_date': self._last_tactical_date.isoformat() if self._last_tactical_date else None}

    def get_decision_log(self, n: int=50) -> List[Dict]:
        return self._decision_log[-n:]

    def _extract_agent_context(self, market_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(market_data, dict):
            return {'macro_context': None, 'analyst_context': None, 'macro_source': None, 'analyst_source': None}
        scope_candidates: List[tuple[str, Dict[str, Any]]] = [('market_data', market_data)]
        for key in ('agent_context', 'agent_state', 'research_state', 'final_state', 'state', 'research_output'):
            value = market_data.get(key)
            if isinstance(value, dict):
                scope_candidates.append((key, value))
        macro_context: Dict[str, Any] = {}
        macro_source: Optional[str] = None
        for scope_name, scope in scope_candidates:
            for key in ('macro_context', 'macro_analysis', 'regime'):
                candidate = scope.get(key)
                if isinstance(candidate, dict) and candidate:
                    macro_source = macro_source or f'{scope_name}.{key}'
                    for field, value in candidate.items():
                        macro_context.setdefault(field, value)
            guidance = scope.get('macro_guidance')
            if guidance and 'macro_guidance' not in macro_context:
                macro_context['macro_guidance'] = self._truncate_text(guidance)
                macro_source = macro_source or f'{scope_name}.macro_guidance'
            alloc = scope.get('target_allocations')
            parsed_alloc = alloc if isinstance(alloc, dict) else self._parse_json_dict(alloc)
            if parsed_alloc and 'target_allocations' not in macro_context:
                macro_context['target_allocations'] = parsed_alloc
                macro_source = macro_source or f'{scope_name}.target_allocations'
        analyst_reports: Dict[str, str] = {}
        analyst_signals: Dict[str, Any] = {}
        factor_tilts: Optional[Dict[str, Any]] = None
        sector_weights: Optional[Dict[str, Any]] = None
        sentiment_score: Optional[float] = None
        analyst_source: Optional[str] = None
        for scope_name, scope in scope_candidates:
            reports = scope.get('analyst_reports')
            if isinstance(reports, dict):
                for key, value in reports.items():
                    if value and key not in analyst_reports:
                        analyst_reports[key] = self._truncate_text(value)
                        analyst_source = analyst_source or f'{scope_name}.analyst_reports'
            for key in ('market_report', 'sentiment_report', 'news_report', 'fundamentals_report'):
                value = scope.get(key)
                if value and key not in analyst_reports:
                    analyst_reports[key] = self._truncate_text(value)
                    analyst_source = analyst_source or f'{scope_name}.{key}'
            signals = scope.get('analyst_signals')
            if isinstance(signals, dict):
                for key, value in signals.items():
                    analyst_signals.setdefault(key, value)
                analyst_source = analyst_source or f'{scope_name}.analyst_signals'
            if factor_tilts is None:
                for key in ('factor_tilts', 'recommended_tilts', 'tactical_factor_tilts'):
                    value = scope.get(key)
                    if isinstance(value, dict) and value:
                        factor_tilts = value
                        analyst_source = analyst_source or f'{scope_name}.{key}'
                        break
            if sector_weights is None:
                for key in ('sector_weights', 'tactical_sector_weights'):
                    value = scope.get(key)
                    if isinstance(value, dict) and value:
                        sector_weights = value
                        analyst_source = analyst_source or f'{scope_name}.{key}'
                        break
            if sentiment_score is None:
                value = scope.get('sentiment_score', scope.get('analyst_sentiment'))
                if isinstance(value, (int, float)):
                    sentiment_score = float(value)
                    analyst_source = analyst_source or f'{scope_name}.sentiment_score'
        analyst_context = None
        if analyst_reports or analyst_signals or factor_tilts or sector_weights or (sentiment_score is not None):
            analyst_context = {'analyst_reports': analyst_reports, 'analyst_signals': analyst_signals, 'factor_tilts': factor_tilts or {}, 'sector_weights': sector_weights or {}}
            if sentiment_score is not None:
                analyst_context['sentiment_score'] = sentiment_score
        return {'macro_context': macro_context or None, 'analyst_context': analyst_context, 'macro_source': macro_source, 'analyst_source': analyst_source}

    @staticmethod
    def _parse_json_dict(raw: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _truncate_text(value: Any, max_chars: int=6000) -> str:
        text = str(value)
        if len(text) <= max_chars:
            return text
        return f'{text[:max_chars]}...'

    def force_strategic_review(self, as_of_date: str=None) -> Dict[str, Any]:
        return self.strategic_layer.make_decision(as_of_date=as_of_date, force=True)

    def force_tactical_review(self, strategic_decision: Dict=None, market_data: Dict=None, as_of_date: str=None) -> Dict[str, Any]:
        if strategic_decision is None:
            strategic_decision = self.strategic_layer.get_last_decision()
            if not strategic_decision:
                strategic_decision = self.strategic_layer.make_decision(as_of_date=as_of_date)
        agent_context = self._extract_agent_context(market_data)
        return self.tactical_layer.make_decision(strategic_decision=strategic_decision, market_data=market_data, as_of_date=as_of_date, analyst_context=agent_context.get('analyst_context'), force=True)
