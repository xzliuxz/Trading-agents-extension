import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = 'INFO'
    WARNING = 'WARNING'
    CRITICAL = 'CRITICAL'

class AlertType(Enum):
    RISK = 'RISK'
    PERFORMANCE = 'PERFORMANCE'
    SYSTEM = 'SYSTEM'
    MARKET = 'MARKET'
    EXECUTION = 'EXECUTION'

@dataclass
class Alert:
    id: str
    level: AlertLevel
    type: AlertType
    title: str
    message: str
    timestamp: datetime
    data: Dict[str, Any] = None
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {'id': self.id, 'level': self.level.value, 'type': self.type.value, 'title': self.title, 'message': self.message, 'timestamp': self.timestamp.isoformat(), 'data': self.data, 'acknowledged': self.acknowledged}

class AlertSystem:

    def __init__(self, config: Dict[str, Any]=None):
        self.config = config or {}
        self.alerts_enabled = self.config.get('alerts_enabled', True)
        self.max_alerts = self.config.get('max_alerts', 1000)
        self.thresholds = {'max_drawdown': self.config.get('alert_max_drawdown', 0.1), 'var_breach': self.config.get('alert_var_threshold', 0.03), 'correlation_spike': self.config.get('alert_correlation', 0.8), 'volatility_spike': self.config.get('alert_volatility', 0.25), 'execution_slippage': self.config.get('alert_slippage', 0.005)}
        self._alerts: List[Alert] = []
        self._alert_counter = 0
        self._callbacks: Dict[AlertLevel, List[Callable]] = {AlertLevel.INFO: [], AlertLevel.WARNING: [], AlertLevel.CRITICAL: []}
        self._cooldowns: Dict[str, datetime] = {}
        self._cooldown_duration = timedelta(minutes=self.config.get('alert_cooldown_minutes', 15))

    def add_callback(self, level: AlertLevel, callback: Callable[[Alert], None]) -> None:
        self._callbacks[level].append(callback)

    def create_alert(self, level: AlertLevel, alert_type: AlertType, title: str, message: str, data: Dict[str, Any]=None, cooldown_key: str=None) -> Optional[Alert]:
        if not self.alerts_enabled:
            return None
        if cooldown_key:
            last_alert = self._cooldowns.get(cooldown_key)
            if last_alert and datetime.now() - last_alert < self._cooldown_duration:
                logger.debug(f'警报在冷却期内: {cooldown_key}')
                return None
        self._alert_counter += 1
        alert = Alert(id=f'ALERT_{self._alert_counter:06d}', level=level, type=alert_type, title=title, message=message, timestamp=datetime.now(), data=data)
        self._alerts.append(alert)
        if len(self._alerts) > self.max_alerts:
            self._alerts = self._alerts[-self.max_alerts:]
        if cooldown_key:
            self._cooldowns[cooldown_key] = datetime.now()
        self._trigger_callbacks(alert)
        log_level = {AlertLevel.INFO: logging.INFO, AlertLevel.WARNING: logging.WARNING, AlertLevel.CRITICAL: logging.ERROR}
        logger.log(log_level[level], f'[{alert.id}] {title}: {message}')
        return alert

    def check_risk_alerts(self, current_drawdown: float=None, var_value: float=None, correlation: float=None, volatility: float=None) -> List[Alert]:
        alerts = []
        if current_drawdown is not None:
            if current_drawdown > self.thresholds['max_drawdown']:
                alert = self.create_alert(AlertLevel.CRITICAL if current_drawdown > self.thresholds['max_drawdown'] * 1.5 else AlertLevel.WARNING, AlertType.RISK, '回撤警报', f"当前回撤 {current_drawdown:.1%} 超过阈值 {self.thresholds['max_drawdown']:.1%}", {'current_drawdown': current_drawdown}, cooldown_key='drawdown_alert')
                if alert:
                    alerts.append(alert)
        if var_value is not None:
            if var_value > self.thresholds['var_breach']:
                alert = self.create_alert(AlertLevel.WARNING, AlertType.RISK, 'VaR 突破', f"VaR(95%) {var_value:.1%} 超过阈值 {self.thresholds['var_breach']:.1%}", {'var': var_value}, cooldown_key='var_alert')
                if alert:
                    alerts.append(alert)
        if correlation is not None:
            if correlation > self.thresholds['correlation_spike']:
                alert = self.create_alert(AlertLevel.WARNING, AlertType.RISK, '相关性飙升', f"平均相关性 {correlation:.2f} 超过阈值 {self.thresholds['correlation_spike']:.2f}", {'correlation': correlation}, cooldown_key='correlation_alert')
                if alert:
                    alerts.append(alert)
        if volatility is not None:
            if volatility > self.thresholds['volatility_spike']:
                alert = self.create_alert(AlertLevel.WARNING, AlertType.RISK, '波动率飙升', f"组合波动率 {volatility:.1%} 超过阈值 {self.thresholds['volatility_spike']:.1%}", {'volatility': volatility}, cooldown_key='volatility_alert')
                if alert:
                    alerts.append(alert)
        return alerts

    def check_performance_alerts(self, daily_return: float=None, weekly_return: float=None, tracking_error: float=None) -> List[Alert]:
        alerts = []
        if daily_return is not None and daily_return < -0.03:
            alert = self.create_alert(AlertLevel.CRITICAL if daily_return < -0.05 else AlertLevel.WARNING, AlertType.PERFORMANCE, '单日大幅亏损', f'今日收益 {daily_return:.1%}', {'daily_return': daily_return}, cooldown_key='daily_loss_alert')
            if alert:
                alerts.append(alert)
        if daily_return is not None and daily_return > 0.05:
            alert = self.create_alert(AlertLevel.INFO, AlertType.PERFORMANCE, '单日大幅盈利', f'今日收益 {daily_return:.1%}，考虑部分止盈', {'daily_return': daily_return}, cooldown_key='daily_gain_alert')
            if alert:
                alerts.append(alert)
        return alerts

    def check_market_alerts(self, vix_level: float=None, regime_change: bool=False, regime_name: str=None) -> List[Alert]:
        alerts = []
        if vix_level is not None:
            if vix_level > 30:
                alert = self.create_alert(AlertLevel.CRITICAL if vix_level > 40 else AlertLevel.WARNING, AlertType.MARKET, 'VIX 警报', f'VIX 达到 {vix_level:.1f}，市场恐慌', {'vix': vix_level}, cooldown_key='vix_alert')
                if alert:
                    alerts.append(alert)
        if regime_change:
            alert = self.create_alert(AlertLevel.WARNING, AlertType.MARKET, '宏观体制变化', f'宏观环境切换至: {regime_name}', {'new_regime': regime_name}, cooldown_key='regime_change_alert')
            if alert:
                alerts.append(alert)
        return alerts

    def check_execution_alerts(self, slippage: float=None, failed_orders: int=None) -> List[Alert]:
        alerts = []
        if slippage is not None:
            if slippage > self.thresholds['execution_slippage']:
                alert = self.create_alert(AlertLevel.WARNING, AlertType.EXECUTION, '执行滑点过大', f'滑点 {slippage:.2%} 超过阈值', {'slippage': slippage}, cooldown_key='slippage_alert')
                if alert:
                    alerts.append(alert)
        if failed_orders is not None and failed_orders > 0:
            alert = self.create_alert(AlertLevel.WARNING if failed_orders < 3 else AlertLevel.CRITICAL, AlertType.EXECUTION, '订单执行失败', f'{failed_orders} 笔订单执行失败', {'failed_orders': failed_orders})
            if alert:
                alerts.append(alert)
        return alerts

    def _trigger_callbacks(self, alert: Alert) -> None:
        for callback in self._callbacks.get(alert.level, []):
            try:
                callback(alert)
            except Exception as e:
                logger.error(f'警报回调执行失败: {e}')

    def acknowledge_alert(self, alert_id: str) -> bool:
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_active_alerts(self, level: AlertLevel=None, alert_type: AlertType=None) -> List[Alert]:
        alerts = [a for a in self._alerts if not a.acknowledged]
        if level:
            alerts = [a for a in alerts if a.level == level]
        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]
        return alerts

    def get_all_alerts(self, n: int=100) -> List[Alert]:
        return self._alerts[-n:]

    def get_alert_summary(self) -> Dict[str, Any]:
        active = self.get_active_alerts()
        summary = {'total_alerts': len(self._alerts), 'active_alerts': len(active), 'by_level': {'CRITICAL': len([a for a in active if a.level == AlertLevel.CRITICAL]), 'WARNING': len([a for a in active if a.level == AlertLevel.WARNING]), 'INFO': len([a for a in active if a.level == AlertLevel.INFO])}, 'by_type': {t.value: len([a for a in active if a.type == t]) for t in AlertType}, 'recent': [a.to_dict() for a in active[-5:]]}
        return summary

    def clear_old_alerts(self, days: int=7) -> int:
        cutoff = datetime.now() - timedelta(days=days)
        old_count = len(self._alerts)
        self._alerts = [a for a in self._alerts if a.timestamp > cutoff or not a.acknowledged]
        return old_count - len(self._alerts)
