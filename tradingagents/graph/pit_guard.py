from __future__ import annotations
import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)

class PITGuard:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('pit_backtest_enforce', False)
        self.strict_mode = config.get('research_pit_strict_mode', False)
        self.auto_warn_in_backtest = config.get('pit_auto_warn_in_backtest', True)
        self.pit_matrix = config.get('pit_vendor_support_matrix', {})
        self.lag_days = config.get('pit_module_lag_days', {})
        self._violations: List[Dict[str, Any]] = []
        self._warned_combos: set = set()

    def is_backtest_mode(self, trade_date: str) -> bool:
        try:
            td = datetime.strptime(trade_date, '%Y-%m-%d').date()
            today = datetime.now().date()
            return td < today
        except Exception:
            return False

    def check_vendor(self, vendor: str, method: str, trade_date: str) -> Dict[str, Any]:
        if not self.is_backtest_mode(trade_date):
            return {'allowed': True, 'pit_safe': True, 'warning': None}
        pit_safe = self.pit_matrix.get(vendor, False)
        if pit_safe:
            return {'allowed': True, 'pit_safe': True, 'warning': None}
        combo_key = f'{vendor}/{method}'
        warning_msg = f"[PIT Guard] ⚠️ Vendor '{vendor}' is NOT PIT-safe for method '{method}' on {trade_date}. Data may contain future information."
        self._violations.append({'vendor': vendor, 'method': method, 'trade_date': trade_date, 'severity': 'block' if self.enabled and self.strict_mode else 'warn'})
        if self.enabled:
            if self.strict_mode:
                logger.warning(warning_msg + ' BLOCKED in strict mode.')
                return {'allowed': False, 'pit_safe': False, 'warning': warning_msg + ' BLOCKED.'}
            else:
                logger.warning(warning_msg)
                warnings.warn(warning_msg, UserWarning, stacklevel=2)
                return {'allowed': True, 'pit_safe': False, 'warning': warning_msg}
        elif self.auto_warn_in_backtest:
            if combo_key not in self._warned_combos:
                self._warned_combos.add(combo_key)
                logger.warning('%s Enable pit_backtest_enforce=True to block this.', warning_msg)
            return {'allowed': True, 'pit_safe': False, 'warning': warning_msg}
        else:
            return {'allowed': True, 'pit_safe': False, 'warning': None}

    def get_safe_vendors(self) -> list:
        return [v for v, safe in self.pit_matrix.items() if safe]

    def get_violations(self) -> list:
        return self._violations.copy()

    def get_violation_count(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for v in self._violations:
            key = v['vendor']
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_violation_summary(self) -> str:
        if not self._violations:
            return 'No PIT violations detected.'
        n = len(self._violations)
        blocked = sum((1 for v in self._violations if v['severity'] == 'block'))
        warned = n - blocked
        lines = [f'PIT Safety Summary: {n} violation(s) ({blocked} blocked, {warned} warned)']
        vendor_counts = self.get_violation_count()
        for vendor, count in sorted(vendor_counts.items(), key=lambda x: -x[1]):
            safe_str = 'PIT-safe' if self.pit_matrix.get(vendor, False) else 'NOT PIT-safe'
            lines.append(f'  - {vendor}: {count} call(s) [{safe_str}]')
        shown = set()
        for v in self._violations:
            key = f"{v['vendor']}/{v['method']}"
            if key not in shown:
                shown.add(key)
                lines.append(f"    [{v['severity'].upper()}] {v['vendor']}/{v['method']} @ {v['trade_date']}")
            if len(shown) >= 10:
                remaining = len(set((f"{x['vendor']}/{x['method']}" for x in self._violations))) - 10
                if remaining > 0:
                    lines.append(f'    ... and {remaining} more unique combinations')
                break
        if not self.enabled:
            lines.append('')
            lines.append('  💡 TIP: Set pit_backtest_enforce=True in config to block non-PIT-safe data sources during backtests.')
        return '\n'.join(lines)

    def suggest_safe_config(self, trade_date: str) -> Dict[str, str]:
        safe = self.get_safe_vendors()
        suggestions = {}
        if 'local' in safe:
            suggestions['core_stock_apis'] = 'local'
            suggestions['technical_indicators'] = 'local'
            suggestions['fundamental_data'] = 'local'
            suggestions['news_data'] = 'local'
        elif 'fred' in safe:
            suggestions['macro_data'] = 'fred'
        return suggestions

def create_pit_guard(config: Dict[str, Any]) -> PITGuard:
    return PITGuard(config)
