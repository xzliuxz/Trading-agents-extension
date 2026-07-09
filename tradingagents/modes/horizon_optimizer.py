from typing import List, Optional, Tuple

def calculate_optimal_horizon(backtest_weeks: int) -> str:
    if backtest_weeks < 1:
        return '1w'
    if backtest_weeks < 12:
        return '1w'
    if backtest_weeks >= 52:
        return '12w'
    if backtest_weeks % 12 == 0:
        return '12w'
    remainder = backtest_weeks % 12
    if remainder >= 6:
        return '12w'
    return '4w'

def get_research_update_schedule(backtest_weeks: int, horizon: Optional[str]=None) -> List[int]:
    if backtest_weeks < 1:
        return [0]
    if horizon is None:
        horizon = calculate_optimal_horizon(backtest_weeks)
    horizon_weeks = _horizon_to_weeks(horizon)
    schedule = [0]
    current = horizon_weeks
    while current < backtest_weeks:
        schedule.append(current)
        current += horizon_weeks
    if schedule[-1] != backtest_weeks:
        schedule.append(backtest_weeks)
    return schedule

def explain_horizon_choice(backtest_weeks: int) -> Tuple[str, str]:
    horizon = calculate_optimal_horizon(backtest_weeks)
    schedule = get_research_update_schedule(backtest_weeks, horizon)
    num_updates = len(schedule)
    if backtest_weeks < 12:
        reason = f'Short-term backtest ({backtest_weeks}w): using weekly (1w) research updates for maximum granularity. {num_updates} research runs scheduled.'
    elif backtest_weeks >= 52:
        reason = f'Long-term backtest ({backtest_weeks}w): using quarterly (12w) research horizon to balance depth and efficiency. {num_updates} research runs scheduled at weeks {schedule}.'
    elif horizon == '12w':
        if backtest_weeks % 12 == 0:
            reason = f'Medium-term backtest ({backtest_weeks}w): using quarterly (12w) horizon (evenly divisible by 12). {num_updates} research runs scheduled at weeks {schedule}.'
        else:
            remainder = backtest_weeks % 12
            reason = f'Medium-term backtest ({backtest_weeks}w): using quarterly (12w) horizon (remainder {remainder}w >= 6w threshold). {num_updates} research runs scheduled at weeks {schedule}.'
    else:
        reason = f'Medium-term backtest ({backtest_weeks}w): using monthly (4w) horizon for finer granularity (12w remainder < 6w). {num_updates} research runs scheduled at weeks {schedule}.'
    return (horizon, reason)

def _horizon_to_weeks(horizon: str) -> int:
    mapping = {'1w': 1, '4w': 4, '12w': 12}
    return mapping.get(horizon, 4)
