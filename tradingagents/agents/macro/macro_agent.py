import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import numpy as np
import yfinance as yf
import pandas as pd
import os
from tradingagents.agents.utils.longform import ensure_long_form_report, resolve_longform_requirements
from tradingagents.utils.llm_retry import invoke_with_retry
from tradingagents.agents.macro.indicators.regime_detector import RegimeDetector
from tradingagents.agents.macro.indicators.yield_curve import YieldCurveAnalyzer
from tradingagents.agents.macro.indicators.credit_spreads import CreditSpreadAnalyzer
from tradingagents.agents.macro.indicators.fund_flows import FundFlowAnalyzer
logger = logging.getLogger(__name__)

def _setup_yfinance_cache():
    try:
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.cache', 'yfinance')
        os.makedirs(cache_dir, exist_ok=True)
        yf.set_tz_cache_location(cache_dir)
    except Exception:
        pass
_setup_yfinance_cache()

def create_macro_analyst(llm, config: Dict[str, Any]=None):
    default_macro_assets = {'growth': ['SPY', 'QQQ'], 'bonds': ['TLT', 'IEF'], 'commodities': ['DBC', 'USO'], 'gold': ['GLD', 'IAU']}
    macro_config = config or {}
    macro_assets = macro_config.get('macro_assets', default_macro_assets)
    lookback_days = macro_config.get('macro_lookback_days', 120)
    momentum_window = macro_config.get('macro_momentum_window', 60)
    growth_diff_threshold = macro_config.get('macro_growth_diff_threshold', 5.0)
    inflation_momentum_threshold = macro_config.get('macro_inflation_momentum_threshold', 5.0)
    cache_days = macro_config.get('macro_cache_days', 365)
    base_allocations = macro_config.get('all_weather_base_weights', {'SPY': 0.3, 'TLT': 0.4, 'GLD': 0.15, 'DBC': 0.15})
    default_quadrant_allocations = {'rising_growth_rising_inflation': {'SPY': 0.4, 'TLT': 0.2, 'GLD': 0.15, 'DBC': 0.25}, 'rising_growth_falling_inflation': {'SPY': 0.45, 'TLT': 0.35, 'GLD': 0.1, 'DBC': 0.1}, 'falling_growth_rising_inflation': {'SPY': 0.15, 'TLT': 0.25, 'GLD': 0.3, 'DBC': 0.3}, 'falling_growth_falling_inflation': {'SPY': 0.2, 'TLT': 0.5, 'GLD': 0.2, 'DBC': 0.1}}
    quadrant_allocations = macro_config.get('macro_quadrant_allocations', default_quadrant_allocations)
    macro_min_words, macro_min_sections, macro_max_rounds = resolve_longform_requirements(macro_config, role_key='macro', default_words=320, default_sections=4)

    def _first_ticker(category: str, default: str) -> str:
        tickers = macro_assets.get(category) or []
        return tickers[0] if tickers else default
    representatives = {'growth': _first_ticker('growth', 'SPY'), 'bonds': _first_ticker('bonds', 'TLT'), 'commodities': _first_ticker('commodities', 'DBC'), 'gold': _first_ticker('gold', 'GLD')}

    def _unique_tickers(items):
        seen = set()
        ordered = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered
    price_cache = {'data': None, 'start': None, 'end': None, 'tickers': None}

    def _fetch_macro_data(trade_date: str) -> pd.DataFrame:
        end_date = datetime.strptime(trade_date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=lookback_days + 30)
        all_tickers = _unique_tickers(list(representatives.values()))
        req_start = start_date
        req_end = end_date + timedelta(days=1)
        cached_data = price_cache['data']
        if cached_data is not None and price_cache['tickers'] == all_tickers and (price_cache['start'] <= req_start) and (price_cache['end'] >= req_end):
            prices = cached_data
        else:
            fetch_end = end_date + timedelta(days=cache_days)
            try:
                data = yf.download(all_tickers, start=req_start.strftime('%Y-%m-%d'), end=(fetch_end + timedelta(days=1)).strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
            except Exception as e:
                print(f'Warning: Failed to fetch macro data: {e}')
                return pd.DataFrame()
            if data is None or data.empty:
                return pd.DataFrame()
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                if 'Close' not in data.columns:
                    return pd.DataFrame()
                prices = data[['Close']].rename(columns={'Close': all_tickers[0]})
            if prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)
            price_cache['data'] = prices
            price_cache['start'] = req_start
            price_cache['end'] = fetch_end
            price_cache['tickers'] = all_tickers
        if prices.empty:
            return prices
        return prices.loc[(prices.index >= req_start) & (prices.index < req_end)]

    def _calculate_momentum(prices: pd.DataFrame, window: int=60) -> Dict[str, float]:
        if prices.empty:
            return {}
        momentum = {}
        for col in prices.columns:
            try:
                series = prices[col].dropna()
                if len(series) >= window:
                    ret = (series.iloc[-1] / series.iloc[-window] - 1) * 100
                    momentum[col] = float(ret)
            except Exception:
                pass
        return momentum

    def _calculate_adaptive_threshold(prices: pd.DataFrame, window: int=60) -> float:
        spy_col = representatives['growth']
        tlt_col = representatives['bonds']
        if prices.empty or spy_col not in prices.columns or tlt_col not in prices.columns:
            return growth_diff_threshold
        try:
            spy_ret = prices[spy_col].pct_change().dropna()
            tlt_ret = prices[tlt_col].pct_change().dropna()
            spread_ret = spy_ret - tlt_ret
            if len(spread_ret) < window:
                return growth_diff_threshold
            recent = spread_ret.iloc[-window:]
            ann_vol = float(recent.std() * np.sqrt(252) * 100)
            adaptive = max(2.0, min(ann_vol * 0.5, 15.0))
            return adaptive
        except Exception:
            return growth_diff_threshold

    def _fetch_fred_signals(trade_date: str) -> Dict[str, Any]:
        result = {'available': False, 'growth_signal': None, 'inflation_signal': None, 'growth_confidence': 0.0, 'inflation_confidence': 0.0, 'reasoning': '', 'components': {}}
        try:
            from tradingagents.agents.macro.indicators.economic import EconomicIndicators
            econ = EconomicIndicators(data_source='fred')
            growth = econ.get_growth_signal(as_of_date=trade_date)
            inflation = econ.get_inflation_signal(as_of_date=trade_date)
            growth_map = {'BULLISH': 'rising', 'NEUTRAL': 'neutral', 'BEARISH': 'falling'}
            inflation_map = {'RISING': 'rising', 'ELEVATED': 'rising', 'STABLE': 'neutral', 'ON_TARGET': 'neutral', 'FALLING': 'falling'}
            g_sig = growth.get('signal', 'NEUTRAL')
            i_sig = inflation.get('signal', 'STABLE')
            result['available'] = True
            result['growth_signal'] = growth_map.get(g_sig, 'neutral')
            result['inflation_signal'] = inflation_map.get(i_sig, 'neutral')
            result['growth_confidence'] = growth.get('confidence', 0.5)
            result['inflation_confidence'] = inflation.get('confidence', 0.5)
            result['reasoning'] = growth.get('reasoning', '')
            result['components'] = {'growth': growth, 'inflation': inflation}
            logger.info('[Macro] FRED signals: growth=%s (%.0f%%), inflation=%s (%.0f%%)', result['growth_signal'], result['growth_confidence'] * 100, result['inflation_signal'], result['inflation_confidence'] * 100)
        except ImportError:
            logger.info('[Macro] fredapi not installed, skipping FRED signals')
        except Exception as e:
            logger.info('[Macro] FRED fetch failed: %s', e)
        return result

    def _determine_quadrant(momentum: Dict[str, float], prices: pd.DataFrame=None, trade_date: str=None) -> Dict[str, Any]:
        spy_mom = momentum.get(representatives['growth'], 0)
        tlt_mom = momentum.get(representatives['bonds'], 0)
        dbc_mom = momentum.get(representatives['commodities'], 0)
        gld_mom = momentum.get(representatives['gold'], 0)
        if prices is not None and (not prices.empty):
            g_threshold = _calculate_adaptive_threshold(prices, window=momentum_window)
        else:
            g_threshold = growth_diff_threshold
        i_threshold = inflation_momentum_threshold
        growth_diff = spy_mom - tlt_mom
        if growth_diff > g_threshold:
            price_growth = 'rising'
        elif growth_diff < -g_threshold:
            price_growth = 'falling'
        else:
            price_growth = 'neutral'
        if dbc_mom > i_threshold:
            price_inflation = 'rising'
        elif dbc_mom < -i_threshold and gld_mom > dbc_mom:
            price_inflation = 'falling'
        else:
            price_inflation = 'neutral'
        fred = _fetch_fred_signals(trade_date) if trade_date else {'available': False}
        fred_available = fred.get('available', False)

        def _fuse(price_sig: str, fred_sig, fred_conf: float, price_weight=0.4, fred_weight=0.6) -> str:
            signal_scores = {'rising': 1.0, 'neutral': 0.0, 'falling': -1.0}
            p_score = signal_scores.get(price_sig, 0.0)
            if fred_available and fred_sig is not None:
                f_score = signal_scores.get(fred_sig, 0.0)
                combined = price_weight * p_score + fred_weight * f_score * fred_conf
            else:
                combined = p_score
            if combined > 0.25:
                return 'rising'
            elif combined < -0.25:
                return 'falling'
            else:
                return 'neutral'
        growth_signal = _fuse(price_growth, fred.get('growth_signal'), fred.get('growth_confidence', 0.5))
        inflation_signal = _fuse(price_inflation, fred.get('inflation_signal'), fred.get('inflation_confidence', 0.5))
        if growth_signal == 'neutral' and inflation_signal == 'neutral':
            quadrant = 'neutral'
        elif growth_signal == 'rising' and inflation_signal == 'rising':
            quadrant = 'rising_growth_rising_inflation'
        elif growth_signal == 'rising' and inflation_signal in ('falling', 'neutral'):
            quadrant = 'rising_growth_falling_inflation'
        elif growth_signal == 'falling' and inflation_signal == 'rising':
            quadrant = 'falling_growth_rising_inflation'
        elif growth_signal == 'falling' and inflation_signal in ('falling', 'neutral'):
            quadrant = 'falling_growth_falling_inflation'
        elif growth_signal == 'neutral' and inflation_signal == 'rising':
            quadrant = 'neutral'
        elif growth_signal == 'neutral' and inflation_signal == 'falling':
            quadrant = 'neutral'
        else:
            quadrant = 'neutral'
        if quadrant == 'falling_growth_rising_inflation':
            risk_level = 'high'
        elif growth_signal == 'falling':
            risk_level = 'medium'
        elif quadrant == 'neutral':
            risk_level = 'medium'
        else:
            risk_level = 'low'
        position_cap_map = {'high': 15.0, 'medium': 30.0, 'low': 50.0}
        position_cap_pct = position_cap_map.get(risk_level, 30.0)
        return {'quadrant': quadrant, 'growth_signal': growth_signal, 'inflation_signal': inflation_signal, 'risk_level': risk_level, 'momentum': momentum, 'adaptive_threshold': g_threshold if prices is not None else growth_diff_threshold, 'fred_available': fred_available, 'fred_components': fred.get('components', {}), 'fred_reasoning': fred.get('reasoning', ''), 'position_cap_pct': position_cap_pct}

    def _calculate_allocations(quadrant_info: Dict[str, Any]) -> Dict[str, float]:
        quadrant = quadrant_info['quadrant']
        if quadrant in quadrant_allocations:
            return quadrant_allocations[quadrant]
        return base_allocations

    def _generate_guidance(quadrant_info: Dict[str, Any], allocations: Dict[str, float]) -> str:
        quadrant = quadrant_info['quadrant']
        risk_level = quadrant_info['risk_level']
        guidance_templates = {'rising_growth_rising_inflation': 'Macro Guidance: The regime is rising growth with rising inflation. Expansion is intact but inflation pressure is elevated, favoring equities and commodities. Recommendation: keep a pro-risk stance with disciplined risk controls.', 'rising_growth_falling_inflation': 'Macro Guidance: The regime is rising growth with moderating inflation (Goldilocks). This is generally supportive for risk assets, especially equities with selective duration exposure. Recommendation: allow moderate risk increase while preserving stop-discipline.', 'falling_growth_rising_inflation': 'Macro Guidance: The regime indicates stagflation risk (slowing growth with elevated inflation). This is a difficult backdrop where both equities and duration can face pressure. Recommendation: stay defensive, emphasize gold/commodities, and reduce equity risk.', 'falling_growth_falling_inflation': 'Macro Guidance: The regime is falling growth with falling inflation (disinflation/slowdown). Defensive assets such as high-quality duration and gold tend to improve on a relative basis. Recommendation: increase defensiveness and keep equity exposure selective.'}
        base_guidance = guidance_templates.get(quadrant, 'Macro Guidance: Signals are mixed/neutral. Maintain balanced positioning and avoid aggressive directional risk.')
        risk_note = ''
        if risk_level == 'high':
            risk_note = '\nRisk level: HIGH. Downstream agents should prioritize capital preservation and tighter risk controls.'
        elif risk_level == 'medium':
            risk_note = '\nRisk level: MEDIUM. Balance return-seeking with clear downside controls.'
        else:
            risk_note = '\nRisk level: LOW. Risk-taking is acceptable but position sizing and invalidation rules remain mandatory.'
        allocation_str = ', '.join([f'{k}: {v:.0%}' for k, v in allocations.items()])
        return f'{base_guidance}{risk_note}\n\nTarget Allocation: {allocation_str}'

    def macro_analyst_node(state) -> dict:
        trade_date = state['trade_date']
        company = state['company_of_interest']
        print(f"\n{'=' * 60}")
        print(f'[Macro Analyst] 执行宏观分析 @ {trade_date}')
        print(f"{'=' * 60}")
        prices = _fetch_macro_data(trade_date)
        if prices.empty:
            print('[Macro] 警告: 宏观数据获取失败，使用基础配置并跳过宏观 LLM')
            quadrant_info = {'quadrant': 'neutral', 'growth_signal': 'neutral', 'inflation_signal': 'neutral', 'risk_level': 'medium', 'momentum': {}}
            allocations = base_allocations
            guidance = f"Macro Guidance: Macro data is unavailable (network/DNS/certificate issue). Use baseline allocation and keep neutral risk posture until data quality is restored.\n\nTarget Allocation: {', '.join([f'{k}: {v:.0%}' for k, v in allocations.items()])}"
            analysis_summary = guidance
            analysis_summary = ensure_long_form_report(llm=llm, report_text=analysis_summary, role_name='Macro Analyst', min_words=macro_min_words, min_sections=macro_min_sections, citation_style_hint='[Source: MA1]', max_expansion_rounds=macro_max_rounds)
            macro_analysis = {'quadrant': quadrant_info['quadrant'], 'growth_signal': quadrant_info['growth_signal'], 'inflation_signal': quadrant_info['inflation_signal'], 'risk_level': quadrant_info['risk_level'], 'analysis_summary': analysis_summary}
            print(f'\n[Macro] 分析完成（降级模式），宏观指导已注入 State')
            print(f"{'=' * 60}\n")
            return {'macro_analysis': macro_analysis, 'target_allocations': json.dumps(allocations), 'macro_guidance': guidance}
        momentum = _calculate_momentum(prices, window=momentum_window)
        print(f'[Macro] 资产动量 (60日): {momentum}')
        quadrant_info = _determine_quadrant(momentum, prices=prices, trade_date=trade_date)
        print(f"[Macro] 象限判断: {quadrant_info['quadrant']}")
        if quadrant_info.get('fred_available'):
            print(f"[Macro] FRED 经济指标已融合 (adaptive threshold: {quadrant_info.get('adaptive_threshold', 'N/A'):.1f})")
        else:
            print(f"[Macro] FRED 不可用，使用纯价格动量 (threshold: {quadrant_info.get('adaptive_threshold', 'N/A'):.1f})")
        print(f"[Macro] 仓位硬约束上限: {quadrant_info.get('position_cap_pct', 30):.0f}%")
        print(f"[Macro] 增长信号: {quadrant_info['growth_signal']}, 通胀信号: {quadrant_info['inflation_signal']}")
        print(f"[Macro] 风险等级: {quadrant_info['risk_level']}")
        allocations = _calculate_allocations(quadrant_info)
        print(f'[Macro] 目标配置: {allocations}')
        guidance = _generate_guidance(quadrant_info, allocations)
        _deep_indicators = {}
        try:
            _yc = YieldCurveAnalyzer()
            _yc_signal = _yc.get_growth_signal(trade_date)
            _yc_spread = _yc.calculate_term_spread(trade_date, '2y', '10y') if hasattr(_yc, 'calculate_term_spread') else {}
            _yc_real = _yc.calculate_real_rate(trade_date) if hasattr(_yc, 'calculate_real_rate') else {}
            _deep_indicators['yield_curve'] = {'signal': _yc_signal.get('signal', 'N/A'), 'confidence': round(_yc_signal.get('confidence', 0), 2), 'reasoning': _yc_signal.get('reasoning', ''), 'term_spread_2y10y': _yc_spread.get('spread', 'N/A'), 'term_spread_signal': _yc_spread.get('signal', 'N/A'), 'real_rate': _yc_real.get('real_rate', 'N/A'), 'real_rate_signal': _yc_real.get('signal', 'N/A'), 'nominal_10y': _yc_real.get('nominal_rate', 'N/A'), 'breakeven_inflation': _yc_real.get('breakeven_inflation', 'N/A')}
        except Exception as e:
            _deep_indicators['yield_curve'] = {'error': str(e)}
        try:
            _cs = CreditSpreadAnalyzer()
            _cs_signal = _cs.get_risk_appetite_signal(trade_date)
            _cs_comps = _cs_signal.get('components', {})
            _deep_indicators['credit_spreads'] = {'signal': _cs_signal.get('signal', 'N/A'), 'confidence': round(_cs_signal.get('confidence', 0), 2), 'reasoning': _cs_signal.get('reasoning', ''), 'ig_spread': _cs_comps.get('ig_spread', {}).get('spread', 'N/A'), 'ig_signal': _cs_comps.get('ig_spread', {}).get('signal', 'N/A'), 'hy_spread': _cs_comps.get('hy_spread', {}).get('spread', 'N/A'), 'hy_signal': _cs_comps.get('hy_spread', {}).get('signal', 'N/A'), 'financial_stress': _cs_comps.get('financial_stress', {}).get('signal', 'N/A')}
        except Exception as e:
            _deep_indicators['credit_spreads'] = {'error': str(e)}
        try:
            _ff = FundFlowAnalyzer()
            _ff_signal = _ff.get_flow_signal(trade_date)
            _ff_comps = _ff_signal.get('components', {})
            _deep_indicators['fund_flows'] = {'signal': _ff_signal.get('signal', 'N/A'), 'confidence': round(_ff_signal.get('confidence', 0), 2), 'reasoning': _ff_signal.get('reasoning', ''), 'spy_flow': _ff_comps.get('spy_flow', {}).get('signal', 'N/A'), 'tlt_flow': _ff_comps.get('tlt_flow', {}).get('signal', 'N/A'), 'gld_flow': _ff_comps.get('gld_flow', {}).get('signal', 'N/A'), 'cycle_phase': _ff_comps.get('sector_rotation', {}).get('cycle_phase', 'N/A'), 'rotation_signal': _ff_comps.get('relative_strength', {}).get('rotation_signal', 'N/A')}
        except Exception as e:
            _deep_indicators['fund_flows'] = {'error': str(e)}
        if llm:
            llm_prompt = f"""You are a macro strategist writing a briefing for a research desk at a global multi-asset fund.\nYour job is NOT to describe data. Your job is to construct a causal narrative: WHY are conditions the way they are, WHAT does it mean mechanically for asset prices, and WHAT are the explicit decision rules that would force a change in positioning.\n\nTrade date: {trade_date}\nEquity under review: {company}\n\n=== QUANTITATIVE INPUTS ===\n\nAsset momentum (60-day returns):\n{json.dumps(momentum, indent=2)}\n\nRegime classification:\n- Quadrant: {quadrant_info['quadrant']}\n- Growth signal: {quadrant_info['growth_signal']}\n- Inflation signal: {quadrant_info['inflation_signal']}\n- Risk level: {quadrant_info['risk_level']}\n- FRED data available: {quadrant_info.get('fred_available', False)}\n\nYield curve & rates:\n{json.dumps(_deep_indicators.get('yield_curve', {}), indent=2, default=str)}\n\nCredit market signals:\n{json.dumps(_deep_indicators.get('credit_spreads', {}), indent=2, default=str)}\n\nFund flows & sector rotation:\n{json.dumps(_deep_indicators.get('fund_flows', {}), indent=2, default=str)}\n\nTarget multi-asset allocation (regime-based):\n{json.dumps(allocations, indent=2)}\n\n=== REQUIRED OUTPUT STRUCTURE (minimum {macro_min_words} words, exactly these sections) ===\n\n## 1. Regime Diagnosis\nState the current macro quadrant. Then explain the CAUSAL MECHANISM: what combination of growth impulse, inflation trajectory, and liquidity conditions produces this regime? Do not just label it — explain the transmission chain.\n\n## 2. Debt Cycle & Liquidity Position\nAssess where we are in the short-term debt cycle (expansion / late-cycle / contraction) and long-term debt cycle based on the yield curve shape, real rate level, and credit spread data above. What does the credit market say about systemic stress vs. calm? How does the current financial stress level compare historically?\n\n## 3. Historical Analogue\nIdentify the 2-3 most similar macro environments from history (e.g. 1995-1997, 2004-2006, 2010-2012, 2016-2018). For EACH analogue state: (a) what the quadrant/rates/credit looked like then, (b) how long the regime lasted, (c) what equities/bonds/gold/commodities did during that period. Be specific with numbers.\n\n## 4. Causal Chain: Macro → {company}\nWrite the explicit causal transmission from macro conditions to this specific stock. Format:\nReal rate trajectory → discount rate → valuation multiple → price direction\nGrowth regime → revenue/earnings cycle → EPS trajectory → price direction\nRisk appetite (credit spreads / fund flows) → risk premium → positioning → price direction\nAll three chains must be written out. Conclude with: "The macro environment is [TAILWIND / HEADWIND / NEUTRAL] for {company} because [one sentence causal summary]."\n\n## 5. Multi-Asset Allocation Implication\nGiven the regime, what is the appropriate multi-asset positioning? Justify each asset class weight in terms of what environment it is being held to protect against. Then explain specifically where {company} fits within the equity sleeve — what type of equity exposure does it represent (growth/value/defensive/cyclical) and is that the right type for this quadrant?\n\n## 6. Explicit Risk Triggers & Decision Rules\nList exactly 5 triggers. Each must follow this format:\nTRIGGER [n]: [Specific measurable condition] → [Exact portfolio action]\nExample: "TRIGGER 1: US 10Y real yield breaks above 2.5% and holds for 10 trading days → reduce equity sleeve by 30%, cut {company} to 0%"\nThese must be specific, measurable, and actionable. No vague language.\n\n## 7. Macro Verdict for {company}\nOne paragraph. State explicitly: BUY / REDUCE / AVOID from a pure macro perspective, with a time horizon. Give the single most important condition that sustains the current call, and the single condition that would invalidate it.\n\n=== CITATION RULE ===\nEvery factual claim (rate level, spread value, growth figure) must cite [Source: X] in the same sentence. Remove any factual statement without a citation.\n"""
            try:
                llm_response = invoke_with_retry(llm, llm_prompt, caller_name='Macro Analyst', fallback_content=guidance)
                analysis_summary = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                analysis_summary = ensure_long_form_report(llm=llm, report_text=analysis_summary, role_name='Macro Analyst', min_words=macro_min_words, min_sections=macro_min_sections, citation_style_hint='[Source: MA1]', max_expansion_rounds=macro_max_rounds)
            except Exception as e:
                print(f'[Macro] LLM 调用失败: {e}')
                analysis_summary = guidance
        else:
            analysis_summary = guidance
        macro_analysis = {'quadrant': quadrant_info['quadrant'], 'growth_signal': quadrant_info['growth_signal'], 'inflation_signal': quadrant_info['inflation_signal'], 'risk_level': quadrant_info['risk_level'], 'analysis_summary': analysis_summary, 'position_cap_pct': quadrant_info.get('position_cap_pct', 30.0), 'fred_available': quadrant_info.get('fred_available', False)}
        print(f'\n[Macro] 分析完成，宏观指导已注入 State')
        print(f"{'=' * 60}\n")
        return {'macro_analysis': macro_analysis, 'target_allocations': json.dumps(allocations), 'macro_guidance': guidance}
    return macro_analyst_node
