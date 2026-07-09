from __future__ import annotations
import json
import logging
from typing import Any, Dict
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.utils.llm_retry import invoke_with_retry
logger = logging.getLogger(__name__)

def create_enhanced_macro_node(llm, config: Dict[str, Any]=None):
    from tradingagents.agents.macro.macro_agent import create_macro_analyst
    from tradingagents.agents.utils.longform import ensure_long_form_report, resolve_longform_requirements
    cfg = config or {}
    use_enhanced = cfg.get('use_enhanced_regime_detector', False)
    base_macro_node = create_macro_analyst(llm, cfg)
    if not use_enhanced:
        return base_macro_node
    macro_min_words, macro_min_sections, macro_max_rounds = resolve_longform_requirements(cfg, role_key='macro', default_words=320, default_sections=4)

    def enhanced_macro_node(state: AgentState) -> Dict[str, Any]:
        trade_date = state['trade_date']
        company = state['company_of_interest']
        print(f"\n{'=' * 60}")
        print(f'[Enhanced Macro] Running RegimeDetector @ {trade_date}')
        print(f"{'=' * 60}")
        regime_result = None
        try:
            from tradingagents.agents.macro.indicators import RegimeDetector
            detector = RegimeDetector(cfg)
            regime_result = detector.detect_regime(as_of_date=trade_date)
        except Exception as e:
            logger.warning(f'RegimeDetector failed, falling back to simple: {e}')
            print(f'[Enhanced Macro] RegimeDetector failed: {e}')
            return base_macro_node(state)
        if not regime_result or not regime_result.get('quadrant'):
            return base_macro_node(state)
        quadrant = regime_result['quadrant']
        confidence = regime_result.get('confidence', 0.5)
        growth_signal = regime_result.get('growth_signal', 'neutral')
        inflation_signal = regime_result.get('inflation_signal', 'neutral')
        risk_level = regime_result.get('risk_level', 'medium')
        warnings_list = regime_result.get('warnings', [])
        recommendations = regime_result.get('recommendations', {})
        default_allocations = cfg.get('all_weather_base_weights', {'SPY': 0.3, 'TLT': 0.4, 'GLD': 0.15, 'DBC': 0.15})
        quadrant_allocations = cfg.get('macro_quadrant_allocations', {})
        allocations = quadrant_allocations.get(quadrant, default_allocations)
        factor_guidance = ''
        try:
            from tradingagents.agents.analysts.factor_analyst import FactorAnalyst
            fa = FactorAnalyst(cfg)
            factor_rec = fa.get_factor_tilt_recommendation(current_quadrant=quadrant)
            if factor_rec:
                favored = ', '.join(factor_rec.get('favored_factors', []))
                unfavored = ', '.join(factor_rec.get('unfavored_factors', []))
                reasoning = factor_rec.get('reasoning', '')
                factor_guidance = f'\n\nFactor Tilt Recommendation:\n- Favored: {favored}\n- Unfavored: {unfavored}\n- Reasoning: {reasoning}'
        except Exception as e:
            logger.debug(f'FactorAnalyst not available: {e}')
        quad_name = regime_result.get('quadrant_name', quadrant)
        guidance = f'ENHANCED Macro Guidance (RegimeDetector, confidence={confidence:.0%}):\nRegime: {quad_name}\nGrowth signal: {growth_signal} | Inflation signal: {inflation_signal}\nRisk level: {risk_level.upper()}\n'
        if warnings_list:
            guidance += '\nWarnings:\n'
            for w in warnings_list[:5]:
                if isinstance(w, dict):
                    guidance += f"  - {w.get('message', str(w))}\n"
                else:
                    guidance += f'  - {w}\n'
        alloc_str = ', '.join([f'{k}: {v:.0%}' for k, v in allocations.items()])
        guidance += f'\nTarget Allocation: {alloc_str}'
        guidance += factor_guidance
        if risk_level == 'high':
            guidance += '\n\n⚠️ Risk level: HIGH. Downstream agents should prioritize capital preservation and tighter risk controls.'
        components_summary = ''
        components = regime_result.get('components', {})
        for comp_name, comp_data in components.items():
            if isinstance(comp_data, dict):
                signal = comp_data.get('signal', comp_data.get('regime', 'N/A'))
                conf = comp_data.get('confidence', 'N/A')
                components_summary += f'  - {comp_name}: signal={signal}, confidence={conf}\n'
        analysis_summary = guidance
        if llm:
            llm_prompt = f"You are a macro strategist using a research-grade regime detection framework.\n\nTrade date: {trade_date}\nTicker under review: {company}\n\nRegime detection results:\n- Quadrant: {quad_name} (confidence: {confidence:.0%})\n- Growth signal: {growth_signal}\n- Inflation signal: {inflation_signal}\n- Risk level: {risk_level}\n\nComponent signals:\n{components_summary}\n\nFactor tilt: {factor_guidance or 'N/A'}\n\nTarget allocation: {json.dumps(allocations, indent=2)}\n\nWarnings: {json.dumps(warnings_list[:5], ensure_ascii=False, default=str)}\n\nWrite a concise research macro outlook (English only, at least {macro_min_words} words, {macro_min_sections}+ sections).\nInclude: Regime Diagnosis, Component Evidence, Factor Implications, Single-Name Impact for {company}, Risk Triggers.\nCitation rule: every factual statement must include [Source: X] in the same sentence."
            try:
                response = invoke_with_retry(llm, llm_prompt, caller_name='Enhanced Macro / Regime Bridge', fallback_content=analysis_summary)
                analysis_summary = response.content if hasattr(response, 'content') else str(response)
                analysis_summary = ensure_long_form_report(llm=llm, report_text=analysis_summary, role_name='Enhanced Macro Analyst', min_words=macro_min_words, min_sections=macro_min_sections, citation_style_hint='[Source: MA1]', max_expansion_rounds=macro_max_rounds)
            except Exception as e:
                print(f'[Enhanced Macro] LLM failed: {e}')
        macro_analysis = {'quadrant': quadrant, 'growth_signal': growth_signal, 'inflation_signal': inflation_signal, 'risk_level': risk_level, 'analysis_summary': analysis_summary}
        print(f'[Enhanced Macro] Regime: {quad_name} (conf={confidence:.0%})')
        print(f'[Enhanced Macro] Risk: {risk_level}, Growth: {growth_signal}')
        print(f"{'=' * 60}\n")
        return {'macro_analysis': macro_analysis, 'target_allocations': json.dumps(allocations), 'macro_guidance': guidance}
    return enhanced_macro_node
