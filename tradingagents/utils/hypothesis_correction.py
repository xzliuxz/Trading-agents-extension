from __future__ import annotations
from typing import Any, Dict, List, Optional

def bonferroni_correction(p_values: List[float], alpha: float=0.05, labels: Optional[List[str]]=None) -> Dict[str, Any]:
    m = len(p_values)
    if m == 0:
        return {'corrected_alpha': alpha, 'rejections': [], 'num_significant': 0, 'significant_labels': [], 'method': 'bonferroni'}
    corrected_alpha = alpha / m
    rejections = [p < corrected_alpha for p in p_values]
    num_sig = sum(rejections)
    sig_labels = []
    if labels:
        sig_labels = [lab for lab, rej in zip(labels, rejections) if rej]
    return {'corrected_alpha': corrected_alpha, 'rejections': rejections, 'num_significant': num_sig, 'total_tests': m, 'significant_labels': sig_labels, 'method': 'bonferroni'}

def benjamini_hochberg_fdr(p_values: List[float], fdr_target: float=0.05, labels: Optional[List[str]]=None) -> Dict[str, Any]:
    m = len(p_values)
    if m == 0:
        return {'threshold': fdr_target, 'rejections': [], 'num_significant': 0, 'fdr_estimate': 0.0, 'significant_labels': [], 'method': 'benjamini_hochberg'}
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejections = [False] * m
    max_k = -1
    for rank_minus_1, (orig_idx, p_val) in enumerate(indexed):
        k = rank_minus_1 + 1
        bh_threshold = k / m * fdr_target
        if p_val <= bh_threshold:
            max_k = rank_minus_1
    if max_k >= 0:
        for rank_idx in range(max_k + 1):
            orig_idx = indexed[rank_idx][0]
            rejections[orig_idx] = True
    num_sig = sum(rejections)
    if num_sig > 0:
        rejected_ps = [p for p, rej in zip(p_values, rejections) if rej]
        fdr_est = sum(rejected_ps) / num_sig * m / max(m, 1)
    else:
        fdr_est = 0.0
    effective_threshold = indexed[max_k][1] if max_k >= 0 else 0.0
    sig_labels = []
    if labels:
        sig_labels = [lab for lab, rej in zip(labels, rejections) if rej]
    return {'threshold': effective_threshold, 'rejections': rejections, 'num_significant': num_sig, 'total_tests': m, 'fdr_estimate': round(fdr_est, 6), 'fdr_target': fdr_target, 'significant_labels': sig_labels, 'method': 'benjamini_hochberg'}

def correct_parameter_sweep(sweep_results: List[Dict[str, Any]], p_value_key: str='p_value', label_key: str='params_label', method: str='benjamini_hochberg', alpha: float=0.05) -> Dict[str, Any]:
    p_values = [r.get(p_value_key, 1.0) for r in sweep_results]
    labels = [r.get(label_key, f'combo_{i}') for i, r in enumerate(sweep_results)]
    if method == 'bonferroni':
        result = bonferroni_correction(p_values, alpha=alpha, labels=labels)
    else:
        result = benjamini_hochberg_fdr(p_values, fdr_target=alpha, labels=labels)
    surviving = []
    for i, (r, rej) in enumerate(zip(sweep_results, result['rejections'])):
        if rej:
            surviving.append({**r, '_correction_survived': True, '_rank': len(surviving) + 1})
    result['surviving_params'] = surviving
    result['survival_rate'] = len(surviving) / max(len(sweep_results), 1)
    return result
