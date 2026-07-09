from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

class LedoitWolfEstimator:

    @staticmethod
    def estimate(returns: List[List[float]], *, shrinkage_intensity: Optional[float]=None) -> Tuple[List[List[float]], float]:
        t = len(returns)
        if t < 2:
            raise ValueError('至少需要 2 个观察期')
        n = len(returns[0])
        if n < 1:
            raise ValueError('至少需要 1 个资产')
        means = [sum((returns[i][j] for i in range(t))) / t for j in range(n)]
        x = [[returns[i][j] - means[j] for j in range(n)] for i in range(t)]
        sample_cov = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                val = sum((x[k][i] * x[k][j] for k in range(t))) / (t - 1)
                sample_cov[i][j] = val
                sample_cov[j][i] = val
        variances = [sample_cov[i][i] for i in range(n)]
        stds = [math.sqrt(max(v, 1e-15)) for v in variances]
        corr_sum = 0.0
        corr_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if stds[i] > 1e-15 and stds[j] > 1e-15:
                    corr_sum += sample_cov[i][j] / (stds[i] * stds[j])
                    corr_count += 1
        r_bar = corr_sum / corr_count if corr_count > 0 else 0.0
        target = [[0.0] * n for _ in range(n)]
        for i in range(n):
            target[i][i] = variances[i]
            for j in range(i + 1, n):
                val = r_bar * stds[i] * stds[j]
                target[i][j] = val
                target[j][i] = val
        if shrinkage_intensity is not None:
            delta = max(0.0, min(1.0, shrinkage_intensity))
        else:
            delta = LedoitWolfEstimator._optimal_shrinkage(x, sample_cov, target, t, n)
        result = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                result[i][j] = delta * target[i][j] + (1.0 - delta) * sample_cov[i][j]
        return (result, delta)

    @staticmethod
    def _optimal_shrinkage(x: List[List[float]], sample_cov: List[List[float]], target: List[List[float]], t: int, n: int) -> float:
        pi_sum = 0.0
        for i in range(n):
            for j in range(n):
                pi_ij = sum(((x[k][i] * x[k][j] - sample_cov[i][j]) ** 2 for k in range(t))) / t
                pi_sum += pi_ij
        gamma_sum = sum(((sample_cov[i][j] - target[i][j]) ** 2 for i in range(n) for j in range(n)))
        if gamma_sum < 1e-15:
            return 1.0
        kappa = pi_sum / t / gamma_sum
        delta = max(0.0, min(1.0, kappa / t))
        return delta

class ExponentialWeightedEstimator:

    @staticmethod
    def estimate(returns: List[List[float]], half_life: int=63) -> List[List[float]]:
        t = len(returns)
        n = len(returns[0]) if t > 0 else 0
        if t < 2 or n < 1:
            raise ValueError('数据不足')
        decay = math.log(2) / max(half_life, 1)
        weights = [math.exp(-decay * (t - 1 - i)) for i in range(t)]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]
        means = [sum((weights[i] * returns[i][j] for i in range(t))) for j in range(n)]
        cov = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                val = sum((weights[k] * (returns[k][i] - means[i]) * (returns[k][j] - means[j]) for k in range(t)))
                val *= t / max(t - 1, 1)
                cov[i][j] = val
                cov[j][i] = val
        return cov

def condition_number(matrix: List[List[float]]) -> float:
    diag = [matrix[i][i] for i in range(len(matrix))]
    if not diag:
        return float('inf')
    max_d = max(diag)
    min_d = min((d for d in diag if d > 1e-15))
    return max_d / min_d if min_d > 1e-15 else float('inf')

def matrix_to_dict(matrix: List[List[float]], labels: List[str]) -> Dict[str, Dict[str, float]]:
    return {labels[i]: {labels[j]: matrix[i][j] for j in range(len(labels))} for i in range(len(labels))}
