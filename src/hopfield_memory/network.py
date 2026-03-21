"""
Modern continuous Hopfield network (Ramsauer et al., 2021).

Energy:  E(xi) = -lse(beta, X^T xi) + 0.5 ||xi||^2
Update:  xi_new = X @ softmax(beta * X^T @ xi)

This is mathematically equivalent to transformer attention with X as both
keys and values.
"""

import warnings
import numpy as np
from typing import List, Tuple, Optional


class ModernHopfieldNetwork:
    """Modern continuous Hopfield network with adaptive inverse temperature.

    Parameters
    ----------
    dim : int
        Dimensionality of stored patterns.
    beta : float
        Base inverse temperature. Higher = sharper retrieval.
    adaptive_beta : bool
        If True, beta is adjusted per-query based on logit separation.
    """

    def __init__(self, dim: int, beta: float = 8.0, adaptive_beta: bool = True):
        self.dim = dim
        self.base_beta = beta
        self.adaptive_beta = adaptive_beta
        self.patterns: List[np.ndarray] = []
        self._X: Optional[np.ndarray] = None

    @property
    def beta(self) -> float:
        return self.base_beta

    @property
    def num_patterns(self) -> int:
        return len(self.patterns)

    def store(self, pattern: np.ndarray) -> int:
        """Store a pattern vector. Returns its index."""
        assert pattern.shape == (self.dim,), f"Expected ({self.dim},), got {pattern.shape}"
        self.patterns.append(pattern.copy())
        self._X = None

        if self.num_patterns > self.dim * 0.5:
            warnings.warn(
                f"Pattern count ({self.num_patterns}) exceeds 50% of dim ({self.dim}). "
                f"Confidence may degrade. Consider increasing dim or using tiered storage.",
                stacklevel=2,
            )

        return self.num_patterns - 1

    def _pattern_matrix(self) -> np.ndarray:
        if self._X is None:
            self._X = np.column_stack(self.patterns)
        return self._X

    def _compute_beta(self, logits: np.ndarray) -> float:
        if not self.adaptive_beta or len(logits) < 2:
            return self.base_beta
        sorted_l = np.sort(logits)[::-1]
        gap = sorted_l[0] - sorted_l[1]
        return self.base_beta / max(gap, 0.05)

    def retrieve(self, query: np.ndarray, num_steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Run the modern Hopfield update rule.

        Returns (retrieved_pattern, attention_weights).
        """
        if not self.patterns:
            raise ValueError("No patterns stored")

        X = self._pattern_matrix()
        xi = query.copy()

        for _ in range(num_steps):
            raw_logits = X.T @ xi
            beta = self._compute_beta(raw_logits)
            logits = beta * raw_logits
            logits -= np.max(logits)
            weights = np.exp(logits)
            weights /= np.sum(weights)
            xi = X @ weights

        return xi, weights

    def energy(self, state: np.ndarray) -> float:
        """Compute the Hopfield energy at a given state.

        Uses the beta-scaled form: E = -(1/beta) * lse(beta * X^T @ xi) + 0.5 * ||xi||^2
        which matches Ramsauer et al. (2021) Equation 3 up to an additive constant.
        """
        X = self._pattern_matrix()
        logits = self.base_beta * (X.T @ state)
        shift = np.max(logits)
        lse = np.log(np.sum(np.exp(logits - shift))) + shift
        return -(1.0 / self.base_beta) * lse + 0.5 * np.dot(state, state)
