"""
Repulsive Modern Hopfield Network (Contrastive Attention).

Extends the standard MHN with negative/repulsive patterns that create
"hills" in the energy landscape, pushing the network away from spurious
states and sharpening decision boundaries between stored memories.

Energy:
    E(xi) = -(1/beta+) * lse(beta+ * X^T xi)
            +(1/beta-) * lse(beta- * Y^T xi)
            + 0.5 * ||xi||^2

Update:
    xi_new = X @ softmax(beta+ * X^T xi) - Y @ softmax(beta- * Y^T xi)
    xi_new = clamp(xi_new, max_norm=R)

Where X = positive patterns (attractors), Y = negative patterns (repellers).
"""

import numpy as np
from typing import List, Tuple, Optional, NamedTuple

from hopfield_memory.network import ModernHopfieldNetwork


class EnergyComponents(NamedTuple):
    positive: float
    negative: float
    quadratic: float
    total: float


class RepulsiveMHN(ModernHopfieldNetwork):
    """Modern Hopfield Network with repulsive (negative) attention.

    Subclass of ModernHopfieldNetwork. Stores both positive patterns
    (attractors) and negative patterns (repellers). The update rule
    pulls toward positives and pushes away from negatives simultaneously.

    Parameters
    ----------
    dim : int
        Dimensionality of stored patterns.
    beta : float
        Inverse temperature for positive (attractive) attention.
    beta_neg : float
        Inverse temperature for negative (repulsive) attention.
        Higher = repel more sharply from the nearest negative.
    adaptive_beta : bool
        Whether to use adaptive beta for the positive term.
    clamp_radius : float
        Maximum norm for the state after each update step.
        Prevents the repulsive term from launching the state to infinity.
    """

    def __init__(
        self,
        dim: int,
        beta: float = 8.0,
        beta_neg: float = 4.0,
        adaptive_beta: bool = False,
        clamp_radius: float = 1.0,
    ):
        super().__init__(dim=dim, beta=beta, adaptive_beta=adaptive_beta)
        self.beta_neg = beta_neg
        self.clamp_radius = clamp_radius
        self.negative_patterns: List[np.ndarray] = []
        self._Y: Optional[np.ndarray] = None

    @property
    def num_negative_patterns(self) -> int:
        return len(self.negative_patterns)

    def store_negative(self, pattern: np.ndarray) -> int:
        """Store a negative (repulsive) pattern. Returns its index."""
        assert pattern.shape == (self.dim,), f"Expected ({self.dim},), got {pattern.shape}"
        self.negative_patterns.append(pattern.copy())
        self._Y = None
        return len(self.negative_patterns) - 1

    def _negative_matrix(self) -> np.ndarray:
        if self._Y is None and self.negative_patterns:
            self._Y = np.column_stack(self.negative_patterns)
        return self._Y

    def _clamp(self, state: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(state)
        if norm > self.clamp_radius and norm > 1e-12:
            state = state * (self.clamp_radius / norm)
        return state

    def retrieve(self, query: np.ndarray, num_steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Dual-attention update: attract to positives, repel from negatives.

        Returns (retrieved_pattern, positive_attention_weights).
        """
        if not self.patterns:
            raise ValueError("No positive patterns stored")

        X = self._pattern_matrix()
        Y = self._negative_matrix()
        xi = query.copy()

        for _ in range(num_steps):
            raw_pos = X.T @ xi
            beta_pos = self._compute_beta(raw_pos)
            pos_logits = beta_pos * raw_pos
            pos_logits -= np.max(pos_logits)
            pos_weights = np.exp(pos_logits)
            pos_weights /= np.sum(pos_weights)

            pull = X @ pos_weights

            if Y is not None:
                neg_logits = self.beta_neg * (Y.T @ xi)
                neg_logits -= np.max(neg_logits)
                neg_weights = np.exp(neg_logits)
                neg_weights /= np.sum(neg_weights)
                push = Y @ neg_weights
                xi = pull - push
            else:
                xi = pull

            xi = self._clamp(xi)

        return xi, pos_weights

    def energy(self, state: np.ndarray) -> float:
        """Compute the contrastive Hopfield energy."""
        return self.energy_components(state).total

    def energy_components(self, state: np.ndarray) -> EnergyComponents:
        """Return individual energy terms for logging.

        Uses the state-dependent beta for the positive term when
        adaptive_beta is enabled, consistent with retrieve().
        """
        X = self._pattern_matrix()
        raw_pos = X.T @ state
        beta_pos = self._compute_beta(raw_pos)
        pos_logits = beta_pos * raw_pos
        pos_shift = np.max(pos_logits)
        pos_lse = np.log(np.sum(np.exp(pos_logits - pos_shift))) + pos_shift
        e_pos = -(1.0 / beta_pos) * pos_lse

        e_neg = 0.0
        Y = self._negative_matrix()
        if Y is not None:
            neg_logits = self.beta_neg * (Y.T @ state)
            neg_shift = np.max(neg_logits)
            neg_lse = np.log(np.sum(np.exp(neg_logits - neg_shift))) + neg_shift
            e_neg = (1.0 / self.beta_neg) * neg_lse

        e_quad = 0.5 * np.dot(state, state)

        return EnergyComponents(
            positive=e_pos,
            negative=e_neg,
            quadratic=e_quad,
            total=e_pos + e_neg + e_quad,
        )
