"""
EMA-smoothed Circuit Breaker State Machine — for Optimization 3.

Applies Exponential Moving Average (EMA) smoothing to the raw cosine
similarity before feeding it into the CLOSED/HALF_OPEN/OPEN FSM.

Smoothing formula:
    S̃_N = α × S_N + (1 − α) × S̃_{N-1}

where α ∈ (0, 1] is the smoothing factor:
  - α = 1.0 → no smoothing (identical to base CircuitBreakerStateMachine)
  - α = 0.7  → light smoothing
  - α = 0.5  → balanced smoothing
  - α = 0.3  → heavy smoothing

Benefit: Attenuates single-step similarity spikes, reducing false positives.
Trade-off: Slightly higher mean trigger iteration for genuine thrashing queries.
"""

from __future__ import annotations

import logging

from engine.heuristic_engine.state_machine import CircuitBreakerStateMachine
from agentic_circuit_breaker.types import BreakerState

logger = logging.getLogger(__name__)


class EMACircuitBreakerStateMachine(CircuitBreakerStateMachine):
    """
    FSM with Exponential Moving Average (EMA) smoothing on the similarity signal.

    Instead of passing raw cosine similarity to the FSM, it first applies:
        S̃_N = α × S_N + (1 − α) × S̃_{N-1}

    The smoothed value S̃_N is what drives state transitions.

    Args:
        alpha: EMA smoothing factor in (0, 1].
               1.0 = no smoothing (raw signal), 0.3 = heavy smoothing.
        half_open_threshold: FSM HALF_OPEN threshold (default from settings).
        open_threshold: FSM OPEN threshold (default from settings).
    """

    def __init__(
        self,
        alpha: float = 0.7,
        half_open_threshold: float | None = None,
        open_threshold: float | None = None,
    ) -> None:
        super().__init__(
            half_open_threshold=half_open_threshold,
            open_threshold=open_threshold,
        )
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._ema: float | None = None  # initialized on first update()

    @property
    def alpha(self) -> float:
        """EMA smoothing factor."""
        return self._alpha

    @property
    def ema_value(self) -> float | None:
        """Current EMA-smoothed similarity value (None before first update)."""
        return self._ema

    def update(self, similarity: float) -> BreakerState:
        """
        Apply EMA smoothing to similarity, then run FSM transition.

        Args:
            similarity: Raw cosine similarity from get_recent_similarity().

        Returns:
            New BreakerState after processing the smoothed value.
        """
        # Apply EMA smoothing
        if self._ema is None:
            self._ema = similarity  # Initialize on first observation
        else:
            self._ema = self._alpha * similarity + (1.0 - self._alpha) * self._ema

        logger.debug(
            f"[EMA FSM] raw_sim={similarity:.4f} | "
            f"ema={self._ema:.4f} | alpha={self._alpha}"
        )

        # Feed smoothed value into the parent FSM
        return super().update(self._ema)

    def reset(self) -> None:
        """Reset FSM state AND EMA accumulator."""
        super().reset()
        self._ema = None

    def __repr__(self) -> str:
        return (
            f"EMACircuitBreakerStateMachine("
            f"alpha={self._alpha}, "
            f"ema={self._ema:.4f if self._ema is not None else None}, "
            f"state={self.current_state.value}, "
            f"steps={len(self.history)})"
        )


__all__ = ["EMACircuitBreakerStateMachine"]
