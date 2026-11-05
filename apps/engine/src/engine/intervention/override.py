"""
Intervention — override.py

When the circuit breaker fires (OPEN state), this module handles the
forced-summarization injection strategy.

The override works through graph state flags rather than direct prompt injection:
  1. `breaker_check_node` sets `circuit_breaker_triggered = True` in graph state
  2. The LangGraph conditional edge `breaker_routing` routes to `writer_node`
  3. `writer_node` detects `circuit_breaker_triggered=True` and uses
     `_FORCED_SUMMARY_SYSTEM_PROMPT` instead of the normal writing prompt

This design keeps the override decoupled from the agent prompts — the agents
themselves remain unchanged; only the routing and writer-mode flag are affected.
"""

from __future__ import annotations

import logging

from engine.agentic_graph.state import AgentState

logger = logging.getLogger(__name__)


def build_override_state_patch(
    run_id: str,
    source_node: str = "researcher",
) -> dict:
    """
    Build the state patch that signals forced-summarization mode to the Writer.

    Sets `circuit_breaker_triggered=True` and `circuit_breaker_source` so the
    Writer node produces a graceful partial answer instead of a full response.

    Args:
        run_id: The current run's UUID (for logging).
        source_node: Which node's repeated output triggered the breaker.

    Returns:
        A partial AgentState dict to merge into the graph state.
    """
    logger.warning(
        f"[Override] Forced-summarization override applied | "
        f"run={run_id[:8]} | source={source_node}"
    )
    return {
        "circuit_breaker_triggered": True,
        "circuit_breaker_source": source_node,
    }


def escalation_summary(state: AgentState) -> str:
    """
    Build a short human-readable summary of why the breaker fired.
    Used for logging and for the human-in-the-loop escalation stub.

    Args:
        state: The current AgentState at the time of intervention.

    Returns:
        A short plain-text explanation string.
    """
    iterations = state.get("iteration_count", 0)
    source = state.get("circuit_breaker_source", "unknown")
    query_snippet = state.get("query", "")[:80]

    return (
        f"Circuit breaker fired after {iterations} iterations. "
        f"Faulty node: '{source}'. "
        f"Query: '{query_snippet}...'. "
        f"Forced summarization applied — partial answer returned."
    )


__all__ = ["build_override_state_patch", "escalation_summary"]
