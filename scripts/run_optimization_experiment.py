"""
Optimization Experiment Script — scripts/run_optimization_experiment.py

Runs the circuit breaker pipeline under all 4 optimization technique
configurations against the 20-query benchmark (10 thrashing + 10 normal),
computes precision/recall/F1 for each configuration, and outputs a
comprehensive comparison report.

Optimizations tested:
  OPT1: Adaptive Threshold Tuning   — 4 threshold pairs
  OPT2: Sliding Window Variation    — 3 window sizes
  OPT3: EMA Smoothing               — 4 alpha values
  OPT4: Combined Similarity+Token   — 4 weight pairs

Usage:
    cd apps/engine
    uv run python ../../scripts/run_optimization_experiment.py
    uv run python ../../scripts/run_optimization_experiment.py --queries ../../data/adversarial_queries/all_queries.jsonl
    uv run python ../../scripts/run_optimization_experiment.py --output ../../data/optimization_results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── path setup ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "apps" / "engine" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_optimization_experiment")

# ── Optimization configurations ────────────────────────────────────────────────

OPT1_THRESHOLD_CONFIGS = [
    {"name": "OPT1-A (aggressive)",    "half_open": 0.75, "open": 0.88},
    {"name": "OPT1-B (baseline)",      "half_open": 0.85, "open": 0.95},
    {"name": "OPT1-C (conservative)",  "half_open": 0.88, "open": 0.96},
    {"name": "OPT1-D (balanced)",      "half_open": 0.80, "open": 0.92},
]

OPT2_WINDOW_CONFIGS = [
    {"name": "OPT2-A (window=1, baseline)", "window": 1},
    {"name": "OPT2-C (window=2)",           "window": 2},
    {"name": "OPT2-D (window=3)",           "window": 3},
]

OPT3_EMA_CONFIGS = [
    {"name": "OPT3-A (no smoothing, baseline)", "alpha": 1.0},
    {"name": "OPT3-B (alpha=0.7, light)",       "alpha": 0.7},
    {"name": "OPT3-C (alpha=0.5, balanced)",    "alpha": 0.5},
    {"name": "OPT3-D (alpha=0.3, heavy)",       "alpha": 0.3},
]

OPT4_WEIGHT_CONFIGS = [
    {"name": "OPT4-A (sim-only, baseline)",  "w_sim": 1.0, "w_token": 0.0},
    {"name": "OPT4-B (0.8/0.2)",             "w_sim": 0.8, "w_token": 0.2},
    {"name": "OPT4-C (0.6/0.4)",             "w_sim": 0.6, "w_token": 0.4},
    {"name": "OPT4-D (0.5/0.5)",             "w_sim": 0.5, "w_token": 0.5},
]

TOKEN_BUDGET = 10_000  # estimated token budget per run for OPT4


# ── Core experiment runner ─────────────────────────────────────────────────────

async def run_single_optimized(
    query: str,
    label: str,
    opt_type: str,
    opt_config: dict[str, Any],
    corpus_docs: list[str],
    recursion_limit: int = 15,
) -> dict[str, Any]:
    """
    Run one query through the circuit-breaker pipeline under a specific
    optimization configuration.

    Returns a result dict with detection outcome, token usage, and timing.
    """
    from engine.agentic_graph.graph import build_graph
    from engine.agentic_graph.state import AgentState
    from engine.interceptor.hooks import InterceptorContext, flush_pending_logs
    from engine.interceptor.log_store import LogStore
    from engine.storage.db import get_session_factory
    from langgraph.errors import GraphRecursionError

    # Build the custom state machine based on opt_type
    sm = _build_state_machine(opt_type, opt_config)

    run_id = str(uuid.uuid4())
    factory = get_session_factory()

    initial_state: AgentState = {
        "query": query,
        "corpus_docs": corpus_docs,
        "research_notes": "",
        "critique": "",
        "draft": "",
        "final_output": "",
        "iteration_count": 0,
        "circuit_breaker_triggered": False,
        "circuit_breaker_source": None,
        "messages": [],
    }

    start = time.perf_counter()
    async with factory() as db:
        log_store = LogStore(db=db, run_id=run_id)
        await log_store.create_run(query=query)

        ctx = InterceptorContext(run_id=run_id, state_machine=sm)

        # OPT2: pass window to breaker_check via ctx
        if opt_type == "OPT2":
            ctx._window = opt_config["window"]  # stored for use in custom breaker

        graph = build_graph(interceptor_ctx=ctx, use_circuit_breaker=True)

        try:
            final_state = graph.invoke(
                initial_state,
                config={"recursion_limit": recursion_limit},
            )
            await flush_pending_logs(ctx, db)
            iterations = final_state.get("iteration_count", 0)
            breaker_triggered = final_state.get("circuit_breaker_triggered", False)
            breaker_source = final_state.get("circuit_breaker_source")
            status = "breaker_triggered" if breaker_triggered else "completed"
            await log_store.finalize_run(
                status=status, iterations=iterations,
                circuit_breaker_triggered=breaker_triggered,
                final_output=final_state.get("final_output", ""),
            )
            await db.commit()

        except GraphRecursionError:
            await flush_pending_logs(ctx, db)
            iterations = recursion_limit
            breaker_triggered = False
            breaker_source = None
            status = "crashed"
            await log_store.finalize_run(
                status="crashed", iterations=iterations,
                error="GraphRecursionError",
            )
            await db.commit()

    duration_ms = (time.perf_counter() - start) * 1000.0
    tokens = ctx.token_accumulator.total_tokens

    # Determine detection outcome
    tp = label == "thrashing" and breaker_triggered
    fp = label == "normal" and breaker_triggered
    tn = label == "normal" and not breaker_triggered
    fn = label == "thrashing" and not breaker_triggered

    return {
        "query": query[:80],
        "label": label,
        "status": status,
        "breaker_triggered": breaker_triggered,
        "breaker_source": breaker_source,
        "iterations": iterations,
        "tokens_used": tokens,
        "duration_ms": duration_ms,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "opt_type": opt_type,
        "opt_config": opt_config["name"],
    }


def _build_state_machine(opt_type: str, opt_config: dict[str, Any]):
    """Build the appropriate state machine for the given optimization type."""
    if opt_type == "OPT1":
        from engine.heuristic_engine.state_machine import CircuitBreakerStateMachine
        return CircuitBreakerStateMachine(
            half_open_threshold=opt_config["half_open"],
            open_threshold=opt_config["open"],
        )
    elif opt_type == "OPT2":
        # Window is handled in the breaker_check node via monkey-patching;
        # the state machine itself stays at default thresholds
        from engine.heuristic_engine.state_machine import CircuitBreakerStateMachine
        return CircuitBreakerStateMachine()
    elif opt_type == "OPT3":
        from engine.heuristic_engine.ema_state_machine import EMACircuitBreakerStateMachine
        return EMACircuitBreakerStateMachine(alpha=opt_config["alpha"])
    elif opt_type == "OPT4":
        # OPT4: composite score passed to standard FSM (via custom breaker check)
        from engine.heuristic_engine.state_machine import CircuitBreakerStateMachine
        return CircuitBreakerStateMachine()
    else:
        from engine.heuristic_engine.state_machine import CircuitBreakerStateMachine
        return CircuitBreakerStateMachine()


async def run_window_query(
    query: str,
    label: str,
    window: int,
    corpus_docs: list[str],
    recursion_limit: int = 15,
) -> dict[str, Any]:
    """
    Specialized runner for OPT2 (window variation).
    Uses a custom breaker check that respects the window parameter.
    """
    from engine.heuristic_engine.similarity import get_recent_similarity
    from engine.heuristic_engine.state_machine import CircuitBreakerStateMachine
    from engine.heuristic_engine.verifier import HalfOpenVerifier
    from engine.interceptor.hooks import InterceptorContext, PendingLog, flush_pending_logs
    from engine.interceptor.log_store import LogStore
    from engine.agentic_graph.state import AgentState
    from engine.agentic_graph.nodes.researcher import researcher_node
    from engine.agentic_graph.nodes.critic import critic_node, should_continue
    from engine.agentic_graph.nodes.writer import writer_node
    from engine.interceptor.hooks import wrap_node
    from engine.storage.db import get_session_factory
    from langgraph.graph import END, START, StateGraph
    from langgraph.errors import GraphRecursionError
    from agentic_circuit_breaker.types import BreakerState

    sm = CircuitBreakerStateMachine()
    verifier = HalfOpenVerifier()
    run_id = str(uuid.uuid4())
    factory = get_session_factory()

    initial_state: AgentState = {
        "query": query,
        "corpus_docs": corpus_docs,
        "research_notes": "",
        "critique": "",
        "draft": "",
        "final_output": "",
        "iteration_count": 0,
        "circuit_breaker_triggered": False,
        "circuit_breaker_source": None,
        "messages": [],
    }

    ctx = InterceptorContext(run_id=run_id, state_machine=sm)

    # Build a custom breaker node with the specified window
    def _custom_breaker(state: dict) -> dict:
        iteration = state.get("iteration_count", 0)
        researcher_embeddings = [
            log.embedding
            for log in ctx.pending_logs
            if log.node == "researcher" and log.embedding is not None
        ]
        if len(researcher_embeddings) < window + 1:
            return {"circuit_breaker_triggered": False, "iteration_count": iteration}

        similarity = get_recent_similarity(researcher_embeddings, window=window)
        if similarity is None:
            return {"circuit_breaker_triggered": False, "iteration_count": iteration}

        new_state = ctx.state_machine.update(similarity)
        if new_state == BreakerState.HALF_OPEN:
            verifier.verify(
                similarity=similarity,
                iteration=iteration,
                total_tokens_used=ctx.token_accumulator.total_tokens,
            )
        if new_state == BreakerState.OPEN:
            if ctx.pending_logs:
                ctx.pending_logs[-1].breaker_state = BreakerState.OPEN.value
            return {
                "circuit_breaker_triggered": True,
                "circuit_breaker_source": "researcher",
                "iteration_count": iteration,
            }
        if ctx.pending_logs:
            ctx.pending_logs[-1].breaker_state = new_state.value
        return {"circuit_breaker_triggered": False, "circuit_breaker_source": None, "iteration_count": iteration}

    def _breaker_routing(state: dict) -> str:
        return "writer" if state.get("circuit_breaker_triggered", False) else "researcher"

    _researcher = wrap_node(researcher_node, "researcher", ctx)
    _critic = wrap_node(critic_node, "critic", ctx)
    _writer = wrap_node(writer_node, "writer", ctx)

    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", _researcher)
    workflow.add_node("critic", _critic)
    workflow.add_node("breaker_check", _custom_breaker)
    workflow.add_node("writer", _writer)
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_conditional_edges("critic", should_continue, {"breaker_check": "breaker_check", "writer": "writer"})
    workflow.add_conditional_edges("breaker_check", _breaker_routing, {"researcher": "researcher", "writer": "writer"})
    workflow.add_edge("writer", END)
    graph = workflow.compile()

    start = time.perf_counter()
    async with factory() as db:
        log_store = LogStore(db=db, run_id=run_id)
        await log_store.create_run(query=query)
        try:
            final_state = graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
            await flush_pending_logs(ctx, db)
            iterations = final_state.get("iteration_count", 0)
            breaker_triggered = final_state.get("circuit_breaker_triggered", False)
            status = "breaker_triggered" if breaker_triggered else "completed"
            await log_store.finalize_run(status=status, iterations=iterations, circuit_breaker_triggered=breaker_triggered)
            await db.commit()
        except GraphRecursionError:
            await flush_pending_logs(ctx, db)
            iterations = recursion_limit
            breaker_triggered = False
            status = "crashed"
            await log_store.finalize_run(status="crashed", iterations=iterations, error="GraphRecursionError")
            await db.commit()

    duration_ms = (time.perf_counter() - start) * 1000.0
    return {
        "query": query[:80],
        "label": label,
        "status": status,
        "breaker_triggered": breaker_triggered,
        "breaker_source": final_state.get("circuit_breaker_source") if status != "crashed" else None,
        "iterations": iterations,
        "tokens_used": ctx.token_accumulator.total_tokens,
        "duration_ms": duration_ms,
        "tp": label == "thrashing" and breaker_triggered,
        "fp": label == "normal" and breaker_triggered,
        "tn": label == "normal" and not breaker_triggered,
        "fn": label == "thrashing" and not breaker_triggered,
        "opt_type": "OPT2",
        "opt_config": f"window={window}",
    }


async def run_opt4_query(
    query: str,
    label: str,
    w_sim: float,
    w_token: float,
    config_name: str,
    corpus_docs: list[str],
    recursion_limit: int = 15,
) -> dict[str, Any]:
    """
    Specialized runner for OPT4 (combined similarity + token signal).
    Uses a composite score = w_sim * similarity + w_token * token_progress.
    """
    from engine.heuristic_engine.similarity import get_recent_similarity
    from engine.heuristic_engine.state_machine import CircuitBreakerStateMachine
    from engine.heuristic_engine.verifier import HalfOpenVerifier
    from engine.interceptor.hooks import InterceptorContext, flush_pending_logs
    from engine.interceptor.log_store import LogStore
    from engine.agentic_graph.state import AgentState
    from engine.agentic_graph.nodes.researcher import researcher_node
    from engine.agentic_graph.nodes.critic import critic_node, should_continue
    from engine.agentic_graph.nodes.writer import writer_node
    from engine.interceptor.hooks import wrap_node
    from engine.storage.db import get_session_factory
    from langgraph.graph import END, START, StateGraph
    from langgraph.errors import GraphRecursionError
    from agentic_circuit_breaker.types import BreakerState

    sm = CircuitBreakerStateMachine()
    verifier = HalfOpenVerifier()
    run_id = str(uuid.uuid4())
    factory = get_session_factory()

    initial_state: AgentState = {
        "query": query, "corpus_docs": corpus_docs,
        "research_notes": "", "critique": "", "draft": "", "final_output": "",
        "iteration_count": 0, "circuit_breaker_triggered": False,
        "circuit_breaker_source": None, "messages": [],
    }

    ctx = InterceptorContext(run_id=run_id, state_machine=sm)

    def _composite_breaker(state: dict) -> dict:
        iteration = state.get("iteration_count", 0)
        researcher_embeddings = [
            log.embedding for log in ctx.pending_logs
            if log.node == "researcher" and log.embedding is not None
        ]
        if len(researcher_embeddings) < 2:
            return {"circuit_breaker_triggered": False, "iteration_count": iteration}

        raw_sim = get_recent_similarity(researcher_embeddings, window=1)
        if raw_sim is None:
            return {"circuit_breaker_triggered": False, "iteration_count": iteration}

        # Composite score: w_sim * similarity + w_token * token_progress
        token_progress = min(ctx.token_accumulator.total_tokens / TOKEN_BUDGET, 1.0)
        score = w_sim * raw_sim + w_token * token_progress

        new_state = ctx.state_machine.update(score)
        if new_state == BreakerState.HALF_OPEN:
            verifier.verify(similarity=score, iteration=iteration, total_tokens_used=ctx.token_accumulator.total_tokens)
        if new_state == BreakerState.OPEN:
            if ctx.pending_logs:
                ctx.pending_logs[-1].breaker_state = BreakerState.OPEN.value
            return {"circuit_breaker_triggered": True, "circuit_breaker_source": "researcher", "iteration_count": iteration}
        if ctx.pending_logs:
            ctx.pending_logs[-1].breaker_state = new_state.value
        return {"circuit_breaker_triggered": False, "circuit_breaker_source": None, "iteration_count": iteration}

    def _routing(state: dict) -> str:
        return "writer" if state.get("circuit_breaker_triggered", False) else "researcher"

    _researcher = wrap_node(researcher_node, "researcher", ctx)
    _critic = wrap_node(critic_node, "critic", ctx)
    _writer = wrap_node(writer_node, "writer", ctx)

    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", _researcher)
    workflow.add_node("critic", _critic)
    workflow.add_node("breaker_check", _composite_breaker)
    workflow.add_node("writer", _writer)
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_conditional_edges("critic", should_continue, {"breaker_check": "breaker_check", "writer": "writer"})
    workflow.add_conditional_edges("breaker_check", _routing, {"researcher": "researcher", "writer": "writer"})
    workflow.add_edge("writer", END)
    graph = workflow.compile()

    start = time.perf_counter()
    async with factory() as db:
        log_store = LogStore(db=db, run_id=run_id)
        await log_store.create_run(query=query)
        try:
            final_state = graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
            await flush_pending_logs(ctx, db)
            iterations = final_state.get("iteration_count", 0)
            breaker_triggered = final_state.get("circuit_breaker_triggered", False)
            status = "breaker_triggered" if breaker_triggered else "completed"
            await log_store.finalize_run(status=status, iterations=iterations, circuit_breaker_triggered=breaker_triggered)
            await db.commit()
        except GraphRecursionError:
            await flush_pending_logs(ctx, db)
            iterations = recursion_limit
            breaker_triggered = False
            status = "crashed"
            await log_store.finalize_run(status="crashed", iterations=iterations, error="GraphRecursionError")
            await db.commit()

    duration_ms = (time.perf_counter() - start) * 1000.0
    return {
        "query": query[:80],
        "label": label,
        "status": status,
        "breaker_triggered": breaker_triggered,
        "breaker_source": final_state.get("circuit_breaker_source") if status != "crashed" else None,
        "iterations": iterations,
        "tokens_used": ctx.token_accumulator.total_tokens,
        "duration_ms": duration_ms,
        "tp": label == "thrashing" and breaker_triggered,
        "fp": label == "normal" and breaker_triggered,
        "tn": label == "normal" and not breaker_triggered,
        "fn": label == "thrashing" and not breaker_triggered,
        "opt_type": "OPT4",
        "opt_config": config_name,
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute precision, recall, F1, and token savings from results."""
    tp = sum(1 for r in results if r["tp"])
    fp = sum(1 for r in results if r["fp"])
    tn = sum(1 for r in results if r["tn"])
    fn = sum(1 for r in results if r["fn"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    tokens_list = [r["tokens_used"] for r in results]
    mean_tokens = sum(tokens_list) / len(tokens_list) if tokens_list else 0.0
    mean_iters = sum(r["iterations"] for r in results) / len(results) if results else 0.0

    trigger_iters = [r["iterations"] for r in results if r["tp"]]
    mean_trigger_iter = sum(trigger_iters) / len(trigger_iters) if trigger_iters else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr,
        "mean_tokens": mean_tokens,
        "mean_iterations": mean_iters,
        "mean_trigger_iteration": mean_trigger_iter,
        "n": len(results),
    }


async def main_async(args: argparse.Namespace) -> None:
    from engine.storage.db import init_db

    await init_db()

    query_file = Path(args.queries)
    if not query_file.exists():
        logger.error(f"Query file not found: {query_file}")
        sys.exit(1)

    queries: list[dict[str, Any]] = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    logger.info(f"Loaded {len(queries)} queries from {query_file}")

    # Load corpus docs
    corpus_dir = _REPO / "data" / "corpus"
    corpus_docs: list[str] = []
    if corpus_dir.exists():
        corpus_docs = [f.read_text(encoding="utf-8").strip() for f in sorted(corpus_dir.glob("*.md"))]
    logger.info(f"Loaded {len(corpus_docs)} corpus documents")

    all_config_results: dict[str, list[dict[str, Any]]] = {}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── OPT1: Adaptive Threshold ─────────────────────────────────────────────
    for cfg in OPT1_THRESHOLD_CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"OPT1 — {cfg['name']} | HO={cfg['half_open']}, Open={cfg['open']}")
        logger.info(f"{'='*60}")
        results = []
        for i, item in enumerate(queries, 1):
            q, lbl = item["query"], item.get("label", "unknown")
            logger.info(f"  [{i}/{len(queries)}] {lbl}: {q[:60]}...")
            r = await run_single_optimized(q, lbl, "OPT1", cfg, corpus_docs)
            results.append(r)
            outcome = "TP" if r["tp"] else "FP" if r["fp"] else "TN" if r["tn"] else "FN"
            logger.info(f"    → status={r['status']} | triggered={r['breaker_triggered']} | iters={r['iterations']} | {outcome}")
        all_config_results[cfg["name"]] = results

    # ── OPT2: Sliding Window ─────────────────────────────────────────────────
    for cfg in OPT2_WINDOW_CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"OPT2 — {cfg['name']} | window={cfg['window']}")
        logger.info(f"{'='*60}")
        results = []
        for i, item in enumerate(queries, 1):
            q, lbl = item["query"], item.get("label", "unknown")
            logger.info(f"  [{i}/{len(queries)}] {lbl}: {q[:60]}...")
            r = await run_window_query(q, lbl, cfg["window"], corpus_docs)
            results.append(r)
            outcome = "TP" if r["tp"] else "FP" if r["fp"] else "TN" if r["tn"] else "FN"
            logger.info(f"    → status={r['status']} | triggered={r['breaker_triggered']} | iters={r['iterations']} | {outcome}")
        all_config_results[cfg["name"]] = results

    # ── OPT3: EMA Smoothing ──────────────────────────────────────────────────
    for cfg in OPT3_EMA_CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"OPT3 — {cfg['name']} | alpha={cfg['alpha']}")
        logger.info(f"{'='*60}")
        results = []
        for i, item in enumerate(queries, 1):
            q, lbl = item["query"], item.get("label", "unknown")
            logger.info(f"  [{i}/{len(queries)}] {lbl}: {q[:60]}...")
            r = await run_single_optimized(q, lbl, "OPT3", cfg, corpus_docs)
            results.append(r)
            outcome = "TP" if r["tp"] else "FP" if r["fp"] else "TN" if r["tn"] else "FN"
            logger.info(f"    → status={r['status']} | triggered={r['breaker_triggered']} | iters={r['iterations']} | {outcome}")
        all_config_results[cfg["name"]] = results

    # ── OPT4: Combined Similarity + Token ────────────────────────────────────
    for cfg in OPT4_WEIGHT_CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"OPT4 — {cfg['name']} | w_sim={cfg['w_sim']}, w_token={cfg['w_token']}")
        logger.info(f"{'='*60}")
        results = []
        for i, item in enumerate(queries, 1):
            q, lbl = item["query"], item.get("label", "unknown")
            logger.info(f"  [{i}/{len(queries)}] {lbl}: {q[:60]}...")
            r = await run_opt4_query(q, lbl, cfg["w_sim"], cfg["w_token"], cfg["name"], corpus_docs)
            results.append(r)
            outcome = "TP" if r["tp"] else "FP" if r["fp"] else "TN" if r["tn"] else "FN"
            logger.info(f"    → status={r['status']} | triggered={r['breaker_triggered']} | iters={r['iterations']} | {outcome}")
        all_config_results[cfg["name"]] = results

    # ── Compute metrics and write output ─────────────────────────────────────
    out_dir = _REPO / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    opt_path = Path(args.output) if args.output else out_dir / f"optimization_results_{timestamp}.jsonl"
    latest_path = out_dir / "optimization_results_latest.jsonl"

    all_rows = []
    metrics_summary = {}
    for config_name, results in all_config_results.items():
        m = compute_metrics(results)
        metrics_summary[config_name] = m
        for r in results:
            all_rows.append({**r, "config_name": config_name})

    with open(opt_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in all_rows)
    with open(latest_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in all_rows)

    logger.info(f"\nOptimization results saved to {opt_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  OPTIMIZATION EXPERIMENT SUMMARY")
    print("=" * 110)
    print(f"{'Configuration':<38} | {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} | {'Prec':>6} {'Recall':>7} {'F1':>6} {'FPR':>5} | {'Tokens':>8} | {'TrigIter':>9}")
    print("-" * 110)
    for config_name, m in metrics_summary.items():
        print(
            f"{config_name:<38} | {m['tp']:>3} {m['fp']:>3} {m['tn']:>3} {m['fn']:>3} | "
            f"{m['precision']:>6.3f} {m['recall']:>7.3f} {m['f1']:>6.3f} {m['fpr']:>5.3f} | "
            f"{m['mean_tokens']:>8.0f} | {m['mean_trigger_iteration']:>9.1f}"
        )
    print("=" * 110)

    # Save metrics summary
    metrics_path = out_dir / f"optimization_metrics_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Metrics summary saved to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optimization experiments for Agentic Circuit Breaker.")
    parser.add_argument(
        "--queries",
        type=str,
        default=str(_REPO / "data" / "adversarial_queries" / "all_queries.jsonl"),
        help="Path to JSONL queries file (default: all_queries.jsonl with 20 queries).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSONL file.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
