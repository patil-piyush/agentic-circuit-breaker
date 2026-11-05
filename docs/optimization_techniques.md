# Optimization Techniques — Agentic Circuit Breaker

**Project:** Agentic Circuit Breakers: Semantic Loop Detection and Cost Mitigation in Stateful Multi-Agent RAG Architectures
**Course:** Research Methodology, B.Tech (PCCOE, Pune, A.Y. 2026-27)

---

## Overview

The baseline Agentic Circuit Breaker uses a fixed CLOSED→HALF_OPEN→OPEN finite state machine (FSM) driven by raw cosine similarity between consecutive Researcher embeddings (N vs N-1), with hard-coded thresholds of 0.85 (HALF_OPEN) and 0.95 (OPEN).

Four optimization techniques are applied and benchmarked to improve the system's:
- **Detection accuracy** — minimizing false positives (normal queries wrongly interrupted) and false negatives (thrashing queries not caught)
- **Detection latency** — reducing wasted iterations before the breaker fires
- **Token efficiency** — maximizing tokens saved without over-triggering
- **Robustness** — stability of the signal under noisy, stochastic LLM outputs

---

## Optimization 1: Adaptive Threshold Tuning

### Background & Motivation

The default thresholds (0.85 HALF_OPEN, 0.95 OPEN) were chosen heuristically based on the project synopsis. However, the optimal decision boundary depends on the embedding model and the linguistic style of the agent's outputs. Fixed thresholds may cause:
- **Too-low thresholds** → false positives: normal queries with semantically related iterations wrongly trigger the breaker
- **Too-high thresholds** → false negatives: thrashing queries that drift slightly between iterations escape detection until the recursion limit

### Theory

The FSM processes similarity signals as a 1D classification problem. Each query's trajectory through embedding space generates a series of cosine similarity values {s1, s2, ..., sk}. The problem reduces to: *given a similarity value s, which threshold pair (τ_ho, τ_o) best separates thrashing from non-thrashing trajectories?*

For a binary labelled dataset (y ∈ {thrashing, normal}), the optimal threshold pair maximizes the F1-score:

  F1 = 2 × (Precision × Recall) / (Precision + Recall)

where Precision = TP / (TP + FP) and Recall = TP / (TP + FN).

### Threshold Configurations Tested

| Config      | HALF_OPEN (τ_ho) | OPEN (τ_o) | Hypothesis                                              |
|:---:        |:---:             |:---:       |:---                                                     |
| **OPT1-A**  | 0.75             | 0.88       | Aggressive — catches more thrashing early, may false-positive |
| **OPT1-B**  | 0.85             | 0.95       | Default baseline — original heuristic values            |
| **OPT1-C**  | 0.88             | 0.96       | Conservative — fewer false positives, may miss slow-onset thrashing |
| **OPT1-D**  | 0.80             | 0.92       | Balanced — splits the difference                        |

### Implementation

Thresholds are loaded from `engine.config.Settings` (.env variables `SIMILARITY_HALF_OPEN` and `SIMILARITY_OPEN`). The optimization experiment runner overrides these per-run using the `CircuitBreakerStateMachine(half_open_threshold=..., open_threshold=...)` constructor — no engine code changes required.

```python
# In run_optimization_experiment.py:
sm = CircuitBreakerStateMachine(
    half_open_threshold=config["half_open"],
    open_threshold=config["open"]
)
```

### Evaluation Metrics

| Metric                     | Formula                        | What it Measures                             |
|:---                        |:---                            |:---                                          |
| True Positive Rate (Recall) | TP / (TP + FN)                | Fraction of thrashing queries correctly caught |
| False Positive Rate         | FP / (FP + TN)                | Fraction of normal queries wrongly interrupted |
| Precision                   | TP / (TP + FP)                | Quality of positive detections               |
| F1-Score                    | 2×P×R / (P+R)                 | Balanced accuracy                            |
| Avg Token Savings (%)       | mean(saved/baseline) × 100    | Economic efficiency                          |
| Avg Detection Latency (iters) | mean(trigger_iteration)     | Speed of detection                           |

### Expected Outcome

OPT1-A or OPT1-D should outperform the baseline (OPT1-B) on recall. OPT1-C should have the lowest FPR. The best config balances F1-Score and token savings.

---

## Optimization 2: Sliding Window Variation

### Background & Motivation

The baseline implementation compares the **latest** embedding against the **immediately preceding** embedding (window = 1), i.e., Sim(N, N-1). This is sensitive to single-step oscillations but may miss slow-onset thrashing where the agent gradually converges rather than immediately repeating.

The PLAN.md's original specification calls for "N vs N-2" (window = 2), which the library supports via `get_recent_similarity(embeddings, window=2)`. Testing multiple window sizes identifies which comparison horizon gives the best signal.

### Theory

Define the window-w similarity as:

  S_N^(w) = cosine(e_N, e_{N-w})

- **window=1** (N vs N-1): Detects *immediate* repetition within one iteration. Low latency but high noise.
- **window=2** (N vs N-2): Compares against two steps back. Skips one-step oscillations, catches sustained repetition.
- **window=3** (N vs N-3): Detects slow-onset convergence. Higher latency, potentially more accurate.

For thrashing trajectories, S_N^(w) rises toward 1.0 as N grows. The key trade-off is:
- **Smaller w** → faster detection → fewer wasted tokens, but more noise
- **Larger w** → smoother signal → more accurate, but more wasted iterations before trigger

### Window Configurations Tested

| Config      | Window Size | Comparison | Hypothesis                              |
|:---:        |:---:        |:---:       |:---                                     |
| **OPT2-A**  | w=1         | N vs N-1   | Fastest detection; noisier              |
| **OPT2-B** (Baseline) | w=1 | N vs N-1 | Current implementation               |
| **OPT2-C**  | w=2         | N vs N-2   | PLAN.md original spec; smoother signal  |
| **OPT2-D**  | w=3         | N vs N-3   | Slowest; best for gradual convergence   |

> **Note:** Larger windows require more embeddings before first comparison (w=2 needs ≥3, w=3 needs ≥4), meaning the breaker cannot fire before iteration 3 or 4 respectively.

### Implementation

```python
# In breaker_check.py (make_breaker_check_node):
similarity = get_recent_similarity(researcher_embeddings, window=WINDOW_SIZE)
```

The optimization experiment passes `window` as a parameter to the breaker check node factory.

### Evaluation Metrics

Same as Optimization 1, plus:
- **Minimum detectable iteration** — minimum iterations needed before first possible trigger
- **Mean trigger iteration** across thrashing queries

### Expected Outcome

Window=2 or window=3 should have higher precision (fewer false positives) at the cost of slightly higher mean trigger iteration. Window=1 will fire faster but may over-trigger on temporarily similar consecutive outputs.

---

## Optimization 3: Exponential Moving Average (EMA) Smoothing

### Background & Motivation

Raw cosine similarity values are noisy — two consecutive LLM responses may happen to be semantically similar without actual thrashing, or a thrashing trajectory may have one divergent iteration due to stochastic sampling. The raw signal causes the FSM to react to individual spikes, potentially triggering false positives or HALF_OPEN↔CLOSED oscillations.

**EMA smoothing** applies a weighted average that emphasizes recent observations while retaining memory of the past, attenuating noise.

### Theory

The EMA-smoothed similarity at step N is:

  S̃_N = α × S_N + (1 − α) × S̃_{N-1}

with initialization S̃_1 = S_1 and smoothing factor α ∈ (0, 1].

- **α close to 1**: Fast adaptation, little smoothing (approaches raw signal)
- **α close to 0**: Heavy smoothing, slow adaptation (lags the true signal)

The FSM then receives the smoothed value S̃_N instead of the raw S_N:

  FSM.update(S̃_N)

**Effect:** A single high-similarity spike from a non-thrashing step is attenuated. A consistent thrashing pattern (multiple consecutive high-S values) still drives S̃ above the threshold.

### EMA Configurations Tested

| Config              | α    | Description                                                    |
|:---:                |:---: |:---                                                            |
| **OPT3-A** (Baseline) | 1.0 | No smoothing (raw signal — current behavior)                |
| **OPT3-B**          | 0.7  | Light smoothing — fast response, some noise reduction          |
| **OPT3-C**          | 0.5  | Balanced smoothing — equal weight to current and history       |
| **OPT3-D**          | 0.3  | Heavy smoothing — slow response, maximum noise reduction       |

### Implementation

The EMA state is maintained in a subclassed state machine, keeping EMA logic fully encapsulated:

```python
class EMACircuitBreakerStateMachine(CircuitBreakerStateMachine):
    def __init__(self, alpha: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha
        self._ema: float | None = None

    def update(self, similarity: float) -> BreakerState:
        # Apply EMA smoothing before FSM logic
        if self._ema is None:
            self._ema = similarity      # Initialize on first call
        else:
            self._ema = self._alpha * similarity + (1 - self._alpha) * self._ema
        return super().update(self._ema)  # Feed smoothed value to FSM
```

The smoothed value used is stored in `self._ema` and can be retrieved for logging/visualization.

### Evaluation Metrics

Same as Optimization 1. Key additional metrics:
- **Smoothed vs raw similarity trace** — visualize how EMA affects the signal trajectory per run
- **FPR reduction** — primary benefit expected here

### Expected Outcome

EMA with α=0.5 or α=0.7 should reduce false positive rate vs baseline (α=1.0). The trade-off is a slightly higher mean trigger iteration for thrashing queries (because the EMA lags the rising signal). Heavy smoothing (α=0.3) may miss fast-onset thrashing.

---

## Optimization 4: Combined Similarity + Token Budget Signal

### Background & Motivation

The baseline circuit breaker uses **only cosine similarity** as its triggering signal. However, two independent signals are available per iteration:
1. **Semantic signal:** Cosine similarity between Researcher embeddings
2. **Economic signal:** Running token consumption rate (tokens used / estimated budget)

Combining these signals gives a more robust decision. A run consuming tokens rapidly AND producing semantically repetitive outputs is far more likely to be thrashing than one with high similarity alone.

### Theory

Define a **composite thrashing score** at iteration N:

  Score_N = w1 × S_N + w2 × P_N^token

where:
- S_N = cosine similarity (semantic signal, range [0, 1])
- P_N^token = min(tokens_used_N / token_budget, 1.0) = fraction of token budget consumed
- w1, w2 ≥ 0 with w1 + w2 = 1.0

The FSM is then driven by Score_N instead of raw S_N:
- If the agent is spending many tokens AND producing repetitive outputs → Score is high → breaker fires early
- If the agent is spending many tokens but making progress → S_N is low → Score stays below threshold even with high P_N^token

### Weight Configurations Tested

| Config              | w_sim (w1) | w_token (w2) | Description                             |
|:---:                |:---:        |:---:          |:---                                     |
| **OPT4-A** (Baseline) | 1.0       | 0.0           | Similarity-only (current behavior)      |
| **OPT4-B**          | 0.8         | 0.2           | Similarity-dominant with token awareness |
| **OPT4-C**          | 0.6         | 0.4           | Balanced — token signal has meaningful weight |
| **OPT4-D**          | 0.5         | 0.5           | Equal weight — maximum multi-signal influence |

### Implementation

A composite score replaces the raw similarity passed to the FSM. The token count comes from the `InterceptorContext.token_accumulator`:

```python
def _compute_composite_score(
    similarity: float,
    total_tokens: int,
    token_budget: int = 10_000,
    w_sim: float = 0.8,
    w_token: float = 0.2,
) -> float:
    token_progress = min(total_tokens / max(token_budget, 1), 1.0)
    return w_sim * similarity + w_token * token_progress

# In breaker_check_node:
raw_sim = get_recent_similarity(researcher_embeddings, window=1)
score = _compute_composite_score(raw_sim, ctx.token_accumulator.total_tokens, w_sim=W_SIM, w_token=W_TOKEN)
new_state = ctx.state_machine.update(score)  # Feed composite score to FSM
```

### Evaluation Metrics

Same as Optimization 1. Key additional metrics:
- **Score decomposition:** Show contribution of similarity vs token signal per run
- **Early detection:** Does the token signal allow detection before similarity fully rises?

### Expected Outcome

OPT4-B (w_sim=0.8, w_token=0.2) should improve detection for cases where thrashing starts with moderate similarity (0.80-0.90) but high token consumption. The token signal acts as an early warning layer. OPT4-D (equal weights) may over-penalize long-running-but-valid queries.

---

## Comparative Evaluation Framework

All 4 optimizations are evaluated against the **same 20-query benchmark**:
- 10 thrashing queries (label = "thrashing") → expected outcome: `circuit_breaker_triggered = True`
- 10 normal queries (label = "normal") → expected outcome: `circuit_breaker_triggered = False`

### Confusion Matrix Definitions

| Term | Definition                                                     |
|:-----|:---                                                            |
| TP   | Thrashing query, breaker fires → correctly caught              |
| FP   | Normal query, breaker fires → false alarm (bad)                |
| TN   | Normal query, breaker doesn't fire → correct (good)            |
| FN   | Thrashing query, breaker doesn't fire → missed (bad)           |

### Statistical Tests Applied

1. **Paired t-test (Welch's):** Token savings per query — baseline vs each optimization config
2. **McNemar's test:** Detection accuracy (2×2 confusion table) — baseline vs best optimization
3. **One-way ANOVA:** Mean token savings across all threshold configs (OPT1)

---

## References

1. Fowler, M. (2014). *Circuit Breaker Pattern.* martinfowler.com/bliki/CircuitBreaker.html
2. Nygard, R. T. (2018). *Release It! Design and Deploy Production-Ready Software.* Pragmatic Bookshelf.
3. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019.*
4. MAESTRO Framework Report (2026). Multi-agent failure taxonomy in production LLM systems.
5. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering.*
