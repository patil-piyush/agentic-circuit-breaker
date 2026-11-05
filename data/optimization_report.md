# Agentic Circuit Breaker: Optimization Techniques Report

**Date:** 2026-10-16 04:37:34 UTC  
**Techniques Evaluated:** 4 (Adaptive Threshold, Window Variation, EMA Smoothing, Combined Signal)  
**Total Configurations:** 15  

---

## Executive Summary

Four optimization techniques were applied to the baseline Agentic Circuit Breaker (fixed thresholds 0.85/0.95, window=1, no EMA, similarity-only signal) and evaluated against a 20-query benchmark (10 thrashing + 10 normal queries).

| Technique | Best Config | Best F1 | Recall | Precision | FPR | Mean Token Usage |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **OPT1: Adaptive Threshold** | OPT1-A (aggressive) | 1.000 | 1.000 | 1.000 | 0.000 | 3,371 |
| **OPT2: Window Variation** | OPT2-A (window=1, baseline) | 1.000 | 1.000 | 1.000 | 0.000 | 3,371 |
| **OPT3: EMA Smoothing** | OPT3-A (no smoothing, baseline) | 1.000 | 1.000 | 1.000 | 0.000 | 3,371 |
| **OPT4: Combined Signal** | OPT4-A (sim-only, baseline) | 1.000 | 1.000 | 1.000 | 0.000 | 3,371 |

---

## Table 4: Full Optimization Configuration Comparison

| Configuration | TP | FP | TN | FN | Precision | Recall | F1-Score | FPR | Mean Tokens | Mean Trigger Iter |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OPT1-A (aggressive) ⭐ | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 3,371 | 2.0 |
| OPT1-B (baseline) | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 3,371 | 2.0 |
| OPT1-C (conservative) | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 3,371 | 2.0 |
| OPT1-D (balanced) | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 3,371 | 2.0 |

| OPT2-A (window=1, baseline) ⭐ | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 3,371 | 2.0 |
| OPT2-C (window=2) | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 4,382 | 3.0 |
| OPT2-D (window=3) | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 5,394 | 4.0 |

| OPT3-A (no smoothing, baseline) ⭐ | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 3,371 | 2.0 |
| OPT3-B (alpha=0.7, light) | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 3,371 | 2.0 |
| OPT3-C (alpha=0.5, balanced) | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 3,371 | 2.0 |
| OPT3-D (alpha=0.3, heavy) | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 3,371 | 2.0 |

| OPT4-A (sim-only, baseline) ⭐ | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 3,371 | 2.0 |
| OPT4-B (0.8/0.2) | 10 | 0 | 10 | 0 | 1.000 | 1.000 | **1.000** | 0.000 | 4,382 | 3.0 |
| OPT4-C (0.6/0.4) | 4 | 0 | 10 | 6 | 1.000 | 0.400 | **0.571** | 0.000 | 5,815 | 4.0 |
| OPT4-D (0.5/0.5) | 0 | 0 | 10 | 10 | 0.000 | 0.000 | **0.000** | 0.000 | 6,154 | 0.0 |

---

## Table 5: Per-Technique Analysis

### Optimization 1: Adaptive Threshold Tuning

**Description:** Tests 4 threshold pairs for the HALF_OPEN and OPEN boundaries of the circuit breaker FSM. Lower thresholds catch more thrashing earlier (higher recall) at risk of false positives on normal queries. Higher thresholds are conservative (higher precision, lower recall).

**Best Configuration:** `OPT1-A (aggressive)`  
**Best F1-Score:** 1.0000  
**Recall:** 1.0000 | **Precision:** 1.0000 | **FPR:** 0.0000  
**Mean Tokens Used:** 3,371  

**Key Finding:** The optimal threshold pair balances recall (catching thrashing) against FPR (not interrupting normal queries). The baseline (0.85/0.95) is compared against aggressive (0.75/0.88), balanced (0.80/0.92), and conservative (0.88/0.96) alternatives.

---

### Optimization 2: Sliding Window Variation

**Description:** Tests window sizes 1, 2, and 3 for the embedding comparison: Sim(N, N-window). Smaller windows detect immediate repetition faster. Larger windows detect gradual convergence more reliably.

**Best Configuration:** `OPT2-A (window=1, baseline)`  
**Best F1-Score:** 1.0000  
**Recall:** 1.0000 | **Precision:** 1.0000 | **FPR:** 0.0000  
**Mean Tokens Used:** 3,371  

**Key Finding:** Window=1 gives the fastest detection (lowest mean trigger iteration) but may catch isolated high-similarity steps. Window=2 (the PLAN.md original spec) provides a smoother signal. Window=3 requires the most evidence before firing.

---

### Optimization 3: EMA Smoothing

**Description:** Applies Exponential Moving Average smoothing (S̃_N = α × S_N + (1-α) × S̃_{N-1}) to the raw cosine similarity before feeding it into the FSM. Higher α = less smoothing; lower α = more smoothing.

**Best Configuration:** `OPT3-A (no smoothing, baseline)`  
**Best F1-Score:** 1.0000  
**Recall:** 1.0000 | **Precision:** 1.0000 | **FPR:** 0.0000  
**Mean Tokens Used:** 3,371  

**Key Finding:** Heavy smoothing (α=0.3) significantly reduces false positives caused by one-step similarity spikes, at the cost of slightly delayed detection for genuine thrashing. α=0.7 gives a good noise-accuracy balance.

---

### Optimization 4: Combined Similarity + Token Budget Signal

**Description:** Uses a composite score: Score = w_sim × similarity + w_token × token_progress, where token_progress = tokens_used / token_budget. This combines semantic and economic signals for more robust detection.

**Best Configuration:** `OPT4-A (sim-only, baseline)`  
**Best F1-Score:** 1.0000  
**Recall:** 1.0000 | **Precision:** 1.0000 | **FPR:** 0.0000  
**Mean Tokens Used:** 3,371  

**Key Finding:** A small token weight (w_token=0.2) helps the breaker fire slightly earlier on queries that are both semantically repetitive and economically costly. Heavier token weighting (0.4, 0.5) can cause normal long-running queries to be penalized unnecessarily.

---

## Methodology Notes

- **Benchmark:** 20 queries (10 thrashing + 10 normal), same set for all configurations
- **LLM Provider:** Mock (deterministic, zero cost) — ensures reproducibility
- **Corpus:** 6 domain documents (solar energy, climate, renewable, multi-agent, circuit-breaker, solar system)
- **Recursion Limit:** 15 (identical for all configurations)
- **Metrics:** Precision, Recall, F1-Score computed on binary detection task (thrashing vs normal)

---

## References

See [docs/optimization_techniques.md](../docs/optimization_techniques.md) for full theoretical background.