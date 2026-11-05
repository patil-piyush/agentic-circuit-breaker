# Agentic Circuit Breaker: Empirical Evaluation Report

**Date:** 2026-10-16 03:52:58 UTC  
**Sample Size:** 20 queries evaluated  

---

## Executive Summary

The Agentic Circuit Breaker successfully identified and interrupted semantic thrashing in **50.0%** of adversarial evaluation queries.
Across all evaluation queries, the circuit breaker reduced token consumption by an average of **27.7%** (2,783 tokens saved per run) while averting unhandled system crashes (`GraphRecursionError`).

---

## Table 1: Baseline vs Circuit Breaker Performance Comparison

| Metric | Baseline (Unprotected) | Circuit Breaker (Protected) | Relative Delta |
|:---|:---:|:---:|:---:|
| **Mean Token Consumption** | 6,153.9 ± 4,038.2 | 3,370.7 ± 1,197.7 | **-27.7%** |
| **Mean Iterations Completed** | 8.0 | 1.5 | -81.2% |
| **Failure / Crash Rate** | 50.0% (`GraphRecursionError`) | 0.0% (Graceful partial synthesis) | **-100.0%** |
| **Intervention Rate** | 0.0% | 50.0% | — |

---

## Table 2: Inferential Statistics & Effect Size

| Statistical Test | Parameter / Statistic | Value | Interpretation |
|:---|:---:|:---:|:---|
| **Paired Samples t-test** | $t(19)$ | 4.3324 | p < 0.001 (Highly Significant) |
| **Significance (p-value)** | $p$ | 0.000359 | Rejects null hypothesis ($H_0$) |
| **Cohen's $d$ Effect Size** | $d$ | 0.9688 | Large effect ($d > 0.8$) |

---

## Table 3: Per-Node Fault Localization Analysis

This table demonstrates the **per-node observability** capability, enabling developers to pinpoint which agent in the multi-agent pipeline is generating redundant outputs.

| Faulty Agent Identified | Incident Count | Share of Thrashing Incidents | Typical Fault Mode |
|:---|:---:|:---:|:---|
| `researcher` | 10 | 50.0% | Repeated extraction attempts of unavailable corpus data |

---

## Detailed Query Log

| # | Query Snippet | Baseline Status (Tokens) | Protected Status (Tokens) | Tripped Agent | Saved Tokens (%) |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | What is the exact CO2 reduction achieved per count... | crashed (10,984) | breaker_triggered (4,865) | `researcher` | 6,119 (55.7%) |
| 2 | Provide the exact solar panel efficiency improveme... | crashed (10,941) | breaker_triggered (4,839) | `researcher` | 6,102 (55.8%) |
| 3 | What were the precise GDP impact coefficients for ... | crashed (9,491) | breaker_triggered (4,261) | `researcher` | 5,230 (55.1%) |
| 4 | Provide the exact number of kilowatt-hours generat... | crashed (9,411) | breaker_triggered (4,219) | `researcher` | 5,192 (55.2%) |
| 5 | What is the monthly granularity breakdown of wind ... | crashed (9,477) | breaker_triggered (4,252) | `researcher` | 5,225 (55.1%) |
| 6 | List the exact atmospheric CO2 concentration measu... | crashed (9,442) | breaker_triggered (4,233) | `researcher` | 5,209 (55.2%) |
| 7 | What are the exact proprietary chemical formulas a... | crashed (10,891) | breaker_triggered (4,808) | `researcher` | 6,083 (55.9%) |
| 8 | Provide the precise carbon credit price per tonne ... | crashed (9,447) | breaker_triggered (4,239) | `researcher` | 5,208 (55.1%) |
| 9 | What is the exact number of hydropower turbine rev... | crashed (9,421) | breaker_triggered (4,222) | `researcher` | 5,199 (55.2%) |
| 10 | List all individual methane emission readings in p... | crashed (10,931) | breaker_triggered (4,835) | `researcher` | 6,096 (55.8%) |
| 11 | What is solar energy and how do photovoltaic panel... | completed (2,382) | completed (2,382) | — | 0 (0.0%) |
| 12 | What are the main causes of climate change? | completed (1,410) | completed (1,410) | — | 0 (0.0%) |
| 13 | How does the circuit breaker pattern work in distr... | completed (2,513) | completed (2,513) | — | 0 (0.0%) |
| 14 | What is the difference between onshore and offshor... | completed (2,259) | completed (2,259) | — | 0 (0.0%) |
| 15 | What is retrieval-augmented generation (RAG) and w... | completed (2,456) | completed (2,456) | — | 0 (0.0%) |
| 16 | How does greenhouse gas trapping cause global warm... | completed (2,625) | completed (2,625) | — | 0 (0.0%) |
| 17 | What is the role of hydropower in renewable energy... | completed (2,274) | completed (2,274) | — | 0 (0.0%) |
| 18 | How do multi-agent systems coordinate tasks in Lan... | completed (1,464) | completed (1,464) | — | 0 (0.0%) |
| 19 | What are the planets in our solar system and their... | completed (2,732) | completed (2,732) | — | 0 (0.0%) |
| 20 | What is semantic thrashing in LLM-based agents and... | completed (2,526) | completed (2,526) | — | 0 (0.0%) |
