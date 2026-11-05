# Step-by-Step Commands — Agentic Circuit Breaker (FA2 Demo)

> Run all commands from the **repo root**: `/home/Raptor/gitrepos/agentic-circuit-breaker/`

---

## STEP 0 — Navigate to Repo Root

```bash
cd /home/Raptor/gitrepos/agentic-circuit-breaker
```

---

## STEP 1 — Install Dependencies

```bash
cd apps/engine
uv sync
uv pip install -e ../../packages/agentic-circuit-breaker/
uv pip install "sqlalchemy[asyncio]" greenlet scipy
cd ../..
```

> ✅ This installs the engine, the standalone library, and all experiment dependencies.  
> If `uv` is not found, run: `pip install uv` first.

---

## STEP 2 — Verify Setup

```bash
cd apps/engine
uv run python -c "
from engine.config import get_settings
s = get_settings()
print(f'✅ Provider : {s.llm_provider}')
print(f'✅ Recursion: {s.recursion_limit}')
print(f'✅ Threshold: HALF_OPEN={s.similarity_half_open}, OPEN={s.similarity_open}')
"
cd ../..
```

**Expected output:**
```
✅ Provider : mock
✅ Recursion: 15
✅ Threshold: HALF_OPEN=0.85, OPEN=0.95
```

---

## STEP 3 — Run Unit Tests (31 tests)

```bash
cd apps/engine
uv run --with pytest --with pytest-asyncio python -m pytest tests/ -v
cd ../..
```

**Expected output:**
```
31 passed in ~21s
```

> This validates: cosine similarity, FSM transitions, embedding service,
> token counter, interceptor DB writes, and full end-to-end circuit breaker behavior.

---

## STEP 4 — Run Baseline Experiment (20 queries × 2 configs)

```bash
cd apps/engine
uv run python ../../scripts/run_experiment.py \
  --queries ../../data/adversarial_queries/all_queries.jsonl \
  --output  ../../data/experiment_results_latest.jsonl
cd ../..
```

**Expected output (summary table):**
```
#   | Type       | Baseline (Tokens/Stat)   | Protected (Tokens/Stat)           | Saved Tokens (%)
1   | thrashing  | 10,984 (crashed)         | 4,865 (breaker_triggered) [researcher] | 6,119 (55.7%)
...
11  | normal     | 2,382 (completed)        | 2,382 (completed)                 | 0 (0.0%)
...
```

> Queries 1–10 (thrashing): baseline **crashes**, protected **triggers breaker + saves ~55% tokens**  
> Queries 11–20 (normal): both configs **complete** — zero false positives ✅

---

## STEP 5 — Generate Baseline Research Report (Tables 1–3)

```bash
cd apps/engine
uv run python ../../scripts/generate_report.py \
  --results ../../data/experiment_results_latest.jsonl \
  --output  ../../data/research_paper_report.md
cd ../..
```

**Expected output:**
```
Evaluated Queries:    20
Baseline Mean Tokens: 6,154  (Crash Rate: 50%)
Protected Mean Tokens:3,371  (Crash Rate: 0%)
Average Token Savings:2,783 tokens (27.7%)
Paired t-statistic:   4.3324 (p = 0.000359)
Cohen's d:            0.9688
Fault Localization:   {'researcher': 10}
```

> Full report with Tables 1, 2, 3 written to → `data/research_paper_report.md`

---

## STEP 6 — Run Optimization Experiment (14 configs × 20 queries = 280 runs)

```bash
cd apps/engine
uv run python ../../scripts/run_optimization_experiment.py \
  --queries ../../data/adversarial_queries/all_queries.jsonl \
  --output  ../../data/optimization_results_latest.jsonl
cd ../..
```

**Expected output (summary table):**
```
Configuration                          |  TP  FP  TN  FN |  Prec  Recall    F1   FPR | Tokens | TrigIter
OPT1-A (aggressive)                    |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   3371 |      2.0
OPT1-B (baseline)                      |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   3371 |      2.0
OPT1-C (conservative)                  |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   3371 |      2.0
OPT1-D (balanced)                      |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   3371 |      2.0
OPT2-A (window=1, baseline)            |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   3371 |      2.0
OPT2-C (window=2)                      |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   4382 |      3.0
OPT2-D (window=3)                      |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   5394 |      4.0
OPT3-A (no smoothing, baseline)        |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   3371 |      2.0
OPT3-B (alpha=0.7, light)              |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   3371 |      2.0
OPT3-C (alpha=0.5, balanced)           |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   3371 |      2.0
OPT3-D (alpha=0.3, heavy)              |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   3371 |      2.0
OPT4-A (sim-only, baseline)            |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   3371 |      2.0
OPT4-B (0.8/0.2)                       |  10   0  10   0 | 1.000   1.000 1.000 0.000 |   4382 |      3.0
OPT4-C (0.6/0.4)                       |   4   0  10   6 | 1.000   0.400 0.571 0.000 |   5815 |      4.0
OPT4-D (0.5/0.5)                       |   0   0  10  10 | 0.000   0.000 0.000 0.000 |   6154 |      0.0
```

> Takes ~90 seconds total (280 pipeline runs with sentence-transformer embeddings)

---

## STEP 7 — Generate Optimization Report (Tables 4–5)

```bash
cd apps/engine
uv run python ../../scripts/generate_optimization_report.py \
  --output ../../data/optimization_report.md
cd ../..
```

**Expected output:**
```
=== BEST CONFIGURATIONS ===
  OPT1: OPT1-A (aggressive)    F1=1.000, Recall=1.000, FPR=0.000
  OPT2: OPT2-A (window=1)      F1=1.000, Recall=1.000, FPR=0.000
  OPT3: OPT3-A (no smoothing)  F1=1.000, Recall=1.000, FPR=0.000
  OPT4: OPT4-A (sim-only)      F1=1.000, Recall=1.000, FPR=0.000
```

> Full report with Tables 4, 5 + per-technique analysis written to → `data/optimization_report.md`

---

## STEP 8 — (Optional) Start the Live API Server

```bash
cd apps/engine
uv run uvicorn engine.main:app --reload --port 8000
```

Then in another terminal, send a query:

```bash
# Test a NORMAL query (should complete cleanly)
curl -s -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"query": "What is solar energy and how do photovoltaic panels work?", "use_circuit_breaker": true}' \
  | python3 -m json.tool

# Test an ADVERSARIAL query (should trigger breaker)
curl -s -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the exact CO2 reduction per country per year broken down by sector with tonnage figures?", "use_circuit_breaker": true}' \
  | python3 -m json.tool

# Health check
curl http://localhost:8000/health
```

---

## All Output Files

| File | What it is |
|:---|:---|
| `data/research_paper_report.md` | Baseline Tables 1, 2, 3 (t-test, Cohen's d, fault localization) |
| `data/optimization_report.md` | Optimization Tables 4, 5 (14 configs, F1/Recall/FPR per technique) |
| `docs/optimization_techniques.md` | Full theory behind all 4 optimizations |
| `data/experiment_results_latest.jsonl` | Raw baseline results (20 queries) |
| `data/optimization_results_latest.jsonl` | Raw optimization results (280 runs) |
| `data/optimization_metrics_*.json` | Computed precision/recall/F1 per config |
| `data/ground_truth_labels.csv` | Ground truth labels for all 20 queries |

---

## Quick Reference (all steps combined)

```bash
cd /home/Raptor/gitrepos/agentic-circuit-breaker

# 1. Install
cd apps/engine && uv sync && uv pip install -e ../../packages/agentic-circuit-breaker/ && uv pip install "sqlalchemy[asyncio]" greenlet scipy && cd ../..

# 2. Test
cd apps/engine && uv run --with pytest --with pytest-asyncio python -m pytest tests/ -v && cd ../..

# 3. Baseline experiment
cd apps/engine && uv run python ../../scripts/run_experiment.py --queries ../../data/adversarial_queries/all_queries.jsonl --output ../../data/experiment_results_latest.jsonl && cd ../..

# 4. Baseline report
cd apps/engine && uv run python ../../scripts/generate_report.py --results ../../data/experiment_results_latest.jsonl --output ../../data/research_paper_report.md && cd ../..

# 5. Optimization experiment
cd apps/engine && uv run python ../../scripts/run_optimization_experiment.py --queries ../../data/adversarial_queries/all_queries.jsonl --output ../../data/optimization_results_latest.jsonl && cd ../..

# 6. Optimization report
cd apps/engine && uv run python ../../scripts/generate_optimization_report.py --output ../../data/optimization_report.md && cd ../..
```
