# TradingAgents Extension

**Clean extension layers** for multi-agent LLM trading research: macro regime detection, three-layer decision architecture, risk budgeting, and report quality tooling.

## Built on / inspired by TradingAgents

This project is **inspired by** the popular open-source multi-agent framework
**[TradingAgents](https://github.com/TauricResearch/TradingAgents)**
(Tauric Research, Apache-2.0, [arXiv:2412.20138](https://arxiv.org/abs/2412.20138)).

TradingAgents is a well-known multi-agent LLM trading research stack
(analyst / researcher / trader / risk roles on LangGraph). This repository is a
**personal extension package** that adds macro regime tooling, layered decision
separation, risk-budget helpers, and report-quality checks on top of that idea.
It is **not** an official TradingAgents release.

> Research software only. Not investment advice. Markets can and will invalidate factor and regime heuristics.

## What this adds (vs. base multi-agent trading demos)

| Module | Role |
|--------|------|
| `tradingagents/agents/macro/` | Macro indicators + regime detector (yield curve, credit, vol, flows, growth/inflation) |
| `tradingagents/agents/analysts/factor_analyst.py` | Style-factor exposure & quadrant → factor tilt suggestions |
| `tradingagents/graph/decision_layers/` | Strategic / tactical / execution layers + coordinator |
| `tradingagents/graph/regime_bridge.py` | Hook regime detector into an agent graph |
| `tradingagents/graph/conflict_mediator.py` | Macro–micro conflict mediation node |
| `tradingagents/graph/pit_guard.py` | Point-in-time / look-ahead guard helpers |
| `tradingagents/utils/risk_framework/` | Risk budget, attribution, stress, dynamic risk |
| `tradingagents/utils/dashboard/` | Portfolio / risk monitors & alerts |
| `tradingagents/modes/layered_decision.py` | Research → Trading → Risk layered decision path |
| `tradingagents/agents/utils/fact_checker.py` | Citation enforcement for numeric claims |
| `tradingagents/agents/utils/report_auditor.py` | Report audit pass for sourced claims |
| `tradingagents/institutional_framework.py` | Single façade wiring the above together |

## Relation to upstream TradingAgents

| | |
|--|--|
| Upstream | [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) (Apache-2.0) |
| Paper | [arXiv:2412.20138](https://arxiv.org/abs/2412.20138) |
| This repo | Independent extension layers; reuses `tradingagents` import path for overlay |
| Standalone | Macro / decision / risk demos work without the full upstream graph |

Please star / cite the upstream project if you use multi-agent trading agents.
See [`NOTICE`](./NOTICE) for attribution.
## Quick start

```bash
cd trading_agents_extension
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional keys (FRED macro, LLM nodes)
cp .env.example .env
```

### Demo: extension façade (macro + layers + risk)

```bash
python examples/institutional_demo.py
```

Uses live market data via `yfinance` where available; FRED series need `FRED_API_KEY`.

### Demo: layered decision (mock data, no LLM)

```bash
python examples/layered_decision_demo.py
```

### Library usage

```python
from tradingagents.institutional_framework import InstitutionalTradingFramework

fw = InstitutionalTradingFramework(config={"macro_data_source": "yfinance"})
macro = fw.analyze_macro_environment()
print(macro["regime"])
```

```python
from tradingagents.graph.decision_layers import DecisionCoordinator

coord = DecisionCoordinator()
result = coord.run_full_cycle(total_value=100_000, force_all=True)
```

```python
from tradingagents.agents.utils.fact_checker import enforce_claim_citations

result = enforce_claim_citations("Revenue grew 25% year over year.")
print(result.checked_text)   # marks unsourced claims as (Unverified)
print(result.claim_ratio)
```

## Optional: combine with full TradingAgents

1. Clone upstream TradingAgents.
2. Put **this** package earlier on `PYTHONPATH`, or copy the extension modules into the upstream tree.
3. Enable flags in config (see `tradingagents/default_config.py`), e.g. `use_enhanced_regime_detector`, `enable_conflict_mediation`.

Full LangGraph orchestration (all stock analysts, debate, trader) lives in upstream; this repo focuses on **extension layers**.

## Project layout

```
trading_agents_extension/
├── tradingagents/
│   ├── institutional_framework.py
│   ├── default_config.py
│   ├── agents/          # macro, factor analyst, report quality
│   ├── graph/           # decision layers, bridges, guards
│   ├── modes/           # layered research → trade → risk
│   └── utils/           # risk framework, dashboard, retry, time machine
├── examples/
├── tests/
├── NOTICE               # attribution
├── LICENSE              # Apache-2.0
└── pyproject.toml
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Smoke tests cover imports and pure helpers (fact-checker, layered decision with mocks). Network-backed demos are optional.

## Design notes (honest)

- **Macro regime** heuristics are simple and regime labels are approximate; treat them as research features, not ground truth.
- **Factor analyst** style exposures are educational-grade implementations, not a production factor library.
- **Layered decision** separates research / trading / risk so disagreements are explicit and auditable.
- **Report tools** push models toward cited numeric claims; they do not verify external truth.

## Scope boundaries

This package focuses on **extension layers** only:

- Not a full trading platform, broker integration, or end-to-end production stack
- Not a dump of large research notebooks, bulk result archives, or pitch materials
- Secrets and local `.env` files are never shipped — use `.env.example` as a template

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

## Citation

If you use upstream TradingAgents ideas, please cite their paper:

```bibtex
@misc{xiao2024tradingagents,
  title={TradingAgents: Multi-Agents LLM Financial Trading Framework},
  author={Xiao, Yijia and others},
  year={2024},
  eprint={2412.20138},
  archivePrefix={arXiv}
}
```
