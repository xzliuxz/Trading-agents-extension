# Architecture

## Extension layers

```
                    ┌─────────────────────────────┐
                    │  InstitutionalTradingFramework │
                    │  (façade / wiring)             │
                    └──────────────┬──────────────┘
           ┌───────────────────────┼───────────────────────┐
           ▼                       ▼                       ▼
   Macro indicators         Decision layers          Risk framework
   + RegimeDetector         Strategic / Tactical     budget / stress /
                            / Execution              dynamic / attribution
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   ▼
                         Optional LangGraph nodes
                    (regime_bridge, conflict_mediator,
                     parallel_analysts, report_summarizer)
```

## Layered research path (`modes/`)

```
Research report → Trading interpretation → Risk review
       │                    │                    │
   schemas.py        layered_decision.py    may veto / cut size
```

Disagreement tracking and horizon optimization are pure helpers (no LLM required for demos).

## Standalone vs overlay

| Mode | How |
|------|-----|
| Standalone | `pip install -e .` and use façade / decision layers / fact-checker |
| Overlay | Place this package ahead of full TradingAgents on `PYTHONPATH` |

## Out of scope

Live broker bridges, full factor research lab, multi-year walk-forward dumps, and 10k+ line reporting CLIs are intentionally not in this package.
