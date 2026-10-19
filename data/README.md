# data/

This folder is intentionally **not published** (see `.gitignore`). It holds confidential or
internal material that the tests and evaluation commands read:

| File | Purpose |
|---|---|
| `gold_set.json`, `narrative_gold_set.json`, `cypher_gold_set.json`, `question_bank.json` | Frozen evaluation references |
| `cypher_examples.json` | Example bank for text-to-query |
| `track_a_*.json` | Simulated and sample pricing-engine (Track A) outputs |
| `knowledge_graph_ai_sample.json` | Graph sample |
| `Syntetic_Data_PoC/` | CSV source of the Neo4j import (synthetic data) |

The full graph export `axiom_pricing_module_Knowledge-graph_PoC.json` is expected in the project root.
Ask the project owner for these files and place them here before running the tests or the demo.
