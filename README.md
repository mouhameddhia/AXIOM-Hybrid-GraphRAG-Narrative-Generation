# AXIOM Pricing Module — Track B: Hybrid GraphRAG & Grounded Narrative

Hybrid retrieval over the AXIOM pricing knowledge graph (lexical + semantic +
graph traversal, D2), natural-language querying that lets the graph compute
exact facts instead of the model (D3), and grounded narrative generation that
explains Track A's pricing numbers with per-claim citations, self-consistency,
and a measured served-wrong rate (D4/D5/D6).

> **Start here:** the final report and handover (`rapport/final_report_latex/`) describes what was built, how to run it, its status and the next steps. The `docs/D*` files are the design records written during the work; where a number in them differs from the final report, the final report is current.

`axiom_pricing_module_Knowledge-graph_PoC.json` (5,794 nodes / 7,765
relationships) is the **single source of truth**: it is the default corpus,
what every evaluation measures against, and what the whole test suite runs on.
**No database is required** to develop, test, or demonstrate the system. A live
Neo4j instance is an interchangeable alternative source, not a prerequisite.

> `data/knowledge_graph_ai_sample.json` is an earlier 24-node extract whose
> schema has since drifted from the export (its `PriceComponent` has no
> `applies_to_id`). Nothing uses it any more; it is kept only for reference.

| Guide | Purpose |
|---|---|
| [docs/TESTING.md](docs/TESTING.md) | How to run it, worked examples, the question bank |
| [docs/MENTOR_DEMO.md](docs/MENTOR_DEMO.md) | A 10-minute walkthrough script |
| [docs/D2_hybrid_retrieval.md](docs/D2_hybrid_retrieval.md) | D2 retrieval: design, corrections, measured results |
| [docs/D3_text_to_cypher.md](docs/D3_text_to_cypher.md) | D3 text-to-query: guardrails, self-healing, measured results |
| [docs/D4_narrative_generation.md](docs/D4_narrative_generation.md) | D4 narrative: the fact-pack/verifier design, citation coverage |
| [docs/D5_D6_reliability_and_evaluation.md](docs/D5_D6_reliability_and_evaluation.md) | D5 reliability layer + D6 evaluation harness, measured results |
| [SESSION_D4_D5_D6.md](SESSION_D4_D5_D6.md) | What was built this session, in the order the defects were found |

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Staged walkthrough: lexical + graph, no model download, ~15 ms
python demo_retrieval.py --offline

# Ask your own question
python main.py --offline "What is the list price of product SKU0001?"

# Measure quality against the bilingual gold set
python evaluate_retrieval.py --cases

# Explore behaviour across 44 diverse questions, EN + AR
python run_questions.py --offline

# Prove the acceptance criterion: hybrid beats any single retriever
python run_questions.py --offline --isolate

# Ask an exact question -- the database computes the number, not the model
python ask_graph.py "How many opportunities were lost to a competitor?"

# Explain a Track A price recommendation, with per-sentence citations
python explain_price.py --facts

# The full chain: retrieval -> text-to-query -> grounded narrative
python explain_end_to_end.py --offline

# One result through the reliability layer (evidence check, corrective
# retrieval, 3 samples, wrong-label check, agreement); --list shows the 24
python scripts/explain_with_reliability.py --id REC-0005

# Reliability layer: k=3 self-consistency, hallucination checks, corrective
# retrieval, scored against the frozen 24-case gold set
python evaluate_narratives.py --offline

# Demo front end (web page; see README_DEMO.md)
.
un_demo.ps1

# Tests (hermetic: no network, no database, ~35 s)
pytest
```

Configuration is optional — the defaults are runnable as-is. Copy `.env.example`
to `.env` only to change something.

---

## Architecture

```
                        question
                           │
                   AnchorResolver          grounds identifier mentions in the graph
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
      BM25              Dense              Graph        … and any further modality
   (lexical)          (semantic)        (traversal)
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                 ReciprocalRankFuser         rank-based, weighted, k-smoothed
                           │
                CrossEncoderReranker         optional, calibrated, thresholded
                           │
                 ranked evidence + trace
```

### Why three modalities

| Modality | Catches | Would miss alone |
|---|---|---|
| **BM25** | Exact identifiers: `SKU0001`, `CUST0015`, `162.5` | Paraphrase — "price reduction" vs `invoice_discount` |
| **Dense** | Meaning without shared vocabulary | Blurs `SKU0001` and `SKU0002` together |
| **Graph** | Structurally related records sharing *no* words with the query | Anything not connected to a named entity |

### Why Reciprocal Rank Fusion

The three produce scores on incompatible scales (unbounded BM25, cosine in
`[-1,1]`, hop-decay in `(0,1]`). RRF discards magnitude and uses only rank:

```
score(d) = Σ  weight(stream) / (k + rank_in_stream(d))
```

At the default `k=60` the head of the curve is flat, so a record several
modalities rank *respectably* beats one a single modality ranks first —
cross-modal agreement is the strongest evidence signal available.

---

## Layout

```
project_trackB/
├── axiom_pricing_module_Knowledge-graph_PoC.json   # graph source (5,794 nodes)
├── data/
│   ├── gold_set.json                    # 23 scored cases (12 bilingual + 11 English)
│   ├── question_bank.json               # 44 bilingual probe questions
│   ├── cypher_examples.json             # few-shot query-plan examples
│   ├── cypher_gold_set.json             # 20 text-to-query cases
│   ├── track_a_payloads_eval.json       # 24 provisional Track A payloads (simulated)
│   ├── narrative_gold_set.json          # 24 frozen narrative cases, English, v1.1
│   └── knowledge_graph_ai_sample.json   # superseded early extract, unused
├── docs/
│   ├── D2_hybrid_retrieval.md           # design, corrections, measured results
│   ├── MENTOR_DEMO.md                   # walkthrough script
│   └── TESTING.md                       # how to run it, with examples
├── src/
│   ├── config/
│   │   ├── env.py                # typed, validated environment readers
│   │   └── settings.py           # every tunable, declared once, frozen
│   ├── graph/
│   │   ├── model.py              # KnowledgeGraph: nodes, edges, BFS traversal
│   │   ├── sources.py            # JsonGraphSource | Neo4jGraphSource
│   │   ├── neo4j_client.py       # read-only driver wrapper
│   │   └── queries.py            # schema introspection (names no label)
│   ├── retrieval/
│   │   ├── base.py               # RetrievalQuery / Result / Stream / Retriever
│   │   ├── corpus.py             # graph nodes -> citable, indexable records
│   │   ├── anchors.py            # query mentions -> graph nodes
│   │   ├── bm25_retriever.py     # lexical, Unicode/Arabic-aware tokenisation
│   │   ├── dense_retriever.py    # semantic, injectable encoder
│   │   ├── graph_retriever.py    # traversal, hop-distance scored
│   │   ├── fusion.py             # RRF over arbitrary weighted streams
│   │   └── reranker.py           # cross-encoder, sigmoid-calibrated
│   ├── pipeline/
│   │   ├── retriever_pipeline.py # orchestration + full stage trace
│   │   └── factory.py            # composition root
│   ├── evaluation/
│   │   ├── metrics.py            # recall@k, MRR, nDCG, MAP
│   │   ├── harness.py            # gold-set runner, bilingual reporting
│   │   ├── question_bank.py      # diverse probe questions, graded expectations
│   │   └── cypher_harness.py     # text-to-Cypher correctness + safety
│   ├── text_to_query/
│   │   ├── schema.py             # live schema + observed values, pruning
│   │   ├── plan.py               # the query plan: validate, execute, -> Cypher
│   │   ├── examples.py           # few-shot bank, validated against the schema
│   │   ├── guardrails.py         # read-only + reference checks (Cypher path)
│   │   ├── llm_client.py         # LLMClient protocol + Ollama
│   │   ├── evidence.py           # query rows -> citable records
│   │   ├── engine.py             # the self-healing loop
│   │   └── factory.py            # composition root
│   ├── narrative/                             # D4: grounded narrative generation
│   │   ├── contract.py           # Track A's result object -- validates, never derives
│   │   ├── evidence.py           # resolves cited handles against the graph
│   │   ├── facts.py              # the closed fact set a narrative may state
│   │   ├── graph_facts.py        # folds D3 text-to-query answers into facts
│   │   ├── generator.py          # prompt, retry, abstain
│   │   ├── verifier.py           # numbers + citation-coverage check
│   │   ├── rtl.py                # Arabic bidirectional handling (built, deferred)
│   │   └── factory.py            # composition root
│   └── reliability/                           # D5: self-consistency, corrective retrieval
│       ├── attribution.py        # hallucination check: does a claim's noun match its value
│       ├── consistency.py        # k-sample voting, consensus-core agreement
│       ├── corrective.py         # evidence assessment + on-subject re-retrieval
│       ├── narrator.py           # the assembled reliability layer
│       └── factory.py            # composition root
├── scripts/
│   ├── build_sample_payload.py       # simulates Track A (its real output doesn't exist yet)
│   └── build_narrative_gold_set.py   # freezes data/narrative_gold_set.json
└── tests/                        # 487 tests (462 offline, 25 live), ~35s
```

`src/evaluation/narrative_harness.py` (D6) sits beside the retrieval evaluation
modules above -- gold-set loading, faithfulness/relevance/coverage scoring,
and the served-wrong vs. weak-citation split, all against ground truth read
straight from `data/narrative_gold_set.json`, never through the narrative code
under test.

---

## Design principles

**No hard-coded schema.** No label, relationship type, or property name appears
anywhere in the codebase. A record's citation handle (`OrderLine:OL000001`) is
derived by scoring each property name against the node's own label:

```
OrderLine  →  {order, line}
  order_line_id →  overlap 2  ← selected
  order_id      →  overlap 1
  product_id    →  overlap 0
```

A label the graph has never contained resolves correctly without a code change.

**One retriever interface.** Every modality takes a `RetrievalQuery` and returns
`list[RetrievalResult]`. The pipeline holds a `Sequence[Retriever]` and knows
nothing about which modalities are in it — adding a fourth means appending one
object and optionally setting a weight in `FUSION_STREAM_WEIGHTS`.

**Configuration over code.** Thresholds, weights, fan-out widths, traversal
depth, and model names all live in `src/config/settings.py`, bound from the
environment at first access and validated at startup:

```
RRF_K must be >= 1, got 0
RERANK_SCORE_TRANSFORM must be one of ['identity', 'sigmoid'], got 'softmax'
```

**Dependency injection.** The encoder, cross-encoder, and graph source are all
injected. Production wires them in `factory.py`; tests substitute deterministic
fakes, which is why the suite runs in half a second with no network.

**Auditability.** Every result carries the rank it held in each contributing
stream, its hop distance and traversal path, and its pre-rerank position. A
ranking can be reconstructed and defended, not just asserted.

**Graceful degradation.** A failing modality contributes an empty stream rather
than failing the request. A cross-encoder outage falls back to fused order and
records that it did so.

---

## Configuration highlights

Full reference in `.env.example`. The values most worth knowing:

| Variable | Default | Effect |
|---|---|---|
| `CORPUS_SOURCE` | `json` | `json` or `neo4j` |
| `DEFAULT_TOP_K` | `10` | Final result count |
| `CANDIDATE_MULTIPLIER` | `4` | Per-stream fan-out before fusion |
| `RRF_K` | `60` | Higher favours cross-modal agreement |
| `FUSION_STREAM_WEIGHTS` | `{}` | e.g. `{"graph": 1.5, "bm25": 1.0}` |
| `GRAPH_MAX_HOPS` | `2` | Traversal radius |
| `GRAPH_HOP_DECAY` | `0.5` | Relevance decay per hop |
| `RERANK_TOP_N` | `20` | Cross-encoder depth (bounds latency) |
| `RERANK_MIN_SCORE` | `0.0` | Calibrated relevance floor |
| `RERANK_MIN_RESULTS` | `1` | `0` permits an empty result (abstention) |
| `NARRATIVE_MIN_CITATION_COVERAGE` | `1.0` | Share of numeric claims that must cite a record |
| `RELIABILITY_SAMPLES` | `3` | k in self-consistency (the plan specifies k=3) |
| `RELIABILITY_MIN_AGREEMENT` | `0.6` | Consensus-core floor before abstaining |
| `RELIABILITY_ENFORCE_ATTRIBUTION` | `true` | Off yields the D4 baseline for comparison |
| `RELIABILITY_MIN_SUPPORT_RATIO` | `0.75` | Records-per-figure floor that triggers corrective retrieval |

---

## Using a live Neo4j database

```bash
# in .env
CORPUS_SOURCE=neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

```bash
python demo_retrieval.py --source neo4j
RUN_LIVE_GRAPH_TESTS=1 pytest    # includes source-parity integration tests
```

Both sources produce the same `KnowledgeGraph`, so retrieval behaviour is
identical either way. `tests/test_live_graph_validation.py` asserts that parity
directly — including that write statements are refused server-side, which the
D3 text-to-Cypher layer depends on.

---

## Testing

```bash
pytest                    # 487 tests (462 pass offline, 25 live tests skipped), ~35s
pytest -k fusion          # one area
pytest -v                 # per-test names
```

| Module | Covers |
|---|---|
| `test_fusion.py` | Arbitrary arity, weighting, `k` behaviour, determinism |
| `test_corpus.py` | Identifier derivation, unseen schemas, collisions, filters |
| `test_graph_model.py` | Loaders, dangling edges, traversal ordering, self-loops |
| `test_retrievers.py` | Tokenisation (incl. Arabic), exact-id matching, hop order |
| `test_reranker.py` | Calibration, thresholds, abstention, fail-open |
| `test_retriever_pipeline.py` | End-to-end quality, trace, failure isolation |
| `test_evaluation.py` | Metric definitions, gold-set validation, bilingual split |
| `test_helpers.py` | UTF-8 console output (Arabic must not crash Windows) |
| `test_text_to_query_schema.py` | Live schema derivation, pruning, determinism |
| `test_text_to_query_guardrails.py` | Write rejection, schema-reference validation |
| `test_text_to_query_engine.py` | Self-heal loop, retry bounds, evidence, examples |
| `test_live_graph_validation.py` | Neo4j parity (skipped unless enabled) |
| `test_narrative_contract.py` | Track A's result object: boundary rejection, waterfall consistency |
| `test_narrative_facts.py` | The closed fact set: no number appears that Track A didn't send |
| `test_narrative_evidence.py` | Citation resolution against the real graph |
| `test_narrative_verifier.py` | Fabrication caught; citation-coverage enforcement |
| `test_narrative_generator.py` | Retry, temperature escalation, abstention paths |
| `test_integration_pipeline.py` | D2+D3+D4 seams: shared citation scheme, shape guard |
| `test_reliability.py` | Attribution (144-swap sweep), self-consistency, corrective retrieval |
| `test_narrative_harness.py` | D6 gold-set integrity, scoring, the D4-vs-D5 gate-value regression |

---

## Status

**Complete —** hybrid retrieval, RRF fusion, cross-encoder reranking, and the
JSON/Neo4j source abstraction. See `docs/D2_hybrid_retrieval.md` for the design
rationale, the defect register, and measured results.

**Measured —**

| Check | Result |
|---|---|
| Gold set (12 cases, EN + AR) | recall@10 **1.000**, MRR 0.635, answer found 100% |
| Question bank (44 questions, EN + AR) | 75.6% of expectations met; English 91%, Arabic 60% |
| Acceptance criterion (Section 8) | hybrid **0.701** MRR vs best single retriever 0.671 — **passes by +0.030** |

The Arabic gap is real and measured, not assumed: the corpus contains no Arabic
text, so an Arabic question without an identifier has nothing to match
lexically. A multilingual encoder is the next experiment.

**Complete (D3) —** self-healing text-to-query. Natural language becomes a
validated query plan against a schema derived live from the JSON graph; the
graph computes every number; the plan is validated before it runs and is
structurally incapable of expressing a write; failures feed the exact error
back for a bounded number of retries; answers cite the same record handles
retrieval uses. The plan also compiles to Cypher, so the query stays
reviewable and runs unchanged against Neo4j when one exists.
See `docs/D3_text_to_cypher.md`.

| Check | Result |
|---|---|
| Correct answers on the 20-question set | **19/20 (95.0%)**, live `qwen2.5:7b`, **no database** (11/12 on the original 12 questions, identical on JSON and Neo4j) |
| Mean attempts | 1.25 — most questions answered first try |
| Self-healing | Verified: recovers from bad labels, bad properties, malformed JSON, impossible filter values |
| The one failure | Genuinely ambiguous in the data — `Channel.type` literally has the value `'key account'`. Left failing rather than tuned away. |

**Complete (D4, English) —** grounded narrative generation over Track A's
result object. The model's only job is wording: every figure a narrative may
state is read from Track A's payload (or, for context, computed by the D3
graph query engine) into a closed fact set before generation, and every
figure the narrative actually states is checked against that set afterward.
`NARRATIVE_MIN_CITATION_COVERAGE=1.0` requires every sentence stating a figure
to cite the record it rests on — not just anywhere in the narrative.
Arabic is explicitly out of scope for this phase; the RTL machinery and
bilingual fact labels exist and are unit-tested (`src/narrative/rtl.py`) so
the path can be re-enabled without introducing a second, divergent set of
numbers, but nothing bilingual is claimed as done.
See `docs/D4_narrative_generation.md`.

| Check | Result |
|---|---|
| Citation coverage, enforced vs. instructed only | Instruction alone: 58% service, 0 first-attempt passes. Worked example added to the prompt: **100% service, all first-attempt** |
| Faithfulness (24-case gold set) | **1.000** — no stated figure absent from ground truth |
| Answer relevance (24-case gold set) | **1.000** |

**Complete (D5) —** the reliability layer: k=3 self-consistency with
abstain-on-disagreement, a hallucination check that catches a correctly
attested figure attached to the wrong noun (D4's verifier cannot see this —
the number is real, the citation resolves, the claim is still false), and
corrective retrieval that re-fetches on-subject records when a pack's
evidence is thin. Self-consistency measures agreement on the *consensus
core* rather than set equality, because independent samples of one pack
differ in how much they say, not what they say — a naive equality test
would abstain on every healthy run. See `docs/D5_D6_reliability_and_evaluation.md`.

| Check | Result |
|---|---|
| Attribution check: correct narratives that must pass | **24/24**, zero false positives (24-payload sweep) |
| Attribution check: injected role swaps that must be caught | **144/144** |
| The gate's value, isolated (deterministic test) | Given one verification-clean but misattributed narrative: D4-equivalent serves it (100% wrong); D5 withholds it (0%) |
| Matched 24-case run (identical checker, same model) | Baseline 96% served / 0.0% wrong; D5 100% served / 0.0% wrong — see the doc for why neither number should be over-read at n=24 |

**Complete (D6) —** the evaluation harness. A 24-case frozen gold set whose
expected values are read from disk and never through the narrative code under
test, with a drift guard that refuses to write a report if the two disagree
(verified end-to-end: a corrupted gold value produces exit code 2 and no
report file). Reports faithfulness, answer relevance, citation coverage, and
a served-wrong rate against the plan's <5% target, and every flagged
narrative is recorded verbatim so the rate can be audited by reading the
actual text. See `docs/D5_D6_reliability_and_evaluation.md` and
`SESSION_D4_D5_D6.md` for how these numbers were arrived at — including five
measurement defects found and fixed along the way.
