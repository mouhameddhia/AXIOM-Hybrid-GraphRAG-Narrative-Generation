# AXIOM demo: ask a question, watch the whole architecture work

A workbench in the browser. You type a question about a price, a Track A result, or the data, and
watch it travel through every stage of the system, each showing the real output of the component behind
it: question understanding and routing, hybrid retrieval, the Track A result, exact facts computed by the
graph, the closed fact pack, generation and verification, the reliability decision, and finally a live
evaluation of the outcome with the project's own evaluation code.

It runs entirely on your machine and never writes to the graph. Nothing on the page is stored or static.

## Start it (3 steps)

1. **Install once:** Python 3.13 or later ([python.org](https://www.python.org/downloads/)) and
   [Ollama](https://ollama.com) with the model:
   ```powershell
   ollama pull qwen2.5:7b
   ```
2. **Start the model** (leave this window open): `ollama serve`
3. **Start the demo** (new PowerShell window, in this folder):
   ```powershell
   .\run_demo.ps1
   ```
   The first run creates the environment and installs packages (a few minutes), then opens
   <http://127.0.0.1:8000>. Press Ctrl+C to stop.

If scripts are blocked: `powershell -ExecutionPolicy Bypass -File .\run_demo.ps1`.

## What to ask

Click an example, or type your own. Three kinds of question, routed automatically (the route and its
reason are shown in stage 1):

| You ask | Route | Example |
|---|---|---|
| About a Track A result, naming a product, customer, order line or id | **Explain a Track A result** | *Why is the pocket price lower than the invoice price for line OL000248?* · *Explain the price recommendation for SKU0037 and CUST0050* |
| For a value stored in the graph | **Exact value from the graph** | *How many opportunities were lost to a competitor?* · *What is the unit cost of product SKU0046?* |
| About a price without naming anything | **Needs a result** | *Explain the recommendation* (then choose one) |

A question the data cannot answer, that is unrelated to the result, or that asks for advice (*Should we give
a bigger discount?*; the system explains figures and does not recommend actions) is **declined, not guessed**
(with a result selected, an unrelated question is declined before any model call; with none selected, *What is the
capital of France?* goes to the graph, finds nothing and is withheld after about 30 seconds).

## Track A results

**Track A results** (top right) opens the list of available outputs of the pricing engine: the
hand-built sample in the engine's own format (a full case and an abstained recommendation) and 24
simulated results. Filter them, open one to see its structure and the questions it can answer, or select
one to ask about it. If your question names none, the selected result is used.

The 24 simulated results use real waterfall levels and record identifiers; recommended prices, ranges,
confidence, margin and acceptance probability are illustrative. The engine-format sample is hand-built.

## The evaluation panel

After each answer the outcome is scored, and every metric says what it is measured against:

* **Retrieval:** how many of the records the Track A result itself cites were retrieved (reference-free);
  MRR, recall and nDCG when the question is one of the frozen retrieval questions.
* **Generation:** figures traced to the source, citation coverage, correct labels and sample agreement
  (reference-free); for the 24 simulated results also faithfulness, answer relevance and the served-wrong
  flag against the frozen reference, whose expected figures were recorded independently of the code.
* **Exact value:** the computed value against the independently computed answer, when the question is in
  the reference set.

The **Safeguards** switch turns off the reliability layer: one sample, checked only by the basic verifier.
It shows what the layer protects against (for example, a figure on the wrong label gets served).

## For the Track A colleague: use it as a test bench

* **Track A results → Add your own Track A output:** paste or load your run (the engine's run object or
  flat results). A malformed result, or one whose figures do not add up, is **refused with the reason**
  (the system does not correct figures it does not own). To load a file at start-up:
  `$env:DEMO_PAYLOADS = "C:\path\to\run.json"; .\run_demo.ps1`.
* The reader for the engine's format is **experimental**: it follows the 0.2.0-draft shape and a *guess*
  at the itemised `steps`. Identifier lists (`transaction_ids`, `discount_ids`, `customer_ids`,
  `product_ids`) become graph handles through one table, `ENGINE_ID_LABELS` in
  `src/narrative/engine_draft.py`. It refuses: amounts that say `complete` but carry no value; confidence
  or probability outside 0 to 1; an inverted range; waterfall levels that do not reconcile; itemised steps
  that do not sum. An unavailable amount is stated as unavailable, never as zero.
* Comparison of the engine's draft with the flat format: `docs/TRACK_A_REAL_CONTRACT.md`. Next steps are in
  the report (`rapport/final_report_latex/Final_Report_and_Handover.pdf`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| "model not reachable" (top right) | Run `ollama serve` and `ollama pull qwen2.5:7b`. |
| Stage 2 says no record shares a word with the question | Lexical search matches words without stemming. Use an identifier (SKU0046) or words from the data, or start with `.\run_demo.ps1 -Dense` for meaning-based search. |
| An answer is **withheld** | Read the reason under it. Typical: the three samples did not agree, or one was flagged. Open the samples in stage 6 to see what the model said. Ask again or rephrase. |
| A run takes 20 to 70 seconds | Expected: three samples are drawn from a local 7B model. An exact question and a specific question are faster than a general explanation. |
| Port in use | `.\run_demo.ps1 -Port 9000` |
| Different wording each time | Expected: the layer samples at temperature 0.7. The serve-or-withhold decision is what is reproducible. |

## Where things are

`demo_app/orchestrator.py` (routing, the stages, live evaluation), `demo_app/server.py` (the service),
`demo_app/runs.py` (Track A output as runs and cases), `demo_app/events.py`, `demo_app/static/` (the page),
`src/narrative/engine_draft.py` (reader for the engine's format), `src/narrative/relevance.py` (the
relevance gate), `tests/test_demo_app.py` and `tests/test_engine_draft.py` (offline tests). Full
documentation: `README.md` and `rapport/final_report_latex/`.
