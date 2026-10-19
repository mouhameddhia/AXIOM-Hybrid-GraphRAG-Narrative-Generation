"""
Demo front end: a small FastAPI service and a single-page workbench.

    python -m uvicorn demo_app.server:app --port 8000     # from the project root
    (or run_demo.ps1, which also prepares the environment and opens the browser)

The user types a question; ``/api/query`` streams one event per stage of the
architecture (see :mod:`demo_app.orchestrator`) so the page can show each
component's real output as it happens. The service adds no logic of its own and is
read-only: nothing here writes to the graph.

Pricing-engine output enters as a *run* (see :mod:`demo_app.runs`): the built-in
sample in the engine's own format and the 24 simulated results are always
available; a file the user adds (here or on the page) becomes another run.

Endpoints
---------
GET  /api/status                      graph size, retrieval streams, language-model reachability
GET  /api/examples                    example questions generated from the available Track A results
GET  /api/runs                        the runs and their results (for the results browser)
POST /api/runs/load                   parse pasted or uploaded engine output into a run
GET  /api/runs/{run}/cases/{case}     one result: structured view, closed fact pack, suggested questions
GET  /api/runs/{run}/raw              the run's JSON, as loaded
GET  /api/query                       server-sent events: the question through the whole architecture

Environment: ``DEMO_DENSE=1`` adds the dense retrieval stream (about 50 s at start-up);
``DEMO_PAYLOADS=path.json`` adds a run from another result file.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from demo_app.events import fact_rows
from demo_app.orchestrator import QueryRunner
from demo_app.runs import DemoCase, DemoRun, RunFormatError, build_run, parse_text
from src.config.settings import PROJECT_ROOT, Settings, get_settings
from src.narrative.evidence import EvidenceIndex
from src.narrative.factory import build_serializer
from src.narrative.generator import NarrativeGenerator
from src.pipeline.factory import build_retrieval_stack
from src.reliability.factory import retrieval_finder
from src.reliability.narrator import ReliableNarrator
from src.text_to_query.factory import build_text_to_query_engine
from src.text_to_query.llm_client import LLMClient, OllamaClient
from src.utils.helpers import setup_logging

__all__ = ["create_app", "app"]

logger = logging.getLogger("demo_app")

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
SIMULATED = PROJECT_ROOT / "data" / "track_a_payloads_eval.json"
ENGINE_SAMPLE = PROJECT_ROOT / "data" / "track_a_real_schema_sample_output.json"

MAX_LOADED_RUNS = 8
MAX_UPLOAD_CHARS = 2_000_000


# --------------------------------------------------------------------------- state
class DemoState:
    """Everything built once at start-up and shared by the endpoints."""

    def __init__(self, settings: Settings, llm_client: LLMClient | None, dense: bool) -> None:
        self.settings = settings
        self.llm_client = llm_client or OllamaClient(settings.llm)
        self.dense = dense
        self.stack = build_retrieval_stack(settings, enable_dense=dense)
        self.graph = self.stack.graph
        self.index = EvidenceIndex.build(self.graph, build_serializer(self.graph, settings))
        self.engine = build_text_to_query_engine(self.graph, settings, llm_client=self.llm_client)
        finder = retrieval_finder(self.stack.pipeline)
        # "explain": answers about a Track A result, with corrective retrieval.
        # "exact": a sentence about a value the graph computed; a computed aggregate has no
        # record to retrieve around, so corrective retrieval is off.
        self.narrators = {
            (kind, safeguards): self._narrator(settings, finder if kind == "explain" else None, kind, safeguards)
            for kind in ("explain", "exact")
            for safeguards in (True, False)
        }
        self.runs: OrderedDict[str, DemoRun] = OrderedDict()
        self._loaded = 0
        self._lock = threading.Lock()
        self._load_builtin_runs()
        self.runner = QueryRunner(self)

    # -- runs
    def _load_builtin_runs(self) -> None:
        sources = [
            ("engine-sample", "Engine output in its own format (hand-built sample)", ENGINE_SAMPLE),
            ("simulated", "24 simulated Track A results", SIMULATED),
        ]
        custom = os.environ.get("DEMO_PAYLOADS")
        if custom:
            sources.append(("custom", f"Your file: {Path(custom).name}", Path(custom)))
        for run_id, title, path in sources:
            try:
                run = build_run(run_id, title, json.loads(path.read_text(encoding="utf-8")), builtin=True)
            except (OSError, ValueError) as exc:
                logger.warning("Could not load run %s from %s: %s", run_id, path, exc)
                continue
            self.runs[run_id] = run

    def add_run(self, title: str, data: Any) -> DemoRun:
        with self._lock:
            self._loaded += 1
            run = build_run(f"u{self._loaded}", title or f"Loaded file {self._loaded}", data)
            loaded = [r for r in self.runs.values() if not r.builtin]
            while len(loaded) >= MAX_LOADED_RUNS:
                del self.runs[loaded.pop(0).id]
            self.runs[run.id] = run
            return run

    def case(self, run_id: str, case_id: str) -> tuple[DemoRun, DemoCase]:
        run = self.runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"No run {run_id!r}")
        case = run.cases.get(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail=f"No case {case_id!r} in run {run_id!r}")
        return run, case

    def case_payload(self, run: DemoRun, case: DemoCase) -> dict[str, Any]:
        """One result as the page shows it: structured view, closed facts, questions."""
        pack = case.pack
        return {
            "run": run.id,
            "run_title": run.title,
            "format": run.format,
            "provenance": dict(run.provenance),
            "id": case.id,
            "title": case.title,
            "subject": pack.subject,
            "currency": pack.currency_code,
            "engine_version": pack.engine_version,
            "view": case.view,
            "facts": fact_rows(pack),
            "statements": list(pack.context),
            "citations": list(pack.citations),
            "unresolvable": list(self.index.missing(pack.citations)),
            "questions": list(case.questions),
            "allowed_citations": list(pack.citations) + ([pack.engine_version] if pack.engine_version else []),
            "context_values": list(pack.context_values),
        }

    # -- narrators
    def _narrator(self, settings: Settings, finder, kind: str, safeguards: bool) -> ReliableNarrator:
        reliability = settings.reliability
        if not safeguards:
            # The baseline: one sample, no quorum, no agreement test, no wrong-label
            # enforcement and no corrective retrieval -- the text is served as soon as
            # it passes the basic verifier.
            reliability = dataclasses.replace(
                reliability, samples=1, min_verified=1, enforce_attribution=False, corrective_enabled=False
            )
        elif kind == "exact":
            reliability = dataclasses.replace(reliability, corrective_enabled=False)
        generator = NarrativeGenerator(
            self.llm_client,
            settings.narrative,
            base_temperature=reliability.sample_temperature if safeguards else settings.llm.temperature,
        )
        return ReliableNarrator(generator, reliability, settings.narrative, evidence=self.index, finder=finder)

    def llm_reachable(self) -> bool:
        if not isinstance(self.llm_client, OllamaClient):
            return True  # an injected client (tests) is assumed available
        url = self.settings.llm.base_url.rstrip("/") + "/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False


class LoadRequest(BaseModel):
    name: str = Field(default="", max_length=120)
    text: str = Field(min_length=2, max_length=MAX_UPLOAD_CHARS)


# --------------------------------------------------------------------------- app
def create_app(
    *,
    settings: Settings | None = None,
    llm_client: LLMClient | None = None,
    dense: bool | None = None,
) -> FastAPI:
    """Build the application. Arguments exist so tests can inject a scripted model."""
    resolved_dense = dense if dense is not None else os.environ.get("DEMO_DENSE") == "1"
    holder: dict[str, DemoState] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        setup_logging(logging.INFO)
        started = time.time()
        holder["state"] = DemoState(settings or get_settings(), llm_client, resolved_dense)
        logger.info("Demo ready in %.1f s", time.time() - started)
        yield

    api = FastAPI(title="AXIOM Track B demo", lifespan=lifespan)

    def state() -> DemoState:
        return holder["state"]

    @api.get("/api/status")
    def status() -> dict[str, Any]:
        st = state()
        stats = st.graph.stats
        return {
            "graph": {
                "nodes": stats.node_count,
                "relationships": stats.edge_count,
                "labels": len(stats.label_counts),
                "relationship_types": len(stats.relationship_counts),
            },
            "retrieval": {"streams": list(st.stack.pipeline.stream_names), "dense": st.dense},
            "llm": {"model": st.settings.llm.model_name, "reachable": st.llm_reachable()},
            "runs": len(st.runs),
            "reliability": {
                "samples": st.settings.reliability.samples,
                "quorum": st.settings.reliability.min_verified,
                "min_agreement": st.settings.reliability.min_agreement,
                "min_support": st.settings.reliability.min_support_ratio,
                "temperature": st.settings.reliability.sample_temperature,
            },
        }

    @api.get("/api/examples")
    def examples() -> list[dict[str, str]]:
        """Questions generated from the Track A results that are actually loaded."""
        out: list[dict[str, str]] = []
        engine = state().runs.get("engine-sample")
        if engine is not None:
            full = next((c for c in engine.cases.values() if c.error is None and c.pack and "waterfall" in c.view), None)
            if full is not None and full.subject.order_line_id:
                out.append({"kind": "result", "text": f"Which deductions are not shown on the invoice for line {full.subject.order_line_id}?"})
                out.append({"kind": "result", "text": f"Is the price elasticity estimate for {full.subject.product_id} reliable enough to use?"})
            gone = next((c for c in engine.cases.values() if c.view.get("recommendation", {}).get("status") == "abstained"), None)
            if gone is not None:
                out.append({"kind": "result", "text": f"Why did the engine not recommend a price for {gone.subject.customer_id} and {gone.subject.product_id}?"})
        sim = state().runs.get("simulated")
        if sim is not None:
            cases = [c for c in sim.cases.values() if c.error is None and c.pack]
            if len(cases) > 4:
                pick = cases[4]
                out.append({"kind": "result", "text": f"Explain the price recommendation for {pick.subject.product_id} and {pick.subject.customer_id}"})
        out += [
            {"kind": "exact", "text": "How many opportunities were lost to a competitor?"},
            {"kind": "exact", "text": "What is the unit cost of product SKU0046?"},
            {"kind": "refuse", "text": "What is the capital of France?"},
        ]
        return out

    # ------------------------------------------------------------------ runs
    def _case_list(run: DemoRun) -> list[dict[str, Any]]:
        out = []
        for c in run.cases.values():
            rec = c.view.get("recommendation") or {}
            price = (rec.get("price_range") or {}).get("mid")
            out.append(
                {
                    "id": c.id,
                    "title": c.title,
                    "error": c.error,
                    "status": rec.get("status"),
                    "product": getattr(c.subject, "product_id", None),
                    "customer": getattr(c.subject, "customer_id", None),
                    "line": getattr(c.subject, "order_line_id", None),
                    "price": price,
                }
            )
        return out

    @api.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return [{**run.summary(), "case_list": _case_list(run)} for run in state().runs.values()]

    @api.post("/api/runs/load")
    def load_run(request: LoadRequest) -> dict[str, Any]:
        try:
            data = parse_text(request.text)
            run = state().add_run(request.name.strip(), data)
        except RunFormatError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {**run.summary(), "case_list": _case_list(run)}

    @api.get("/api/runs/{run_id}/raw")
    def run_raw(run_id: str) -> JSONResponse:
        run = state().runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"No run {run_id!r}")
        return JSONResponse(run.raw)

    @api.get("/api/runs/{run_id}/cases/{case_id}")
    def case_detail(run_id: str, case_id: str) -> dict[str, Any]:
        run, case = state().case(run_id, case_id)
        if case.error or case.pack is None:
            raise HTTPException(status_code=422, detail=case.error or "this result cannot be explained")
        return state().case_payload(run, case)

    # ------------------------------------------------------------------ the question
    @api.get("/api/query")
    def query(
        q: str = Query(..., min_length=2, max_length=300, description="The question"),
        result: str | None = Query(None, description="run:case of a result selected beside the question"),
        safeguards: bool = Query(True),
    ) -> StreamingResponse:
        st = state()
        selected = None
        if result and ":" in result:
            run_id, case_id = result.split(":", 1)
            selected = (run_id, case_id)
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()

        def work() -> None:
            try:
                st.runner.run(q.strip(), selected, bool(safeguards), events.put)
            except Exception as exc:  # surfaced to the page, not hidden
                logger.exception("Query failed")
                events.put({"stage": "error", "message": str(exc)})
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()

        def stream() -> Iterator[str]:
            while True:
                item = events.get()
                if item is None:
                    return
                yield f"data: {json.dumps(item, default=str)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @api.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    api.mount("/static", StaticFiles(directory=STATIC), name="static")
    return api


#: ``uvicorn demo_app.server:app``
app = create_app()
