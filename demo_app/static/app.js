"use strict";

/* ====================================================================== helpers */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (n, d = 2) => Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* keep status text */ }
    throw new Error(detail);
  }
  return response.json();
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

let STATUS = null;
let SELECTED = null;      // {run, case, title}: a Track A result chosen beside the question
let SOURCE = null;        // active EventSource
let RS = null;            // state of the run in progress

/* ====================================================================== the architecture */
const STAGES = [
  { id: "understand", name: "Understand the question", desc: "Finds identifiers and decides which Track A result, or the graph itself, the question is about." },
  { id: "retrieve", name: "Retrieve evidence", desc: "Hybrid retrieval over the knowledge graph: BM25 and a graph walk, fused by rank." },
  { id: "tracka", name: "Track A result", desc: "The structured output of the pricing engine that the question is about." },
  { id: "exact", name: "Exact facts from the graph", desc: "The model writes a query plan; the graph computes the value. The model never calculates." },
  { id: "pack", name: "Closed fact pack", desc: "The only figures and statements an answer may use, each with its citation." },
  { id: "generate", name: "Generate and verify", desc: "Samples are written, then checked for invented numbers, unknown citations and wrong labels." },
  { id: "reliability", name: "Reliability decision", desc: "Evidence check, corrective retrieval, quorum and agreement: serve the answer or withhold it." },
];
const ARCH = [...STAGES.map((s) => ({ id: s.id, name: { understand: "Understand", retrieve: "Retrieve", tracka: "Track A result", exact: "Exact facts", pack: "Fact pack", generate: "Generate", reliability: "Decide" }[s.id] })), { id: "evaluate", name: "Evaluate" }];

function buildStages() {
  $("#stages").innerHTML = STAGES.map((s, i) => `
    <section class="stage idle" id="st-${s.id}">
      <header data-toggle="${s.id}"><span class="num">${i + 1}</span><span class="name">${esc(s.name)}<span class="desc">${esc(s.desc)}</span></span><span class="ms"></span></header>
      <div class="body"></div>
    </section>`).join("");
  $("#arch").innerHTML = ARCH.map((s, i) => `<div class="node" id="ar-${s.id}" data-go="${s.id}"><span class="n">${i + 1}</span><b>${esc(s.name)}</b></div>`).join("");
  $$("#stages header").forEach((h) => h.addEventListener("click", () => $("#st-" + h.dataset.toggle).classList.toggle("collapsed")));
  $$(".node").forEach((n) => n.addEventListener("click", () => { const el = $("#st-" + n.dataset.go); if (el) el.scrollIntoView({ behavior: "smooth", block: "center" }); }));
}
function resetStages() {
  STAGES.forEach((s) => { const el = $("#st-" + s.id); el.className = "stage idle"; $(".body", el).innerHTML = ""; $(".ms", el).textContent = ""; });
  ARCH.forEach((s) => { $("#ar-" + s.id).className = "node"; });
}
function setStage(id, state, ms) {
  const el = $("#st-" + id); if (!el) return;
  el.className = "stage " + state;
  const arch = $("#ar-" + id);
  if (arch) arch.className = "node " + ({ running: "active", done: "done", skipped: "skipped", fail: "fail" }[state] || "");
  if (ms != null) $(".ms", el).textContent = ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : ms + " ms";
}
const body = (id) => $("#st-" + id + " .body");

/* ====================================================================== status, examples */
async function loadStatus() {
  try {
    STATUS = await api("/api/status");
    const s = STATUS;
    $("#status").innerHTML = [
      `<span class="pill"><b>${s.graph.nodes.toLocaleString("en-US")}</b> nodes</span>`,
      `<span class="pill">retrieval <b>${esc(s.retrieval.streams.join(" + "))}</b></span>`,
      `<span class="pill ${s.llm.reachable ? "ok" : "off"}"><b>${esc(s.llm.model)}</b> ${s.llm.reachable ? "ready" : "not reachable"}</span>`,
    ].join("");
    $("#hint").textContent = s.llm.reachable ? "" : "The language model is not reachable: start it with ollama serve.";
  } catch (err) {
    $("#status").innerHTML = `<span class="pill off">server error: ${esc(err.message)}</span>`;
  }
}

async function loadExamples() {
  try {
    const list = await api("/api/examples");
    $("#examples").innerHTML = "";
    list.forEach((ex) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "chip k-" + ex.kind; b.textContent = ex.text;
      b.addEventListener("click", () => { SELECTED = null; renderSelected(); $("#q").value = ex.text; startRun(); });
      $("#examples").appendChild(b);
    });
  } catch (_) { /* examples are optional */ }
}

function renderSelected() {
  const el = $("#selected");
  if (!SELECTED) { el.hidden = true; return; }
  el.hidden = false;
  el.innerHTML = `Result: ${esc(SELECTED.title)} <button type="button" title="Clear" aria-label="Clear the selected result">×</button>`;
  $("button", el).addEventListener("click", () => { SELECTED = null; renderSelected(); });
}

$("#safeguards").addEventListener("change", () => { $("#safeguards-text").textContent = $("#safeguards").checked ? "Safeguards on" : "Safeguards off"; });

/* ====================================================================== result view (shared) */
const BAND = { high: "high", medium: "medium", low: "low", none: "none" };
const pct = (x) => (x == null ? "n/a" : `${Math.round(x * 100)}%`);
const confidencePill = (c) => (c ? `${c.band ? `<span class="pillb ${BAND[c.band] || "unknown"}">${esc(c.band)}</span> ` : ""}${c.score != null ? pct(c.score) : ""}` : "");

function rangeBar(r, cur) {
  if (!r || r.low == null || r.high == null) return "";
  const pad = (r.high - r.low) * 0.15 || 1, lo = r.low - pad, hi = r.high + pad;
  const at = (v) => (100 * (v - lo) / (hi - lo)).toFixed(2);
  return `<div class="rangebar"><div class="track"></div><div class="span" style="left:${at(r.low)}%;width:${(at(r.high) - at(r.low)).toFixed(2)}%"></div>
    ${r.mid != null ? `<div class="mid" style="left:${at(r.mid)}%"></div><div class="lab top" style="left:${at(r.mid)}%">${fmt(r.mid)} ${esc(cur)}</div>` : ""}
    <div class="lab" style="left:${at(r.low)}%">${fmt(r.low)}</div><div class="lab" style="left:${at(r.high)}%">${fmt(r.high)}</div></div>`;
}
function elasticityBar(e) {
  if (!e.interval) return "";
  const [a, b] = e.interval;
  const lo = Math.min(a, e.beta ?? a, -0.25), hi = Math.max(b, e.beta ?? b, 0.25), pad = (hi - lo) * 0.08, L = lo - pad, H = hi + pad;
  const at = (v) => (100 * (v - L) / (H - L)).toFixed(2), cross = a < 0 && b > 0;
  return `<div class="elbar"><div class="track"></div><div class="span${cross ? " cross" : ""}" style="left:${at(a)}%;width:${(at(b) - at(a)).toFixed(2)}%"></div>
    <div class="zero" style="left:${at(0)}%"></div>${e.beta != null ? `<div class="beta" style="left:${at(e.beta)}%"></div>` : ""}
    <div class="lab" style="left:${at(a)}%">${fmt(a)}</div><div class="lab" style="left:${at(0)}%">0</div><div class="lab" style="left:${at(b)}%">${fmt(b)}</div></div>
    ${cross ? `<p class="small muted">The interval crosses zero: the data cannot say whether demand falls when the price rises.</p>` : ""}`;
}
function waterfallHtml(wf, cur) {
  const get = (k) => (wf.levels.find((l) => l.key === k) || {}).amount;
  const list = get("list_value"), pocket = get("pocket_price"), invoice = get("invoice_price");
  if (list == null || pocket == null) return "";
  const on = invoice != null ? Math.max(0, list - (get("negotiated_off_list") || 0) - invoice) : 0;
  const off = invoice != null ? Math.max(0, invoice - pocket) : Math.max(0, list - pocket);
  const w = (x) => (100 * x / list).toFixed(2) + "%";
  const steps = (wf.steps || []).map((s) => `<tr><td>${esc(String(s.name || "").replace(/_/g, " "))}</td><td>${s.on_invoice ? "shown on the invoice" : "<b>not</b> shown on the invoice"}</td><td class="num">${fmt(Math.abs(s.amount))}</td></tr>`).join("");
  return `<div class="wf-bar"><div class="wf-seg seg-pocket" style="width:${w(pocket)}">pocket ${fmt(pocket)}</div><div class="wf-seg seg-off" style="width:${w(off)}"></div><div class="wf-seg seg-on" style="width:${w(on)}"></div></div>
    <div class="wf-legend"><span>list ${fmt(list)} ${esc(cur)}</span><span><i style="background:#b3472f"></i>shown on invoice ${fmt(on)}</span><span><i style="background:#c98a1a"></i>not on invoice ${fmt(off)}</span><span><i style="background:var(--good)"></i>pocket ${fmt(pocket)}</span></div>
    ${steps ? `<table class="t" style="margin-top:6px"><tbody>${steps}</tbody></table>` : ""}`;
}
function viewHtml(d) {
  const v = d.view || {}, cur = d.currency || "", out = [];
  if (v.recommendation) {
    const r = v.recommendation;
    out.push(`<div class="sect"><h3>Recommendation ${r.status && r.status !== "complete" ? `<span class="pillb low">${esc(r.status)}</span>` : ""}</h3>
      ${r.price_range ? rangeBar(r.price_range, cur) : `<p class="notice warn">The engine did not recommend a price.</p>`}
      <div class="kv">${r.acceptance != null ? `<span>acceptance <b>${r.acceptance}%</b></span>` : ""}${r.comparable_deals != null ? `<span>comparable deals <b>${r.comparable_deals}</b></span>` : ""}${r.expected_margin != null ? `<span>expected margin <b>${fmt(r.expected_margin)} ${esc(cur)}</b></span>` : ""}${r.expected_margin_pct != null ? `<span>expected margin <b>${r.expected_margin_pct}%</b></span>` : ""}<span>confidence ${confidencePill(r.confidence)}</span></div>
      ${r.confidence && r.confidence.reasons && r.confidence.reasons.length ? `<ul class="plain">${r.confidence.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}</div>`);
  }
  if (v.waterfall) out.push(`<div class="sect"><h3>List price to pocket price</h3>${waterfallHtml(v.waterfall, cur)}</div>`);
  if (v.segment) {
    const s = v.segment;
    out.push(`<div class="sect"><h3>Customer segment</h3><div class="kv"><span><b>${esc(s.label || "")}</b></span>${s.volume_tier ? `<span>tier <b>${esc(s.volume_tier)}</b></span>` : ""}${s.score != null ? `<span>score <b>${s.score}%</b></span>` : ""}<span>confidence ${confidencePill(s.confidence)}</span></div></div>`);
  }
  if (v.elasticity) {
    const e = v.elasticity;
    out.push(`<div class="sect"><h3>Price elasticity ${e.fit_for_optimisation === false ? `<span class="pillb low">not fit for optimisation</span>` : ""}</h3>${elasticityBar(e)}<div class="kv">${e.n_observations != null ? `<span>observations <b>${e.n_observations}</b></span>` : ""}<span>confidence ${confidencePill(e.confidence)}</span></div></div>`);
  }
  if (v.flags) out.push(`<div class="sect"><h3>Data-quality flags</h3><ul class="plain">${v.flags.map((f) => `<li><b>${esc(f.type || "flag")}</b></li>`).join("")}</ul></div>`);
  return out.join("");
}

const TOKEN = /(\[[^\[\]\s]+\])|\b[A-Za-z][A-Za-z0-9_\-]*\d[A-Za-z0-9_\-]*\b|(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)/g;
function highlight(text, facts, allowedCitations, extraValues = []) {
  const values = [...facts.map((f) => Math.abs(f.value)), ...extraValues.map(Math.abs)];
  const allowed = new Set([...allowedCitations, ...facts.map((f) => f.citation).filter(Boolean)]);
  let out = "", last = 0;
  for (const m of text.matchAll(TOKEN)) {
    out += esc(text.slice(last, m.index));
    if (m[1]) out += `<span class="cite${allowed.has(m[1].slice(1, -1)) ? "" : " bad"}">${esc(m[1])}</span>`;
    else if (m[2]) {
      const v = Math.abs(parseFloat(m[2].replace(/,/g, ""))), year = /^(19|20)\d{2}$/.test(m[2]);
      const ok = year || values.some((x) => Math.abs(x - v) <= Math.max(0.01, 0.005 * x));
      out += year ? esc(m[2]) : `<span class="fig${ok ? "" : " bad"}">${esc(m[2])}</span>`;
    } else out += esc(m[0]);
    last = m.index + m[0].length;
  }
  return out + esc(text.slice(last));
}
const meter = (value, max, mark) => `<div class="meter"><span style="width:${Math.max(0, Math.min(100, 100 * value / max))}%"></span>${mark != null ? `<i style="left:${Math.min(100, 100 * mark / max)}%" title="threshold"></i>` : ""}</div>`;

/* ====================================================================== running a question */
$("#form").addEventListener("submit", (e) => { e.preventDefault(); startRun(); });

function startRun() {
  const q = $("#q").value.trim();
  if (q.length < 2) return;
  if (SOURCE) SOURCE.close();
  const safeguards = $("#safeguards").checked;
  resetStages();
  $("#answer").className = ""; $("#answer").innerHTML = `<div class="q-asked"><b>Question:</b> ${esc(q)}</div><p class="muted"><span class="spin"></span>Working through the pipeline…</p>`;
  $("#evaluation").className = "idle"; $("#evaluation").textContent = "Scored after the answer.";
  $("#run").disabled = true;
  RS = { q, safeguards, prevT: 0, path: null, trackA: null, exact: null, retrieval: null, samples: {}, consistency: null, k: safeguards ? 3 : 1, decision: null, facts: [], allowed: [], ctx: [] };
  setStage("understand", "running");
  const url = `/api/query?q=${encodeURIComponent(q)}&safeguards=${safeguards ? 1 : 0}` + (SELECTED ? `&result=${encodeURIComponent(SELECTED.run + ":" + SELECTED.case)}` : "");
  SOURCE = new EventSource(url);
  SOURCE.onmessage = (m) => { const ev = JSON.parse(m.data); const dt = ev.t != null ? ev.t - RS.prevT : null; if (ev.t != null) RS.prevT = ev.t; onEvent(ev, dt); };
  SOURCE.onerror = () => { if (SOURCE) { SOURCE.close(); SOURCE = null; } $("#run").disabled = false; };
}

function onEvent(ev, dt) {
  switch (ev.stage) {
    case "route": return onRoute(ev, dt);
    case "retrieval": return onRetrieval(ev, dt);
    case "trackA": return onTrackA(ev, dt);
    case "exact": return onExact(ev, dt);
    case "pack": return onPack(ev, dt);
    case "assessment": return onAssessment(ev);
    case "correction": return onCorrection(ev);
    case "sample": return onSample(ev);
    case "consistency": return onConsistency(ev);
    case "decision": return onDecision(ev, dt);
    case "evaluation": return onEvaluation(ev, dt);
    case "done":
      SOURCE && SOURCE.close(); SOURCE = null; $("#run").disabled = false;
      $("#evaluation").insertAdjacentHTML("beforeend", `<div class="timeline">Whole run: ${ev.elapsed_s} s</div>`);
      return;
    case "error":
      $("#answer").innerHTML = `<div class="notice bad"><b>Error:</b> ${esc(ev.message)}<br>Is the language model running (<code>ollama serve</code>)?</div>`;
      $("#run").disabled = false; return;
  }
}

/* ---- 1 understand */
const PATH_LABEL = { explain: "Explain a Track A result", exact: "Exact value from the graph", need_result: "Needs a result" };
function onRoute(ev, dt) {
  RS.path = ev.path;
  let html = `<div><span class="pathb ${ev.path}">${PATH_LABEL[ev.path]}</span> <span class="small">${esc(ev.reason)}</span></div>`;
  if (ev.identifiers.length) html += `<div class="kv"><span>identifiers: ${ev.identifiers.map((i) => `<span class="tag">${esc(i)}</span>`).join("")}</span></div>`;
  else html += `<p class="small muted">No identifier in the question.</p>`;
  if (ev.matched) html += `<div class="small">Track A result: <b>${esc(ev.matched.title)}</b>${ev.matched.matched_on.length ? ` · matched on ${ev.matched.matched_on.map(esc).join(", ")}` : ""}</div>`;
  if (ev.note) html += `<div class="notice warn small">${esc(ev.note)}.</div>`;
  if (ev.candidates && ev.candidates.length > 1) {
    html += `<div class="small muted" style="margin-top:6px">Other matching results (click to ask about one instead):</div><div class="chips">${ev.candidates.slice(1).map((c) => `<button type="button" class="chip" data-run="${esc(c.run)}" data-case="${esc(c.case)}" data-title="${esc(c.title)}">${esc(c.title)}</button>`).join("")}</div>`;
  }
  body("understand").innerHTML = html;
  $$("#st-understand .chip").forEach((b) => b.addEventListener("click", () => { SELECTED = { run: b.dataset.run, case: b.dataset.case, title: b.dataset.title }; renderSelected(); startRun(); }));
  setStage("understand", "done", dt);
  setStage("retrieve", "running");
  if (ev.path === "need_result") {
    $("#answer").innerHTML = `<div class="verdict declined"><div class="vh">NEEDS A RESULT</div><p>This question is about a price but does not say which one. Name a product, customer or order line (for example <span class="tag">SKU0026</span>), or choose a result:</p><button type="button" class="ghost" id="pick">Choose a Track A result</button></div>`;
    $("#pick").addEventListener("click", openDrawer);
  }
}

/* ---- 2 retrieve */
function renderRetrieval() {
  const ev = RS.retrieval; if (!ev) return;
  const cites = new Set((RS.trackA && RS.trackA.citations) || []);
  const row = (x) => `<div class="rec"><div class="h"><span class="r">${x.rank}</span><span class="id">${esc(x.record_id)}</span>${Object.entries(x.streams).map(([n, r]) => `<span class="badge ${esc(n)}">${esc(n)} #${r}</span>`).join("")}${cites.has(x.record_id) ? `<span class="star">★ cited by Track A</span>` : ""}</div><div class="t">${esc(x.text)}</div></div>`;
  body("retrieve").innerHTML = `<p class="small muted">Streams: ${ev.streams.map(esc).join(" + ")} · fused by Reciprocal Rank Fusion. A badge such as <i>bm25 #4</i> means that stream ranked the record 4th.</p>` +
    ev.results.slice(0, 5).map(row).join("") +
    (ev.results.length > 5 ? `<details><summary>show the other ${ev.results.length - 5}</summary>${ev.results.slice(5).map(row).join("")}</details>` : "") +
    (ev.results.length ? "" : `<div class="notice warn">No record shares a word with this question. BM25 matches words without stemming; the dense stream would match meaning (start the demo with <code>-Dense</code>).</div>`) +
    `<p class="timeline">${Object.entries(ev.timings_ms).map(([k, v]) => `${k} ${v} ms`).join(" · ")}</p>`;
}
function onRetrieval(ev, dt) {
  RS.retrieval = ev; renderRetrieval();
  setStage("retrieve", "done", dt);
  setStage(RS.path === "exact" ? "tracka" : "tracka", "running");
}

/* ---- 3 Track A result */
function onTrackA(ev, dt) {
  if (ev.skipped) {
    body("tracka").innerHTML = `<p class="small">${esc(ev.reason)}</p>`;
    setStage("tracka", "skipped"); setStage("exact", "running"); return;
  }
  RS.trackA = ev; RS.facts = ev.facts; RS.allowed = ev.allowed_citations; RS.ctx = ev.context_values || [];
  body("tracka").innerHTML = `<p class="small muted">${esc(ev.run_title)} · ${ev.format === "engine" ? "engine format" : "flat format"}${ev.provenance && ev.provenance.engine_version ? ` · engine ${esc(ev.provenance.engine_version)}, contract ${esc(ev.provenance.contract_version)}` : ""}</p>` + viewHtml(ev) +
    (ev.unresolvable.length ? `<div class="notice warn">${ev.unresolvable.length} cited record(s) do not exist in the graph; an explanation would be withheld.</div>` : "");
  setStage("tracka", "done", dt);
  renderRetrieval();
  setStage("exact", "running");
}

/* ---- 4 exact facts */
function onExact(ev, dt) {
  RS.exact = ev;
  if (ev.skipped) { body("exact").innerHTML = `<p class="small">${esc(ev.reason)}</p>`; setStage("exact", "skipped"); return; }
  if (!ev.success) {
    body("exact").innerHTML = `<div class="notice bad"><b>No value computed, and that is deliberate.</b> After ${ev.attempts} attempt(s) the plan could not be validated against the graph, so nothing was calculated.<br><br>${esc(ev.error || "")}</div>`;
    setStage("exact", "fail", dt); return;
  }
  const answers = Object.entries(ev.answer).map(([k, v]) => `<div class="value"><span class="k">${esc(k)}</span><span class="v">${esc(typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString("en-US") : fmt(v)) : v)}</span></div>`).join("");
  body("exact").innerHTML = `${answers}<p class="small muted">Computed by the graph, not by the model · attempts: ${ev.attempts}${ev.attempts > 1 ? " (self-healed)" : ""}</p>
    <h3>Query plan the model wrote</h3><pre class="block">${esc(JSON.stringify(ev.plan, null, 2))}</pre><h3>Equivalent Cypher</h3><pre class="block">${esc(ev.cypher || "")}</pre>
    ${ev.evidence.length ? `<h3>Cited records</h3>${ev.evidence.map((e) => `<div class="rec"><div class="h"><span class="id">${esc(e.record_id)}</span></div><div class="t">${esc(e.text)}</div></div>`).join("")}` : `<p class="small muted">A pure aggregate has no single record behind it.</p>`}`;
  setStage("exact", "done", dt);
}

/* ---- 5 fact pack */
function onPack(ev, dt) {
  RS.tPack = ev.t; RS.k = ev.k || RS.k; RS.facts = ev.facts; RS.allowed = ev.citations.concat(ev.engine_version ? [ev.engine_version] : []);
  body("pack").innerHTML = `<p class="small">${ev.figures} figure${ev.figures === 1 ? "" : "s"}${ev.statements ? `, ${ev.statements} statements from the engine` : ""} and ${ev.records} cited record${ev.records === 1 ? "" : "s"}. ${RS.q ? "The answer may use only these." : ""}</p>
    <details><summary>show the closed set</summary><table class="t"><thead><tr><th>Figure</th><th class="num">Value</th><th>Cited record</th></tr></thead><tbody>${ev.facts.map((f) => `<tr><td>${esc(f.label)}</td><td class="num">${esc(f.formatted)}</td><td class="ref">${esc(f.citation || "")}</td></tr>`).join("")}</tbody></table>${ev.statement_lines.map((s) => `<div class="stmt">${esc(s)}</div>`).join("")}</details>`;
  setStage("pack", "done", dt);
  setStage("reliability", "running");
  setStage("generate", "running");
  body("generate").innerHTML = `<p class="small muted"><span class="spin"></span>Drawing ${RS.k} sample${RS.k === 1 ? "" : "s"} from the local model…</p><div id="samples"></div>`;
}

/* ---- 6 generate and verify; 7 reliability */
function relBlock(id, html) {
  let el = $("#rel-" + id); const b = body("reliability");
  if (!el) { el = document.createElement("div"); el.id = "rel-" + id; b.appendChild(el); }
  el.innerHTML = html;
}
function onAssessment(ev) {
  setStage("reliability", "running");
  const exactPath = RS.path === "exact";
  relBlock("assess", `<h3>Evidence check</h3><p class="small">Support: <b>${ev.records}</b> record${ev.records === 1 ? "" : "s"} for <b>${ev.figures}</b> figure${ev.figures === 1 ? "" : "s"} = <b>${ev.support.toFixed(2)}</b> (floor ${ev.floor}).</p>${meter(Math.min(ev.support, 1), 1, ev.floor)}` +
    (ev.weak ? `<div class="notice warn small">${exactPath ? "A computed aggregate has no record behind it: the graph query itself attests the value." : (RS.safeguards ? "Thin evidence: the layer looks for more records about the same product, customer and order line." : "Thin evidence; without safeguards nothing is done about it.")}</div>` : `<div class="notice good small">Enough records for the figures.</div>`));
}
function onCorrection(ev) {
  relBlock("correct", `<h3>Corrective retrieval</h3><p class="small">Support ${ev.support_before.toFixed(2)} → <b>${ev.support_after.toFixed(2)}</b>. ${ev.added.length ? `Added ${ev.added.length} record${ev.added.length === 1 ? "" : "s"} naming this subject: ${ev.added.map((a) => `<span class="cite">${esc(a)}</span>`).join(" ")}. ` : "No new on-subject record found. "}${ev.rejected.length} off-subject record${ev.rejected.length === 1 ? "" : "s"} refused. Records are added, figures never.</p>`);
}
function onSample(ev) {
  RS.samples[ev.index] = ev;
  const a = ev.attribution;
  const label = !a ? "" : a.ok ? `<span><span class="mark ok">✓</span>labels correct</span>` : `<span><span class="mark no">✗</span>wrong label${RS.safeguards ? "" : " (not enforced)"}</span>`;
  const text = ev.verified ? `<details><summary>show the text</summary><p>${highlight(ev.text, RS.facts, RS.allowed, RS.ctx)}</p></details>` : `<p class="small">${esc(ev.reason || "")}${(ev.violations || []).slice(0, 2).map((v) => "<br>" + esc(v)).join("")}</p>`;
  const probs = a && !a.ok ? `<p class="small"><b>Wrong-label check:</b> ${a.problems.map(esc).join("<br>")}</p>` : "";
  const html = `<div class="sample" id="sample-${ev.index}"><div class="sh"><b>Sample ${ev.index}</b><span><span class="mark ${ev.verified ? "ok" : "no"}">${ev.verified ? "✓" : "✗"}</span>${ev.verified ? "verified" : "not verified"}</span>${label}<span>${ev.figures} figures</span><span>${ev.attempts} attempt${ev.attempts === 1 ? "" : "s"}</span></div>${text}${probs}</div>`;
  $("#samples").insertAdjacentHTML("beforeend", html);
  const p = $("#st-generate .body > p"); if (p) p.innerHTML = ev.index < RS.k ? `<span class="spin"></span>Sample ${ev.index + 1} of ${RS.k}…` : "All samples drawn.";
}
function onConsistency(ev) {
  RS.consistency = ev;
  setStage("generate", "done", RS.tPack != null ? ev.t - RS.tPack : null);
  const thr = STATUS ? STATUS.reliability.min_agreement : 0.6, quorum = STATUS ? STATUS.reliability.quorum : 2;
  relBlock("agree", RS.safeguards
    ? `<h3>Quorum and agreement</h3><p class="small">${ev.usable} of ${ev.k} samples usable (verified and correctly labelled; quorum ${quorum}). ${ev.agreement_kind === "overlap" ? `Agreement of the answers = how much of the shorter answer the longer one contains, averaged over pairs: a short answer inside a long one agrees` : `Agreement = figures a majority state / figures any sample states = ${ev.core} / ${ev.union}`} = <b>${ev.agreement.toFixed(2)}</b> (minimum ${thr}).</p>${meter(ev.agreement, 1, thr)}${ev.usable < quorum ? `<div class="notice bad small">Fewer than ${quorum} usable samples.</div>` : ""}`
    : `<h3>Single sample</h3><p class="small">One sample, checked only by the basic verifier: no quorum, no agreement test, no wrong-label enforcement.</p>`);
  if (ev.elected) { const s = $(`#sample-${ev.elected}`); if (s) s.classList.add("elected"); }
}

/* ---- answer */
function onDecision(ev, dt) {
  RS.decision = ev;
  setStage("reliability", ev.served ? "done" : "fail", RS.tPack != null ? ev.t - RS.tPack : dt);
  if (!RS.consistency) setStage("generate", ev.nogen ? "skipped" : "done");
  const q = `<div class="q-asked"><b>Question:</b> ${esc(RS.q)}</div>`;
  const ex = RS.exact && RS.exact.success ? Object.entries(RS.exact.answer).map(([k, v]) => `<div class="value"><span class="k">computed by the graph · ${esc(k)}</span><span class="v">${esc(typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString("en-US") : fmt(v)) : v)}</span></div>`).join("") : "";
  let verdict;
  if (ev.served) {
    const elected = Object.values(RS.samples).find((s) => s.text === ev.text);
    const wrong = elected && elected.attribution && !elected.attribution.ok;
    const k = RS.consistency;
    const checks = [`<li><span class="mark ok">✓</span>every figure traces to ${RS.path === "exact" ? "the graph query" : "the Track A output"}</li>`,
      `<li><span class="mark ok">✓</span>every citation resolves to a record or to the source's attestation</li>`,
      wrong ? `<li><span class="mark no">✗</span>a figure sits on the wrong label (safeguard off)</li>` : `<li><span class="mark ok">✓</span>no figure attached to the wrong label</li>`,
      RS.safeguards && k ? `<li><span class="mark ok">✓</span>${k.usable} of ${k.k} samples usable, agreement ${k.agreement.toFixed(2)}</li>` : `<li><span class="mark no">–</span>single sample, no agreement test</li>`].join("");
    verdict = `${ex}<div class="verdict served"><div class="vh">${RS.path === "exact" ? "CITED ANSWER" : (ev.question ? "ANSWER" : "EXPLANATION")}</div><div class="answer-text">${highlight(ev.text, ev.facts, ev.allowed_citations, ev.context_values || [])}</div>
      <p class="legend"><span class="fig">green</span> a figure the source sent · <span class="cite">blue</span> a citation that resolves · red would mark an unsupported one</p><ul class="checks">${checks}</ul>
      ${wrong ? `<div class="notice bad small"><b>Misattributed figure.</b> ${elected.attribution.problems.map(esc).join("<br>")}<br>With safeguards on this sample would be rejected.</div>` : ""}</div>`;
  } else {
    const declined = /declined|does not relate|cannot answer/.test(ev.reason || "") || ev.nogen;
    verdict = `${ex}<div class="verdict ${declined ? "declined" : "withheld"}"><div class="vh">${declined ? "DECLINED" : "WITHHELD"}</div><p>The system declines to answer instead of guessing.</p><p class="small"><b>Reason:</b> ${esc(ev.reason || "unknown")}</p>${declined ? "" : `<p class="small muted">Open the samples in stage 6 to see what the model said.</p>`}</div>`;
  }
  $("#answer").className = ""; $("#answer").innerHTML = q + verdict;
}

/* ---- evaluation */
const metric = (name, value, note, tip) => `<div class="metric" ${tip ? `title="${esc(tip)}"` : ""}><span>${esc(name)}${note ? `<small>${esc(note)}</small>` : ""}</span><span class="v">${value}</span></div>`;
function onEvaluation(ev, dt) {
  setStage("generate", RS.samples[1] ? "done" : "skipped");
  const a = $("#ar-evaluate"); if (a) a.className = "node done";
  renderRetrieval();
  const out = [];
  const r = ev.retrieval || {};
  if (r.trackA || r.reference) {
    out.push(`<h3>Retrieval</h3>`);
    if (r.trackA) out.push(`<div class="refsrc">Reference-free: the records Track A itself cited <span class="tagref free">no frozen reference needed</span></div>` +
      metric("Track A evidence recall@10", `${r.trackA.found} / ${r.trackA.wanted} (${Math.round(r.trackA.recall * 100)}%)`, "how many of the records the engine cited were retrieved", "Share of the records cited in the Track A result that appear in the top 10 retrieved") +
      metric("First Track A record at rank", r.trackA.first_rank ?? "not found"));
    if (r.reference) out.push(`<div class="refsrc">Against the frozen reference: ${esc(r.reference.source)} <span class="tagref ref">independent</span></div>` +
      metric("MRR", r.reference.mrr, "1 / rank of the answering record") + metric("Recall@10", r.reference.recall_at_10) + (r.reference.ndcg_at_10 != null ? metric("nDCG@10", r.reference.ndcg_at_10) : "") +
      (r.reference.primary ? metric("Answering record", r.reference.primary_rank ? `${esc(r.reference.primary)} at rank ${r.reference.primary_rank}` : `${esc(r.reference.primary)}: not in top 10`) : ""));
  }
  const g = ev.generation;
  if (g) {
    out.push(`<h3>Generation</h3>`);
    if (!g.served) out.push(`<div class="notice warn small">Nothing was served, so there is no text to score. ${esc(g.reason || "")}</div>`);
    else {
      const f = g.reference_free;
      out.push(`<div class="refsrc">Reference-free: the answer against its own closed fact pack <span class="tagref free">checks provenance</span></div>` +
        metric("Figures traced to the source", `${f.figures_traced} / ${f.figures_stated}`, "every stated number must be one the source sent") +
        metric("Citation coverage", `${Math.round(f.citation_coverage * 100)}%`, "figure sentences that cite a record") +
        metric("Labels correct", f.labels_correct ? "yes" : `no (${f.misattributed.join(", ")})`) + metric("Sample agreement", g.agreement.toFixed(2)));
      if (g.reference) {
        const x = g.reference;
        out.push(`<div class="refsrc">Against ${esc(x.source)} <span class="tagref ref">independent</span></div>` +
          metric("Faithfulness", x.faithfulness.toFixed(2), "stated figures present in the reference truth") +
          (x.answer_relevance != null ? metric("Answer relevance", x.answer_relevance.toFixed(2), "required figures actually stated") : metric("Answer relevance", "n/a", "defined for a general explanation, not for a specific question")) +
          metric("Citations", `${x.citations} (floor ${x.min_citations})`) + metric("Served-wrong", x.served_wrong ? "yes" : "no"));
      } else out.push(`<p class="small muted">No frozen reference exists for this ${RS.path === "exact" ? "answer" : "result"}, so only the reference-free checks apply.</p>`);
    }
  }
  const e = ev.exact;
  if (e) {
    out.push(`<h3>Exact value</h3>`);
    if (e.reference) out.push(`<div class="refsrc">Against ${esc(e.reference.source)} <span class="tagref ref">independent</span></div>` + metric("Computed value", `${esc(e.reference.got ?? "record")} ${e.reference.correct ? '<span class="mark ok">✓</span>' : '<span class="mark no">✗</span>'}`, `expected ${esc(e.reference.expected)}`));
    else out.push(`<p class="small muted">No frozen reference for this question. The value is exact by construction: the graph computed it.</p>`);
  }
  if (ev.error) out.push(`<div class="notice bad small">Evaluation error: ${esc(ev.error)}</div>`);
  if (!out.length) out.push(`<p class="small muted">Nothing to score for this question.</p>`);
  $("#evaluation").className = ""; $("#evaluation").innerHTML = out.join("") + `<p class="timeline">Evaluation ran in ${dt} ms.</p>`;
}

/* ====================================================================== Track A results drawer */
let RUNS = [];
function openDrawer() { $("#drawer").classList.add("open"); $("#scrim").hidden = false; $("#drawer").setAttribute("aria-hidden", "false"); loadRuns(); }
function closeDrawer() { $("#drawer").classList.remove("open"); $("#scrim").hidden = true; $("#drawer").setAttribute("aria-hidden", "true"); }
$("#open-drawer").addEventListener("click", openDrawer); $("#close-drawer").addEventListener("click", closeDrawer); $("#scrim").addEventListener("click", closeDrawer);
$("#filter").addEventListener("input", renderRuns);

async function loadRuns() { RUNS = await api("/api/runs"); renderRuns(); }
function renderRuns() {
  const f = $("#filter").value.trim().toLowerCase();
  $("#runs").innerHTML = RUNS.map((run) => {
    const items = run.case_list.filter((c) => !f || [c.id, c.title, c.product, c.customer, c.line].join(" ").toLowerCase().includes(f));
    if (!items.length) return "";
    return `<div class="runh">${esc(run.title)} · ${items.length}</div>` + items.map((c) => `
      <div class="item${c.error ? " refused" : ""}" data-run="${esc(run.id)}" data-case="${esc(c.id)}">
        <div class="row1"><span class="ttl">${esc(c.title)}<small>${c.error ? "refused: " + esc(c.error.slice(0, 80)) : `${esc(c.product || "")} · ${esc(c.customer || "")}${c.line ? " · " + esc(c.line) : ""}${c.price != null ? " · " + fmt(c.price) : ""}`}</small></span>
        ${c.error ? "" : `<span class="acts"><button type="button" class="ghost" data-act="ask">Ask</button><button type="button" class="ghost" data-act="open">Open</button></span>`}</div>
        <div class="detail"></div></div>`).join("");
  }).join("") || `<p class="muted small">No result matches.</p>`;
  $$("#runs .item").forEach((it) => {
    const run = it.dataset.run, cid = it.dataset.case, title = it.querySelector(".ttl").childNodes[0].textContent;
    const ask = it.querySelector('[data-act="ask"]'), open = it.querySelector('[data-act="open"]');
    if (ask) ask.addEventListener("click", (e) => { e.stopPropagation(); selectResult(run, cid, title); });
    if (open) open.addEventListener("click", async (e) => {
      e.stopPropagation();
      const det = $(".detail", it);
      if (it.classList.toggle("open") && !det.innerHTML) {
        det.innerHTML = `<p class="muted small"><span class="spin"></span>Loading…</p>`;
        try {
          const d = await api(`/api/runs/${encodeURIComponent(run)}/cases/${encodeURIComponent(cid)}`);
          det.innerHTML = viewHtml(d) + `<h3>Questions this result can answer</h3><div class="chips">${d.questions.map((qq) => `<button type="button" class="chip" data-q="${esc(qq)}">${esc(qq)}</button>`).join("")}</div>`;
          $$(".chip", det).forEach((b) => b.addEventListener("click", () => { selectResult(run, cid, title, b.dataset.q); }));
        } catch (err) { det.innerHTML = `<div class="notice bad small">${esc(err.message)}</div>`; }
      }
    });
  });
}
async function selectResult(run, cid, title, question) {
  SELECTED = { run, case: cid, title }; renderSelected(); closeDrawer();
  if (question) { $("#q").value = question === "Explain this result in plain words." ? "Explain this result" : question; startRun(); }
  else { if (!$("#q").value.trim()) $("#q").value = "Explain this result"; $("#q").focus(); }
}

/* add your own */
$("#file").addEventListener("change", (e) => { const f = e.target.files[0]; if (!f) return; const r = new FileReader(); r.onload = () => { $("#paste").value = String(r.result); $("#load-msg").innerHTML = `<p class="small muted">Read ${esc(f.name)}. Press Load.</p>`; }; r.readAsText(f); });
$("#load-btn").addEventListener("click", async () => {
  const text = $("#paste").value.trim();
  if (!text) { $("#load-msg").innerHTML = `<div class="notice warn small">Paste JSON or choose a file first.</div>`; return; }
  try {
    const run = await post("/api/runs/load", { name: ($("#file").files[0] || {}).name || "", text });
    $("#load-msg").innerHTML = `<div class="notice good small">Loaded: ${run.cases} result(s)${run.refused ? `, ${run.refused} refused (shown with the reason)` : ""}.</div>`;
    await loadRuns(); loadExamples();
  } catch (err) { $("#load-msg").innerHTML = `<div class="notice bad small"><b>Refused.</b> ${esc(err.message)}</div>`; }
});

/* ====================================================================== start */
buildStages(); loadStatus(); loadExamples();
const params = new URLSearchParams(location.search);
if (params.get("result")) { const [run, cs] = params.get("result").split(":"); SELECTED = { run, case: cs, title: cs }; renderSelected(); }
if (params.get("safeguards") === "0") { $("#safeguards").checked = false; $("#safeguards").dispatchEvent(new Event("change")); }
if (params.get("q")) { $("#q").value = params.get("q"); if (params.get("autorun") === "1") startRun(); }
