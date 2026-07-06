"""Tiny zero-dependency web UI to browse results/*.jsonl runs.

    python webui.py                 # serve ./results on http://127.0.0.1:8000
    python webui.py --results results --port 8000

Stdlib only (http.server). Pure read-only over the JSONL traces -- it never touches a model,
matching the decoupled-analysis principle (DESIGN 8). The renderer is schema-generic: it shows
gold/accuracy for the solve task and target_y_ce/Val for the (agentic) counterfactual task,
based on which fields are present, so it keeps working once CounterfactualTask (#12) lands.
"""
import argparse
import glob
import json
import os
import sys
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Make the repo root importable regardless of how this file is invoked (`python webui.py` from
# within harness/, or `python harness/webui.py` from the repo root) so the absolute imports below
# resolve either way.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.io_jsonl import read_records, read_meta
from shared_utils.record_model import reconstruct_records

RESULTS_DIR = "results"


def grades_path(out_path: str) -> str:
    return out_path.replace(".jsonl", ".grades.jsonl")


def is_cf_shaped(records: list) -> bool:
    """The counterfactual loop writes one flat line per STEP (kind: solve/cf, several lines per
    item); the solve task writes one nested record per ITEM. Detect the former by the 'kind' key
    every CF step line carries."""
    return bool(records) and "kind" in records[0]


def cf_records(full_path: str, raw: list) -> list:
    """Collapse the flat per-step CF lines into the same nested-iteration record shape the solve
    task already uses (via reconstruct_records), then merge in the run's Haiku grades (first CF
    vs final CF validity) when a .grades.jsonl sibling exists."""
    grades = {g["item_id"]: g for g in read_records(grades_path(full_path))}
    out = []
    for rec in reconstruct_records(raw):
        d = asdict(rec)
        g = grades.get(rec.item_id)
        if g:
            d["first_valid_obj"] = g.get("first_valid_obj")
            d["final_valid_obj"] = g.get("valid_obj")
            d["same_candidate"] = g.get("same_candidate")
        out.append(d)
    return out


def list_runs():
    """All *.jsonl files under RESULTS_DIR (recursive), with a light meta summary."""
    runs = []
    base = os.path.abspath(RESULTS_DIR)
    for path in sorted(glob.glob(os.path.join(base, "**", "*.jsonl"), recursive=True)):
        rel = os.path.relpath(path, base)
        meta = read_meta(path) or {}
        try:
            n = sum(1 for _ in open(path, encoding="utf-8"))
        except OSError:
            n = 0
        runs.append({"path": rel, "name": rel.replace(os.sep, "/"),
                     "meta": meta, "count": n})
    return runs


def safe_path(rel):
    """Resolve a client-supplied relative path, refusing anything outside RESULTS_DIR."""
    base = os.path.abspath(RESULTS_DIR)
    full = os.path.abspath(os.path.join(base, rel))
    if os.path.commonpath([base, full]) != base or not full.endswith(".jsonl"):
        return None
    return full


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet

    def _send(self, code, body, content_type):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/":
            return self._send(200, INDEX_HTML, "text/html; charset=utf-8")

        if route == "/api/runs":
            return self._json(200, {"runs": list_runs(), "results_dir": os.path.abspath(RESULTS_DIR)})

        if route == "/api/run":
            rel = (parse_qs(parsed.query).get("path") or [""])[0]
            full = safe_path(rel)
            if not full or not os.path.exists(full):
                return self._json(404, {"error": f"run not found: {rel}"})
            raw = read_records(full)
            records = cf_records(full, raw) if is_cf_shaped(raw) else raw
            return self._json(200, {"meta": read_meta(full), "records": records})

        return self._json(404, {"error": "not found"})


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Counterfactual Benchmark — Results</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --line:#2a2f3a;
          --text:#e6e9ef; --muted:#9aa3b2; --accent:#5b9dff; --good:#3fb950; --bad:#f85149; --warn:#d29922; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }
  #app { display:grid; grid-template-columns:300px 1fr; height:100vh; }
  aside { background:var(--panel); border-right:1px solid var(--line); overflow:auto; padding:14px; }
  aside h1 { font-size:15px; margin:0 0 4px; }
  #dir { color:var(--muted); font-size:11px; word-break:break-all; margin-bottom:10px; }
  button { background:var(--panel2); color:var(--text); border:1px solid var(--line);
           border-radius:6px; padding:5px 9px; cursor:pointer; font-size:12px; }
  button:hover { border-color:var(--accent); }
  ul#runs { list-style:none; margin:12px 0 0; padding:0; }
  ul#runs li { padding:8px 10px; border:1px solid var(--line); border-radius:8px; margin-bottom:8px;
               cursor:pointer; background:var(--panel2); }
  ul#runs li:hover { border-color:var(--accent); }
  ul#runs li.active { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent) inset; }
  ul#runs li .r-name { font-weight:600; word-break:break-all; }
  ul#runs li .r-sub { color:var(--muted); font-size:12px; margin-top:2px; }
  main { overflow:auto; padding:18px 22px; }
  #placeholder { color:var(--muted); margin-top:40px; text-align:center; }
  header.run { display:flex; flex-wrap:wrap; gap:8px 18px; align-items:baseline; margin-bottom:14px; }
  header.run h2 { margin:0; font-size:18px; }
  header.run .pill { background:var(--panel2); border:1px solid var(--line); border-radius:999px;
                     padding:2px 10px; font-size:12px; color:var(--muted); }
  .cards { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:16px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:120px; }
  .card .v { font-size:22px; font-weight:700; }
  .card .l { color:var(--muted); font-size:12px; }
  .controls { display:flex; gap:10px; margin-bottom:12px; flex-wrap:wrap; }
  input[type=text], select { background:var(--panel2); color:var(--text); border:1px solid var(--line);
                             border-radius:6px; padding:6px 9px; font-size:13px; }
  input[type=text] { min-width:240px; }
  table { width:100%; border-collapse:collapse; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--muted); font-weight:600; font-size:12px; position:sticky; top:0; background:var(--bg); }
  tr.item { cursor:pointer; }
  tr.item:hover td { background:var(--panel); }
  td.q { max-width:520px; color:var(--text); }
  .badge { padding:1px 8px; border-radius:999px; font-size:12px; font-weight:600; display:inline-block; }
  .b-good { background:rgba(63,185,80,.15); color:var(--good); }
  .b-bad { background:rgba(248,81,73,.15); color:var(--bad); }
  .b-warn { background:rgba(210,153,34,.15); color:var(--warn); }
  .b-mut { background:var(--panel2); color:var(--muted); }
  tr.detail td { background:var(--panel); }
  .iter { border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin:8px 0; background:var(--panel2); }
  .iter h4 { margin:0 0 6px; font-size:13px; display:flex; gap:10px; align-items:center; }
  .iter .role { color:var(--muted); font-size:12px; margin:8px 0 2px; text-transform:uppercase; letter-spacing:.04em; }
  pre { white-space:pre-wrap; word-break:break-word; margin:0; background:#0b0d12; border:1px solid var(--line);
        border-radius:6px; padding:8px 10px; font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace; max-height:340px; overflow:auto; }
  .kv { color:var(--muted); font-size:12px; }
  .mono { font-family:ui-monospace,Menlo,Consolas,monospace; }
</style>
</head>
<body>
<div id="app">
  <aside>
    <h1>Runs</h1>
    <div id="dir"></div>
    <button id="refresh">↻ Refresh</button>
    <ul id="runs"></ul>
  </aside>
  <main>
    <div id="placeholder">Select a run on the left.</div>
    <div id="run" hidden>
      <header class="run" id="run-header"></header>
      <div class="cards" id="summary"></div>
      <div class="controls">
        <input type="text" id="search" placeholder="filter by question / item id…">
        <select id="outcome">
          <option value="all">All outcomes</option>
          <option value="pass">Pass (correct / Val)</option>
          <option value="fail">Fail</option>
          <option value="genfail">Gen-fail</option>
        </select>
        <span class="kv" id="shown"></span>
      </div>
      <table>
        <thead><tr id="head-row"></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </main>
</div>
<script>
const $ = (s, r=document) => r.querySelector(s);
let CURRENT = null; // {meta, records, isCF}

const pct = x => (x*100).toFixed(1) + "%";
const esc = s => (s==null? "" : String(s)).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const short = (s,n=140) => { s=String(s==null?"":s); return s.length>n ? s.slice(0,n)+"…" : s; };

async function loadRuns() {
  const data = await (await fetch("/api/runs")).json();
  $("#dir").textContent = data.results_dir;
  const ul = $("#runs"); ul.innerHTML = "";
  if (!data.runs.length) { ul.innerHTML = '<li class="r-sub">No *.jsonl files found.</li>'; return; }
  for (const run of data.runs) {
    const c = (run.meta && run.meta.config) || {};
    const li = document.createElement("li");
    li.innerHTML = `<div class="r-name">${esc(run.name)}</div>
      <div class="r-sub">${esc(c.model||"?")} · ${esc(c.strategy||"?")} · n=${run.count}</div>`;
    li.onclick = () => { document.querySelectorAll("#runs li").forEach(x=>x.classList.remove("active"));
                         li.classList.add("active"); openRun(run.path); };
    ul.appendChild(li);
  }
}

async function openRun(path) {
  const data = await (await fetch("/api/run?path=" + encodeURIComponent(path))).json();
  if (data.error) { alert(data.error); return; }
  const records = data.records || [];
  const isCF = records.some(r => r.target_y_ce !== null && r.target_y_ce !== undefined)
            || records.some(r => r.final_val !== null && r.final_val !== undefined);
  const hasGrades = records.some(r => r.first_valid_obj !== null && r.first_valid_obj !== undefined);
  CURRENT = { meta:data.meta||{}, records, isCF, hasGrades };
  $("#placeholder").hidden = true; $("#run").hidden = false;
  renderHeader(); renderSummary(); renderHead(); applyFilter();
}

function renderHeader() {
  const c = (CURRENT.meta.config) || {};
  const pills = [["model",c.model],["task",c.task],["strategy",c.strategy],["n",c.n],
                 ["max_loops",c.max_loops],["temp",c.temp],["seed",CURRENT.meta.seed ?? c.seed],
                 ["git",(CURRENT.meta.git_hash||"").slice(0,8)]];
  $("#run-header").innerHTML = `<h2>${esc(c.task||"run")} · ${esc(c.strategy||"")}</h2>` +
    pills.filter(([,v])=>v!==undefined&&v!==null&&v!=="").map(([k,v])=>`<span class="pill">${k}: ${esc(v)}</span>`).join("");
}

const outcomeOf = r => {
  if (r.gen_fail) return "genfail";
  const v = CURRENT.isCF ? r.final_val : r.final_correct;
  if (v === true) return "pass";
  if (v === false) return "fail";
  return "unknown";
};

function renderSummary() {
  const recs = CURRENT.records, n = recs.length;
  const genfail = recs.filter(r=>r.gen_fail).length;
  const iters = recs.map(r => (r.iterations||[]).length);
  const avgIter = n ? (iters.reduce((a,b)=>a+b,0)/n) : 0;
  const cards = [];
  cards.push(["Items", n]);
  if (CURRENT.isCF) {
    const pass = recs.filter(r=>r.final_val===true).length;
    const scored = recs.filter(r=>r.final_val===true||r.final_val===false).length;
    cards.push(["Val (self-consistency)", scored? pct(pass/scored) : "—"]);
  }
  if (CURRENT.hasGrades) {
    const graded = recs.filter(r=>r.first_valid_obj!==null && r.first_valid_obj!==undefined);
    const g = graded.length;
    const firstValid = graded.filter(r=>r.first_valid_obj===true).length;
    const finalValid = graded.filter(r=>r.final_valid_obj===true).length;
    cards.push(["Val_obj (Haiku) 1st CF", g? pct(firstValid/g) : "—"]);
    cards.push(["Val_obj (Haiku) final CF", g? pct(finalValid/g) : "—"]);
    if (g) cards.push(["ΔVal_obj (final − 1st)", ((finalValid-firstValid)/g*100).toFixed(1) + "pp"]);
  }
  const corr = recs.filter(r=>r.final_correct===true).length;
  const corrScored = recs.filter(r=>r.final_correct===true||r.final_correct===false).length;
  if (corrScored) cards.push([CURRENT.isCF?"Solve acc (orig)":"Accuracy", pct(corr/corrScored)]);
  cards.push(["Gen-fail", n? pct(genfail/n) : "—"]);
  cards.push(["Avg loops", avgIter.toFixed(2)]);
  // verdict / loops-to-accept (loop strategy only)
  const accepts = recs.map(r => (r.iterations||[]).findIndex(it=>it.verdict==="accept"))
                      .filter(i=>i>=0).map(i=>i+1);
  if (accepts.length) cards.push(["Avg loops→accept", (accepts.reduce((a,b)=>a+b,0)/accepts.length).toFixed(2)]);
  $("#summary").innerHTML = cards.map(([l,v])=>`<div class="card"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
}

function renderHead() {
  const cols = ["#", "item", "question"];
  cols.push("gold");
  if (CURRENT.isCF) cols.push("target yCE");
  cols.push("orig ans");
  cols.push(CURRENT.isCF ? "Val (self)" : "correct");
  if (CURRENT.hasGrades) { cols.push("Val_obj 1st"); cols.push("Val_obj final"); }
  cols.push("loops");
  $("#head-row").innerHTML = cols.map(c=>`<th>${c}</th>`).join("");
}

function badge(outcome) {
  if (outcome==="pass") return '<span class="badge b-good">pass</span>';
  if (outcome==="fail") return '<span class="badge b-bad">fail</span>';
  if (outcome==="genfail") return '<span class="badge b-warn">gen-fail</span>';
  return '<span class="badge b-mut">—</span>';
}

function boolBadge(v) {
  if (v === true) return '<span class="badge b-good">valid</span>';
  if (v === false) return '<span class="badge b-bad">invalid</span>';
  return '<span class="badge b-mut">—</span>';
}

function applyFilter() {
  const q = $("#search").value.toLowerCase().trim();
  const f = $("#outcome").value;
  const rows = $("#rows"); rows.innerHTML = "";
  let shown = 0;
  CURRENT.records.forEach((r, i) => {
    if (q && !(String(r.question||"").toLowerCase().includes(q) || String(r.item_id||"").includes(q))) return;
    const oc = outcomeOf(r);
    if (f !== "all" && f !== oc) return;
    shown++;
    const tr = document.createElement("tr"); tr.className = "item";
    const cellList = [
      `<td class="mono">${i}</td>`,
      `<td class="mono">${esc((r.item_id||"").slice(0,8))}</td>`,
      `<td class="q">${esc(short(r.question))}</td>`,
      `<td class="mono">${esc(r.gold)}</td>`,
    ];
    if (CURRENT.isCF) cellList.push(`<td class="mono">${esc(r.target_y_ce)}</td>`);
    cellList.push(`<td class="mono">${esc(r.solver_original_answer)}</td>`, `<td>${badge(oc)}</td>`);
    if (CURRENT.hasGrades) {
      cellList.push(`<td>${boolBadge(r.first_valid_obj)}</td>`, `<td>${boolBadge(r.final_valid_obj)}</td>`);
    }
    cellList.push(`<td class="mono">${(r.iterations||[]).length}</td>`);
    tr.innerHTML = cellList.join("");
    const detail = document.createElement("tr"); detail.className = "detail"; detail.hidden = true;
    detail.innerHTML = `<td colspan="${cellList.length}"></td>`;
    tr.onclick = () => { if (detail.hidden && !detail.dataset.built) { detail.firstChild.innerHTML = renderTrace(r); detail.dataset.built="1"; }
                         detail.hidden = !detail.hidden; };
    rows.appendChild(tr); rows.appendChild(detail);
  });
  $("#shown").textContent = `${shown} / ${CURRENT.records.length} shown`;
}

function renderTrace(r) {
  let html = `<div class="kv">item_id <span class="mono">${esc(r.item_id)}</span>`;
  if (r.human_audit!=null) html += ` · human_audit: ${esc(JSON.stringify(r.human_audit))}`;
  if (r.same_candidate!=null) html += ` · 1st CF == final CF: ${esc(r.same_candidate)}`;
  html += `</div><div class="role">Question</div><pre>${esc(r.question)}</pre>`;
  const nIters = (r.iterations||[]).length;
  (r.iterations||[]).forEach((it, idx) => {
    const vb = it.verdict==="accept" ? '<span class="badge b-good">accept</span>'
             : it.verdict==="reject" ? '<span class="badge b-bad">reject</span>'
             : '<span class="badge b-mut">single-shot</span>';
    const tags = [];
    if (CURRENT.isCF && idx===0) tags.push('<span class="badge b-mut">1st CF</span>');
    if (CURRENT.isCF && idx===nIters-1) tags.push('<span class="badge b-mut">final CF</span>');
    html += `<div class="iter"><h4>iteration ${it.iteration} ${vb} ${tags.join(" ")}
       <span class="kv">`;
    if (it.solver_solve != null) html += `solver→ <b class="mono">${esc(it.solver_solve)}</b> &nbsp;`;
    if (it.verifier_says!=null || it.verifier_output!=null) {
      const vs = it.verifier_says===true ? "yes" : it.verifier_says===false ? "no" : "—";
      html += `verifier→ <b class="mono">${vs}</b>`;
    }
    html += `</span></h4>`;
    html += `<div class="role">solver output${CURRENT.isCF?" (candidate x_CE)":""}</div><pre>${esc(it.candidate)}</pre>`;
    if (it.verifier_output != null)
      html += `<div class="role">verifier output (step-check → yes/no)</div><pre>${esc(it.verifier_output)}</pre>`;
    if (CURRENT.hasGrades) {
      const gv = idx===0 ? r.first_valid_obj : (idx===nIters-1 ? r.final_valid_obj : null);
      if (gv != null) html += `<div class="role">Val_obj (Haiku-graded)</div><pre>${gv===true?"valid":"invalid"}</pre>`;
    }
    html += `</div>`;
  });
  return html;
}

$("#refresh").onclick = loadRuns;
$("#search").oninput = () => CURRENT && applyFilter();
$("#outcome").onchange = () => CURRENT && applyFilter();
loadRuns();
</script>
</body>
</html>
"""


def main():
    global RESULTS_DIR
    parser = argparse.ArgumentParser(description="Browse results/*.jsonl in a local web UI.")
    parser.add_argument("--results", default="results", help="directory of *.jsonl runs")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    args = parser.parse_args()
    RESULTS_DIR = args.results

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving results from '{os.path.abspath(RESULTS_DIR)}'")
    print(f"Open {url}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
