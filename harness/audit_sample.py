"""Manual-audit sampler for the Haiku grades (RQ plan D1).

``export`` pulls a lightly model-stratified ~30-item sample (question / final candidate / target
from the reconstructed records, Haiku's verdict + reason from the grades files) into
``results/audit_sample.csv`` with a blank ``human_valid`` column for you to fill. ``frontend``
builds a self-contained HTML page (data embedded, no server) to judge the sample by clicking
Valid/Invalid; it exports a filled CSV that drops straight back onto ``audit_sample.csv``.
``report`` re-reads the filled CSV and prints grader<->human agreement over the rows you judged.

CLI:
    python -m harness.audit_sample export [config.json]
    python -m harness.audit_sample frontend [audit_sample.csv]   # -> results/audit_frontend.html
    python -m harness.audit_sample report [audit_sample.csv]
"""
import csv
import json
import random
import sys

from harness.io_jsonl import read_records
from harness.cf_config import default_out_path, load_configs
from harness.haiku_grader import grades_path
from shared_utils.record_model import reconstruct_records

DEFAULT_CONFIG = "counterfactual_config.json"
CSV_PATH = "results/audit_sample.csv"
HTML_PATH = "results/audit_frontend.html"
COLUMNS = ["item_id", "model", "question", "candidate", "target",
           "haiku_valid", "haiku_reason", "human_valid"]
SAMPLE_N = 30


def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "t", "valid")


def export(config_path: str = DEFAULT_CONFIG, out_csv: str = CSV_PATH) -> None:
    configs = load_configs(config_path)
    per_run = max(1, round(SAMPLE_N / len(configs))) if configs else SAMPLE_N
    rng = random.Random(42)
    rows = []
    for cfg in configs:
        out_path = default_out_path(cfg)
        grades = {g["item_id"]: g for g in read_records(grades_path(out_path))}
        pool = []
        for r in reconstruct_records(read_records(out_path)):
            g = grades.get(r.item_id)
            if g is None or not r.iterations:
                continue
            pool.append({
                "item_id": r.item_id,
                "model": cfg.name,
                "question": r.question,
                "candidate": r.iterations[-1].candidate,
                "target": r.target_y_ce,
                "haiku_valid": g.get("valid_obj"),
                "haiku_reason": g.get("reason"),
                "human_valid": "",
            })
        rng.shuffle(pool)
        rows.extend(pool[:per_run])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sampled items to {out_csv} (fill the human_valid column, then: report)")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Counterfactual Audit</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1a1d21; --mut:#6b7280; --line:#e4e7eb;
          --ok:#137a3f; --okbg:#e6f4ea; --no:#b3261e; --nobg:#fce8e6; --acc:#2b5fd0; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { position:sticky; top:0; background:var(--card); border-bottom:1px solid var(--line);
           padding:10px 18px; display:flex; align-items:center; gap:16px; flex-wrap:wrap; z-index:5; }
  header h1 { font-size:16px; margin:0; font-weight:700; }
  .bar { flex:1; height:8px; background:var(--line); border-radius:5px; overflow:hidden; min-width:120px; }
  .bar > div { height:100%; background:var(--acc); width:0; transition:width .2s; }
  .stat { font-size:13px; color:var(--mut); white-space:nowrap; }
  .stat b { color:var(--ink); }
  main { max-width:820px; margin:22px auto; padding:0 18px 80px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:22px; }
  .meta { display:flex; justify-content:space-between; align-items:center; color:var(--mut);
          font-size:13px; margin-bottom:14px; }
  .tag { background:#eef1f6; border-radius:6px; padding:2px 8px; font-weight:600; color:#374151; }
  .lbl { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--mut);
         margin:16px 0 6px; font-weight:700; }
  .prob { background:#fbfcfd; border:1px solid var(--line); border-radius:8px; padding:12px 14px;
          white-space:pre-wrap; }
  .prob.edit ins { background:var(--okbg); color:var(--ok); text-decoration:none; border-radius:3px; padding:0 2px; }
  .prob.edit del { background:var(--nobg); color:var(--no); border-radius:3px; padding:0 2px; }
  .target { font-size:15px; margin:16px 0; }
  .target b { font-size:20px; background:#fff7db; border:1px solid #f2e2a6; border-radius:6px; padding:1px 10px; }
  .ask { font-weight:600; margin:18px 0 10px; }
  .ask small { display:block; font-weight:400; color:var(--mut); margin-top:3px; }
  .btns { display:flex; gap:12px; flex-wrap:wrap; }
  button.j { flex:1; min-width:140px; padding:14px; border-radius:10px; border:2px solid var(--line);
             background:#fff; font-size:15px; font-weight:700; cursor:pointer; }
  button.j.valid { color:var(--ok); }
  button.j.invalid { color:var(--no); }
  button.j.valid.on { background:var(--okbg); border-color:var(--ok); }
  button.j.invalid.on { background:var(--nobg); border-color:var(--no); }
  .note { width:100%; margin-top:12px; padding:8px 10px; border:1px solid var(--line);
          border-radius:8px; font:inherit; resize:vertical; }
  .nav { display:flex; justify-content:space-between; margin-top:18px; gap:10px; }
  .nav button, .tool button { padding:9px 16px; border-radius:8px; border:1px solid var(--line);
             background:#fff; cursor:pointer; font:inherit; }
  .nav button:disabled { opacity:.4; cursor:default; }
  .grader { margin-top:16px; border-top:1px dashed var(--line); padding-top:12px; font-size:13px; }
  .grader summary { cursor:pointer; color:var(--mut); font-weight:600; }
  .grader .v { font-weight:700; }
  .grader .v.t { color:var(--ok); } .grader .v.f { color:var(--no); }
  .agree { margin-top:8px; }
  .agree.hit { color:var(--ok); } .agree.miss { color:var(--no); }
  .tool { max-width:820px; margin:10px auto 0; padding:0 18px; display:flex; gap:10px; align-items:center;
          flex-wrap:wrap; }
  .tool .res { font-size:14px; color:var(--mut); margin-left:auto; }
  .tool .res b { color:var(--ink); font-size:16px; }
  kbd { background:#eef1f6; border:1px solid var(--line); border-bottom-width:2px; border-radius:4px;
        padding:0 5px; font-size:12px; font-family:inherit; }
</style>
</head>
<body>
<header>
  <h1>Counterfactual Audit</h1>
  <div class="bar"><div id="progress"></div></div>
  <div class="stat"><b id="ndone">0</b>/<b id="ntot">0</b> judged</div>
  <div class="stat">item <b id="pos">1</b></div>
</header>

<div class="tool">
  <button id="dl">Download CSV</button>
  <button id="reset">Reset all</button>
  <span class="res">grader-human agreement: <b id="agreepct">-</b> <span id="agreen"></span></span>
</div>

<main id="main"></main>

<script>
const DATA = __DATA__;
const COLS = ["item_id","model","question","candidate","target","haiku_valid","haiku_reason","human_valid"];
const KEY = "cf_audit_v1";
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;

function save(){ localStorage.setItem(KEY, JSON.stringify(state)); }
function graderValid(row){ return String(row.haiku_valid).trim().toLowerCase()==="true"; }

// tiny safe hyperscript: all text goes through textContent, never innerHTML
function h(tag, props){
  const e=document.createElement(tag);
  if(props) for(const k in props){
    if(k==="class") e.className=props[k];
    else if(k==="text") e.textContent=props[k];
    else if(k==="value") e.value=props[k];
    else if(k==="disabled"){ if(props[k]) e.disabled=true; }
    else if(k.slice(0,2)==="on") e.addEventListener(k.slice(2), props[k]);
    else e.setAttribute(k, props[k]);
  }
  for(let a=2;a<arguments.length;a++){
    let kids=arguments[a]; if(!Array.isArray(kids)) kids=[kids];
    for(const c of kids){ if(c==null||c===false) continue;
      e.appendChild(typeof c==="object"?c:document.createTextNode(String(c))); }
  }
  return e;
}

// word-level LCS diff -> array of DOM nodes (ins/del spans + plain text)
function diffNodes(a, b){
  const A=a.split(/(\s+)/), B=b.split(/(\s+)/), n=A.length, m=B.length;
  const dp=Array.from({length:n+1},()=>new Int32Array(m+1));
  for(let x=n-1;x>=0;x--) for(let y=m-1;y>=0;y--)
    dp[x][y] = A[x]===B[y] ? dp[x+1][y+1]+1 : Math.max(dp[x+1][y], dp[x][y+1]);
  const out=[]; let x=0,y=0;
  const push=(tag,txt)=>{ if(!txt) return;
    out.push(tag? h(tag,{text:txt}) : document.createTextNode(txt)); };
  while(x<n&&y<m){
    if(A[x]===B[y]){ push(null,B[y]); x++; y++; }
    else if(dp[x+1][y]>=dp[x][y+1]){ if(A[x].trim()) push("del",A[x]); x++; }
    else { push(B[y].trim()?"ins":null, B[y]); y++; }
  }
  while(x<n){ if(A[x].trim()) push("del",A[x]); x++; }
  while(y<m){ push(B[y].trim()?"ins":null, B[y]); y++; }
  return out;
}

function graderDetails(row, st, g){
  const kids=[
    h("summary",{text:"Show grader verdict (only after you decide)"}),
    h("div",{}, "Opus grader: ",
      h("span",{class:"v "+(g?"t":"f"), text:g?"VALID":"INVALID"}),
      " - "+(row.haiku_reason||""))];
  if(st.valid!=null){ const ok=st.valid===g;
    kids.push(h("div",{class:"agree "+(ok?"hit":"miss"),
      text: ok?"agrees with your judgement":"disagrees with your judgement"})); }
  return h("details",{class:"grader"}, kids);
}

function render(){
  const row=DATA[i], st=state[row.item_id]||{}, g=graderValid(row);
  const card=h("div",{class:"card"},
    h("div",{class:"meta"}, h("span",{text:"item "+(i+1)+" of "+DATA.length}), h("span",{class:"tag",text:row.model})),
    h("div",{class:"lbl",text:"Original problem"}),
    h("div",{class:"prob",text:row.question}),
    h("div",{class:"lbl",text:"Edited (counterfactual) - changes vs original highlighted"}),
    h("div",{class:"prob edit"}, diffNodes(row.question, row.candidate)),
    h("div",{class:"target"}, "Claimed correct answer to the edited problem: ", h("b",{text:String(row.target)})),
    h("div",{class:"ask"}, "Is this a VALID counterfactual?",
      h("small",{}, "Valid = the edited problem really solves to ", h("b",{text:String(row.target)}),
        " AND the edit is minimal (same scenario, only the change needed to move the answer).")),
    h("div",{class:"btns"},
      h("button",{class:"j valid"+(st.valid===true?" on":""), onclick:()=>judge(true)}, "Valid ", h("kbd",{text:"v"})),
      h("button",{class:"j invalid"+(st.valid===false?" on":""), onclick:()=>judge(false)}, "Invalid ", h("kbd",{text:"x"}))),
    h("textarea",{class:"note", placeholder:"optional note", value:st.note||"",
      oninput:e=>setNote(e.target.value)}),
    graderDetails(row, st, g),
    h("div",{class:"nav"},
      h("button",{disabled:i===0, onclick:()=>go(-1)}, "Prev ", h("kbd",{text:"j"})),
      h("button",{disabled:i===DATA.length-1, onclick:()=>go(1)}, "Next ", h("kbd",{text:"k"}))));
  document.getElementById("main").replaceChildren(card);
  document.getElementById("pos").textContent = i+1;
  updateStats();
}

function judge(v){ const id=DATA[i].item_id; state[id]=Object.assign({},state[id],{valid:v}); save();
  render(); if(i<DATA.length-1) setTimeout(()=>go(1),140); }
function setNote(t){ const id=DATA[i].item_id; state[id]=Object.assign({},state[id],{note:t}); save(); updateStats(); }
function go(d){ i=Math.min(DATA.length-1,Math.max(0,i+d)); render(); }

function updateStats(){
  const done=DATA.filter(r=>(state[r.item_id]||{}).valid!=null);
  document.getElementById("ndone").textContent=done.length;
  document.getElementById("ntot").textContent=DATA.length;
  document.getElementById("progress").style.width=(100*done.length/DATA.length)+"%";
  if(done.length){
    const hit=done.filter(r=>state[r.item_id].valid===graderValid(r)).length;
    document.getElementById("agreepct").textContent=(100*hit/done.length).toFixed(0)+"%";
    document.getElementById("agreen").textContent="("+hit+"/"+done.length+")";
  } else { document.getElementById("agreepct").textContent="-"; document.getElementById("agreen").textContent=""; }
}

function csvCell(s){ s=String(s==null?"":s); return /[",\n\r]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s; }
document.getElementById("dl").onclick=()=>{
  const lines=[COLS.join(",")];
  for(const row of DATA){ const st=state[row.item_id]||{};
    const out=Object.assign({}, row, {human_valid: st.valid==null?"":(st.valid?"true":"false")});
    lines.push(COLS.map(c=>csvCell(out[c])).join(",")); }
  const blob=new Blob([lines.join("\r\n")],{type:"text/csv"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob);
  a.download="audit_sample.csv"; a.click();
};
document.getElementById("reset").onclick=()=>{ if(confirm("Clear all judgements?")){ state={}; save(); render(); } };
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="TEXTAREA") return;
  if(e.key==="v") judge(true); else if(e.key==="x") judge(false);
  else if(e.key==="j"||e.key==="ArrowLeft") go(-1);
  else if(e.key==="k"||e.key==="ArrowRight") go(1);
});
render();
</script>
</body>
</html>
"""


def frontend(csv_path: str = CSV_PATH, out_html: str = HTML_PATH) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    html = _HTML_TEMPLATE.replace("__DATA__", json.dumps(rows, ensure_ascii=False))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out_html} ({len(rows)} items). Open it in a browser, judge every item, "
          f"then click 'Download CSV' and save it over {csv_path}. Finally: report")


def report(csv_path: str = CSV_PATH) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    filled = [r for r in rows if r.get("human_valid", "").strip() != ""]
    if not filled:
        print(f"No human_valid filled in {csv_path} yet.")
        return
    agree = sum(1 for r in filled if _truthy(r["haiku_valid"]) == _truthy(r["human_valid"]))
    print(f"Haiku<->human agreement: {agree}/{len(filled)} = {agree / len(filled):.1%}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "export"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "export":
        export(arg or DEFAULT_CONFIG)
    elif cmd == "frontend":
        frontend(arg or CSV_PATH)
    elif cmd == "report":
        report(arg or CSV_PATH)
    else:
        print(__doc__)
