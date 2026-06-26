"""
EcoLens Objective 2 - Step 8: Retrieval Dashboard (Multi-Model)

Generates an interactive standalone HTML dashboard visualizing
the retrieval engine results and evaluation metrics from Steps 6-7.

Supports multiple foundation models with a model selector dropdown.

Features:
  - Model selector (Prithvi-100M / ViT-Base / ResNet-50)
  - Cross-model performance comparison bar chart
  - Similarity method selector (cosine / euclidean / kNN)
  - Per-category performance bar chart
  - Confusion matrix heatmap
  - Enhanced ecosystem search with multi-method comparison

Prerequisites:
    Run 06_retrieval_engine.py and 07_evaluate_retrieval.py first.

Run:
    python 08_retrieval_dashboard.py
"""

import json
import os
import numpy as np
from config import (
    METADATA_CATALOG_PATH, EMBEDDINGS_DIR, RESULTS_DIR,
    SUPPORTED_MODELS,
)


def cosine_similarity(a, b):
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_similarity(a, b):
    return 1.0 / (1.0 + float(np.linalg.norm(a - b)))


def compute_model_data(catalog, model_key):
    """Compute PCA coordinates + similarity matrices for a single model."""
    model_cfg = SUPPORTED_MODELS[model_key]
    emb_dir = model_cfg["embeddings_dir"]

    ids, vectors, valid = [], [], []
    for entry in catalog:
        emb_path = f"{emb_dir}/{entry['id']}.npy"
        if os.path.exists(emb_path):
            ids.append(entry["id"])
            vectors.append(np.load(emb_path))
            valid.append(entry)

    if len(valid) < 2:
        return None

    X = np.stack(vectors)

    # PCA projection
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    X_2d = X_centered @ eigenvectors[:, idx][:, :2]

    # Compute similarity matrices for all three methods
    sims_cos, sims_euc, sims_knn = {}, {}, {}

    for i, id_a in enumerate(ids):
        sims_cos[id_a], sims_euc[id_a], sims_knn[id_a] = {}, {}, {}
        for j, id_b in enumerate(ids):
            cs = cosine_similarity(vectors[i], vectors[j])
            es = euclidean_similarity(vectors[i], vectors[j])
            sims_cos[id_a][id_b] = cs
            sims_euc[id_a][id_b] = es
            sims_knn[id_a][id_b] = es

    dashboard_data = []
    for i, entry in enumerate(valid):
        dashboard_data.append({
            "id": entry["id"], "ecosystem": entry["ecosystem"],
            "name": entry["name"], "lon": entry["lon"], "lat": entry["lat"],
            "protected_area": entry.get("protected_area", False),
            "climatic_region": entry.get("climatic_region", "Unknown"),
            "x": float(X_2d[i, 0]), "y": float(X_2d[i, 1]),
        })

    return {
        "dashboard_data": dashboard_data,
        "sims_cos": sims_cos,
        "sims_euc": sims_euc,
        "sims_knn": sims_knn,
    }


def build_html(all_model_data, eval_data, model_labels):
    # Use first available model as default
    default_model = list(all_model_data.keys())[0]

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EcoLens: Multi-Model Ecosystem Retrieval Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root {{--bg:#0b0f19;--panel:rgba(20,27,45,.7);--border:rgba(255,255,255,.08);--t1:#f8fafc;--t2:#94a3b8;--pri:#8b5cf6;--glow:rgba(139,92,246,.3);--green:#10b981;--blue:#3b82f6;--orange:#f59e0b;--pink:#ec4899;--cyan:#06b6d4;--ff:'Plus Jakarta Sans',sans-serif;--tf:'Outfit',sans-serif}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);background-image:radial-gradient(circle at 10% 20%,rgba(99,102,241,.05) 0%,transparent 40%),radial-gradient(circle at 90% 80%,rgba(139,92,246,.05) 0%,transparent 40%);color:var(--t1);font-family:var(--ff);line-height:1.6;padding:30px}}
.container{{max-width:1500px;margin:0 auto}}
header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:30px;border-bottom:1px solid var(--border);padding-bottom:20px;flex-wrap:wrap;gap:15px}}
.brand h1{{font-family:var(--tf);font-size:2rem;font-weight:700;background:linear-gradient(135deg,#a78bfa,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.brand p{{color:var(--t2);font-size:.9rem;font-weight:300}}
.quick-stats{{display:flex;gap:15px;flex-wrap:wrap}}
.stat-card{{background:var(--panel);border:1px solid var(--border);backdrop-filter:blur(16px);border-radius:12px;padding:10px 18px;text-align:center;min-width:120px}}
.stat-card .label{{font-size:.7rem;color:var(--t2);text-transform:uppercase;letter-spacing:1px;margin-bottom:2px}}
.stat-card .value{{font-family:var(--tf);font-size:1.3rem;font-weight:700}}
.stat-card.glow .value{{color:var(--green)}}
.model-select-bar{{background:var(--panel);border:1px solid var(--border);backdrop-filter:blur(16px);border-radius:12px;padding:14px 20px;margin-bottom:25px;display:flex;align-items:center;gap:15px;flex-wrap:wrap}}
.model-select-bar label{{font-size:.85rem;color:var(--t2);font-weight:600;text-transform:uppercase;letter-spacing:1px}}
.model-select{{background:#141b2d;border:1px solid var(--border);color:var(--t1);padding:8px 14px;border-radius:8px;font-family:var(--ff);font-size:.9rem;cursor:pointer;outline:none;min-width:200px}}
.model-select:focus{{border-color:var(--pri)}}
.model-desc{{color:var(--t2);font-size:.82rem;font-style:italic}}
.grid-main{{display:grid;grid-template-columns:1fr 380px;gap:25px;align-items:start}}
@media(max-width:1024px){{.grid-main{{grid-template-columns:1fr}}}}
.panel{{background:var(--panel);border:1px solid var(--border);backdrop-filter:blur(16px);border-radius:16px;padding:22px;margin-bottom:25px}}
.panel-title{{font-family:var(--tf);font-size:1.2rem;font-weight:600;margin-bottom:16px;display:flex;align-items:center;gap:8px}}
.metrics-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:20px;margin-bottom:25px}}
.chart-box{{background:rgba(10,14,23,.4);border-radius:12px;border:1px solid rgba(255,255,255,.03);padding:15px;height:320px;position:relative}}
.method-tabs{{display:flex;gap:8px;margin-bottom:15px;flex-wrap:wrap}}
.method-tab{{background:rgba(255,255,255,.04);border:1px solid var(--border);color:var(--t2);padding:6px 14px;border-radius:20px;cursor:pointer;font-size:.82rem;transition:all .2s}}
.method-tab:hover{{background:rgba(255,255,255,.08);color:var(--t1)}}
.method-tab.active{{background:var(--pri);border-color:var(--pri);color:var(--t1);box-shadow:0 0 12px var(--glow)}}
.search-select{{width:100%;background:#141b2d;border:1px solid var(--border);color:var(--t1);padding:10px;border-radius:8px;font-family:var(--ff);font-size:.9rem;margin-bottom:16px;cursor:pointer;outline:none}}
.search-select:focus{{border-color:var(--pri)}}
.detail-card{{background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:18px;display:none}}
.detail-title{{font-family:var(--tf);font-size:1.05rem;font-weight:600;margin-bottom:12px;border-bottom:1px solid rgba(255,255,255,.05);padding-bottom:8px}}
.meta-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;font-size:.82rem}}
.meta-item .label{{color:var(--t2);font-size:.7rem;text-transform:uppercase;letter-spacing:.5px}}
.meta-item .val{{font-weight:600}}
.ranking-item{{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:8px;margin-bottom:8px;transition:transform .2s,background .2s;cursor:pointer}}
.ranking-item:hover{{transform:translateX(4px);background:rgba(255,255,255,.05);border-color:rgba(139,92,246,.3)}}
.ranking-name{{font-size:.82rem;font-weight:600}}
.ranking-meta{{font-size:.72rem;color:var(--t2);display:flex;gap:6px;align-items:center}}
.ranking-score{{font-family:var(--tf);font-size:1rem;font-weight:700;color:var(--green)}}
.eco-badge{{display:inline-block;padding:2px 7px;border-radius:10px;font-size:.7rem;font-weight:600;text-transform:capitalize}}
.badge-forest{{background:rgba(16,185,129,.15);color:var(--green)}}
.badge-wetland{{background:rgba(59,130,246,.15);color:var(--blue)}}
.badge-mangrove{{background:rgba(6,182,212,.15);color:var(--cyan)}}
.badge-agricultural{{background:rgba(245,158,11,.15);color:var(--orange)}}
.badge-urban_green{{background:rgba(236,72,153,.15);color:var(--pink)}}
.no-sel{{color:var(--t2);text-align:center;font-size:.85rem;padding:30px 0;border:1px dashed var(--border);border-radius:12px}}
.footer{{margin-top:40px;text-align:center;color:var(--t2);font-size:.75rem;border-top:1px solid var(--border);padding-top:15px}}
.conf-table{{width:100%;border-collapse:collapse;font-size:.75rem;margin-top:10px}}
.conf-table th,.conf-table td{{padding:6px 8px;text-align:center;border:1px solid var(--border)}}
.conf-table th{{background:rgba(255,255,255,.04);color:var(--t2);font-weight:600}}
</style>
</head>
<body>
<div class="container">
<header>
<div class="brand">
<h1>EcoLens Retrieval Dashboard</h1>
<p>Multi-Model Ecosystem Similarity Retrieval Framework | Prithvi-100M vs ViT-Base vs ResNet-50</p>
</div>
<div class="quick-stats">
<div class="stat-card"><div class="label">Models</div><div class="value" id="stat-models">{len(all_model_data)}</div></div>
<div class="stat-card"><div class="label">Patches</div><div class="value" id="stat-patches">--</div></div>
<div class="stat-card glow"><div class="label">mAP (Cosine)</div><div class="value" id="stat-map">--</div></div>
<div class="stat-card"><div class="label">MRR (Cosine)</div><div class="value" id="stat-mrr">--</div></div>
</div>
</header>

<div class="model-select-bar">
<label>Foundation Model:</label>
<select id="model-select" class="model-select"></select>
<span class="model-desc" id="model-desc"></span>
</div>

<div class="panel" id="cross-model-panel" style="display:none">
<h2 class="panel-title">Cross-Model mAP Comparison (Cosine Similarity)</h2>
<div class="chart-box"><canvas id="cross-model-chart"></canvas></div>
</div>

<div class="metrics-grid">
<div class="panel">
<h2 class="panel-title">Per-Category Retrieval Performance</h2>
<div class="chart-box"><canvas id="cat-chart"></canvas></div>
</div>
<div class="panel">
<h2 class="panel-title">Retrieval Confusion Matrix</h2>
<div id="conf-matrix-container"></div>
</div>
</div>

<div class="grid-main">
<div class="panel">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;flex-wrap:wrap;gap:10px">
<h2 class="panel-title" style="margin-bottom:0">Similarity Search</h2>
<div class="method-tabs" id="method-tabs">
<button class="method-tab active" data-method="cosine">Cosine</button>
<button class="method-tab" data-method="euclidean">Euclidean</button>
<button class="method-tab" data-method="knn">kNN</button>
</div>
</div>
<div class="chart-box" style="height:450px"><canvas id="scatter-chart"></canvas></div>
</div>
<div class="panel">
<h2 class="panel-title">Ecosystem Search Database</h2>
<select id="patch-select" class="search-select">
<option value="" disabled selected>Select a patch to search...</option>
</select>
<div id="no-sel" class="no-sel">Select a patch or click a point on the scatter plot.</div>
<div id="detail-card" class="detail-card">
<div class="detail-title" id="d-name">Name</div>
<div class="meta-grid">
<div class="meta-item"><div class="label">ID</div><div class="val" id="d-id"></div></div>
<div class="meta-item"><div class="label">Category</div><div class="val" id="d-cat"></div></div>
<div class="meta-item"><div class="label">Climate</div><div class="val" id="d-climate"></div></div>
<div class="meta-item"><div class="label">Protected</div><div class="val" id="d-prot"></div></div>
<div class="meta-item" style="grid-column:span 2"><div class="label">Coordinates</div><div class="val" id="d-coords"></div></div>
</div>
</div>
<div id="sim-results" style="display:none">
<h3 class="panel-title" style="font-size:.95rem;margin-bottom:10px">Top Similar Ecosystems</h3>
<div id="rankings"></div>
</div>
</div>
</div>
<div class="footer"><p>EcoLens - Multi-Model Ecosystem Similarity Retrieval Framework | Prithvi-100M, ViT-Base, ResNet-50</p></div>
</div>

<script>
const ALL_MODEL_DATA={json.dumps(all_model_data)};
const EVAL={json.dumps(eval_data)};
const MODEL_LABELS={json.dumps(model_labels)};
const colors={{'forest':'#10b981','wetland':'#3b82f6','mangrove':'#06b6d4','agricultural':'#f59e0b','urban_green':'#ec4899'}};
const modelColors={{'prithvi':'#8b5cf6','vit':'#3b82f6','resnet':'#f59e0b'}};
let currentModel='{default_model}';
let currentMethod='cosine';
let scatterChart=null;
let catChart=null;
let crossModelChart=null;

// Populate model selector
const modelSel=document.getElementById('model-select');
Object.keys(ALL_MODEL_DATA).forEach(mk=>{{const o=document.createElement('option');o.value=mk;o.textContent=MODEL_LABELS[mk]||mk;modelSel.appendChild(o)}});
modelSel.value=currentModel;

// Cross-model chart (if multiple models)
function buildCrossModelChart(){{
  const panel=document.getElementById('cross-model-panel');
  if(Object.keys(ALL_MODEL_DATA).length<2){{panel.style.display='none';return}}
  panel.style.display='block';
  const cats=new Set();
  Object.keys(EVAL).forEach(mk=>{{const pc=EVAL[mk]?.cosine?.per_category||{{}};Object.keys(pc).forEach(c=>cats.add(c))}});
  const sortedCats=[...cats].sort();
  const datasets=Object.keys(EVAL).map(mk=>{{
    const pc=EVAL[mk]?.cosine?.per_category||{{}};
    return{{label:MODEL_LABELS[mk]||mk,data:sortedCats.map(c=>(pc[c]?.mAP||0)),backgroundColor:(modelColors[mk]||'#8b5cf6')+'99',borderColor:modelColors[mk]||'#8b5cf6',borderWidth:1}}
  }});
  const ctx=document.getElementById('cross-model-chart').getContext('2d');
  if(crossModelChart)crossModelChart.destroy();
  crossModelChart=new Chart(ctx,{{type:'bar',data:{{labels:sortedCats.map(c=>c.charAt(0).toUpperCase()+c.slice(1)),datasets}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#94a3b8',font:{{size:11}}}}}}}},scales:{{x:{{ticks:{{color:'#94a3b8'}},grid:{{color:'rgba(255,255,255,.05)'}}}},y:{{ticks:{{color:'#94a3b8'}},grid:{{color:'rgba(255,255,255,.05)'}},title:{{display:true,text:'mAP (Cosine)',color:'#94a3b8'}}}}}}}}}});
}}
buildCrossModelChart();

function switchModel(mk){{
  currentModel=mk;
  const md=ALL_MODEL_DATA[mk];
  if(!md)return;
  const D=md.dashboard_data;
  // Update stats
  const ev=EVAL[mk]||{{}};
  document.getElementById('stat-patches').textContent=D.length;
  document.getElementById('stat-map').textContent=(ev.cosine?.overall?.mAP||0).toFixed(3);
  document.getElementById('stat-mrr').textContent=(ev.cosine?.overall?.MRR||0).toFixed(3);
  document.getElementById('model-desc').textContent=(MODEL_LABELS[mk]||mk);

  // Rebuild dropdown
  const sel=document.getElementById('patch-select');
  sel.innerHTML='<option value="" disabled selected>Select a patch to search...</option>';
  D.forEach(d=>{{const o=document.createElement('option');o.value=d.id;o.textContent=`[${{d.ecosystem.toUpperCase()}}] ${{d.name}} (${{d.id}})`;sel.appendChild(o)}});

  // Rebuild scatter chart
  const ctx=document.getElementById('scatter-chart').getContext('2d');
  if(scatterChart)scatterChart.destroy();
  const scatterData={{datasets:D.map(d=>({{label:d.id,data:[{{x:d.x,y:d.y}}],backgroundColor:colors[d.ecosystem]||'#8b5cf6',pointRadius:8,pointHoverRadius:11,borderColor:'rgba(255,255,255,.4)',borderWidth:1,meta:d}}))}}; 
  scatterChart=new Chart(ctx,{{type:'scatter',data:scatterData,options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>{{const m=c.dataset.meta;return[`${{m.id}}: ${{m.name}}`,`Ecosystem: ${{m.ecosystem.toUpperCase()}}`,`Climate: ${{m.climatic_region}}`]}}}}}}}},scales:{{x:{{grid:{{color:'rgba(255,255,255,.05)'}},ticks:{{display:false}},title:{{display:true,text:'PC1',color:'#94a3b8'}}}},y:{{grid:{{color:'rgba(255,255,255,.05)'}},ticks:{{display:false}},title:{{display:true,text:'PC2',color:'#94a3b8'}}}}}},onClick:(e,el)=>{{if(el.length){{const m=scatterChart.data.datasets[el[0].datasetIndex].meta;sel.value=m.id;doSearch(m.id)}}}}}}}});

  // Rebuild category chart
  const cats=Object.keys(ev.cosine?.per_category||{{}}).sort();
  const catCtx=document.getElementById('cat-chart').getContext('2d');
  if(catChart)catChart.destroy();
  catChart=new Chart(catCtx,{{type:'bar',data:{{labels:cats.map(c=>c.charAt(0).toUpperCase()+c.slice(1)),datasets:[{{label:'Cosine mAP',data:cats.map(c=>(ev.cosine?.per_category?.[c]?.mAP||0)),backgroundColor:'rgba(139,92,246,.6)',borderColor:'#8b5cf6',borderWidth:1}},{{label:'Euclidean mAP',data:cats.map(c=>(ev.euclidean?.per_category?.[c]?.mAP||0)),backgroundColor:'rgba(59,130,246,.6)',borderColor:'#3b82f6',borderWidth:1}},{{label:'kNN mAP',data:cats.map(c=>(ev.knn?.per_category?.[c]?.mAP||0)),backgroundColor:'rgba(6,182,212,.6)',borderColor:'#06b6d4',borderWidth:1}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#94a3b8',font:{{size:11}}}}}}}},scales:{{x:{{ticks:{{color:'#94a3b8'}},grid:{{color:'rgba(255,255,255,.05)'}}}},y:{{ticks:{{color:'#94a3b8'}},grid:{{color:'rgba(255,255,255,.05)'}},title:{{display:true,text:'mAP',color:'#94a3b8'}}}}}}}}}});

  // Rebuild confusion matrix
  const conf=ev.cosine?.confusion_matrix||{{}};
  const confCats=Object.keys(conf).sort();
  const cmContainer=document.getElementById('conf-matrix-container');
  if(confCats.length){{let h='<table class="conf-table"><tr><th>Query/Retr</th>';confCats.forEach(c=>h+=`<th>${{c.slice(0,8)}}</th>`);h+='</tr>';confCats.forEach(qc=>{{h+=`<tr><th>${{qc.slice(0,8)}}</th>`;confCats.forEach(rc=>{{const v=conf[qc]?.[rc]||0;const bg=qc===rc?`rgba(16,185,129,${{Math.min(v*1.5,.8)}})`:`rgba(236,72,153,${{Math.min(v*2,.6)}})`;h+=`<td style="background:${{bg}}">${{v.toFixed(3)}}</td>`}});h+='</tr>'}});h+='</table>';cmContainer.innerHTML=h}}else{{cmContainer.innerHTML='<p style="color:var(--t2);text-align:center">No confusion matrix data available.</p>'}}

  // Reset search
  document.getElementById('no-sel').style.display='block';
  document.getElementById('detail-card').style.display='none';
  document.getElementById('sim-results').style.display='none';
}}

// Initialize with default model
switchModel(currentModel);

// Model selector change
modelSel.addEventListener('change',e=>switchModel(e.target.value));

// Method tabs
document.querySelectorAll('.method-tab').forEach(btn=>{{btn.addEventListener('click',()=>{{document.querySelectorAll('.method-tab').forEach(b=>b.classList.remove('active'));btn.classList.add('active');currentMethod=btn.dataset.method;const qid=document.getElementById('patch-select').value;if(qid)doSearch(qid)}})}});

document.getElementById('patch-select').addEventListener('change',e=>doSearch(e.target.value));

function doSearch(qid){{
  const md=ALL_MODEL_DATA[currentModel];
  if(!md)return;
  const D=md.dashboard_data;
  const d=D.find(x=>x.id===qid);
  if(!d)return;

  document.getElementById('no-sel').style.display='none';
  document.getElementById('detail-card').style.display='block';
  document.getElementById('sim-results').style.display='block';
  document.getElementById('d-name').textContent=d.name;
  document.getElementById('d-id').textContent=d.id;
  document.getElementById('d-cat').textContent=d.ecosystem.toUpperCase();
  document.getElementById('d-climate').textContent=d.climatic_region;
  document.getElementById('d-prot').textContent=d.protected_area?'Protected':'Unprotected';
  document.getElementById('d-coords').textContent=`${{d.lon.toFixed(4)}}, ${{d.lat.toFixed(4)}}`;

  const simKey=currentMethod==='knn'?'sims_knn':currentMethod==='euclidean'?'sims_euc':'sims_cos';
  const sims=md[simKey]?.[qid]||{{}};
  const sorted=Object.keys(sims).filter(id=>id!==qid).map(id=>({{id,score:sims[id],meta:D.find(x=>x.id===id)}})).filter(x=>x.meta).sort((a,b)=>b.score-a.score);

  const rc=document.getElementById('rankings');rc.innerHTML='';
  sorted.slice(0,15).forEach(m=>{{const div=document.createElement('div');div.className='ranking-item';div.onclick=()=>{{document.getElementById('patch-select').value=m.id;doSearch(m.id)}};
  div.innerHTML=`<div><span class="ranking-name">${{m.meta.name}}</span><div class="ranking-meta"><span class="eco-badge badge-${{m.meta.ecosystem}}">${{m.meta.ecosystem}}</span><span>${{m.meta.climatic_region}}</span></div></div><span class="ranking-score">${{m.score.toFixed(4)}}</span>`;
  rc.appendChild(div)}});

  if(scatterChart){{scatterChart.data.datasets.forEach(ds=>{{if(ds.label===qid){{ds.pointRadius=14;ds.borderWidth=3;ds.borderColor='#fff'}}else{{ds.pointRadius=8;ds.borderWidth=1;ds.borderColor='rgba(255,255,255,.4)'}}}});scatterChart.update()}}
}}
</script>
</body>
</html>'''


def main():
    if not os.path.exists(METADATA_CATALOG_PATH):
        print(f"Error: Catalog not found at {METADATA_CATALOG_PATH}. Run steps 1-3 first.")
        return

    with open(METADATA_CATALOG_PATH) as f:
        catalog = json.load(f)

    # Compute data for each model that has embeddings
    all_model_data = {}
    model_labels = {}

    for model_key, model_cfg in SUPPORTED_MODELS.items():
        emb_dir = model_cfg["embeddings_dir"]
        if not os.path.isdir(emb_dir):
            continue

        result = compute_model_data(catalog, model_key)
        if result is not None:
            all_model_data[model_key] = result
            model_labels[model_key] = model_cfg["label"]
            print(f"Loaded {model_cfg['label']}: {len(result['dashboard_data'])} patches")

    if not all_model_data:
        print("Error: No model embeddings found. Run 03_extract_embeddings.py first.")
        return

    # Load evaluation data if available
    eval_path = f"{RESULTS_DIR}/evaluation_report.json"
    eval_data = {}
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_data = json.load(f)
        print("Loaded evaluation report.")
    else:
        print(f"Warning: {eval_path} not found. Dashboard will have empty metrics.")

    html = build_html(all_model_data, eval_data, model_labels)

    out_path = "retrieval_dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nRetrieval dashboard saved to: {out_path}")
    print(f"Models included: {list(model_labels.values())}")
    print("Open in a browser to explore ecosystem similarity retrieval results.")


if __name__ == "__main__":
    main()
