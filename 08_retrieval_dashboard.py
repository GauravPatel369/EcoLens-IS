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
    """Compute PCA/t-SNE coordinates + similarity matrices for a single model."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

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

    # 1. PCA projection using sklearn
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # 2. t-SNE projection using sklearn
    perp = min(30, len(valid) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca")
    X_tsne = tsne.fit_transform(X)

    # Compute similarity matrices for all three methods
    sims_cos, sims_euc, sims_knn = {}, {}, {}

    for i, id_a in enumerate(ids):
        sims_cos[id_a], sims_euc[id_a], sims_knn[id_a] = {}, {}, {}
        for j, id_b in enumerate(ids):
            cs = float(np.dot(vectors[i], vectors[j]))
            dist = float(np.linalg.norm(vectors[i] - vectors[j]))
            es = 1.0 / (1.0 + dist)
            sims_cos[id_a][id_b] = cs
            sims_euc[id_a][id_b] = es
            sims_knn[id_a][id_b] = es

    dashboard_data = []
    for i, entry in enumerate(valid):
        # Determine protected status using naming heuristic
        name_str = entry.get("name", "").lower()
        is_protected = entry.get("protected_area", False)
        if "national park" in name_str or "reserve" in name_str or "sanctuary" in name_str or "forest reserve" in name_str:
            is_protected = True

        dashboard_data.append({
            "id": entry["id"], 
            "ecosystem": entry["ecosystem"],
            "name": entry["name"], 
            "lon": entry["lon"], 
            "lat": entry["lat"],
            "protected_area": is_protected,
            "climatic_region": entry.get("climatic_region", "Unknown"),
            "x_pca": float(X_pca[i, 0]), 
            "y_pca": float(X_pca[i, 1]),
            "x_tsne": float(X_tsne[i, 0]), 
            "y_tsne": float(X_tsne[i, 1]),
            "x": float(X_pca[i, 0]), 
            "y": float(X_pca[i, 1]),
        })

    return {
        "dashboard_data": dashboard_data,
        "sims_cos": sims_cos,
        "sims_euc": sims_euc,
        "sims_knn": sims_knn,
    }


def build_html(all_model_data, eval_data, explain_data, model_labels):
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
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
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
<div> <!-- Left Column -->
  <div class="panel">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;flex-wrap:wrap;gap:10px">
      <h2 class="panel-title" style="margin-bottom:0">Similarity Search</h2>
      <div style="display:flex;gap:15px;align-items:center">
        <div class="method-tabs" style="margin-bottom:0">
          <button id="btn-pca" class="method-tab active">PCA</button>
          <button id="btn-tsne" class="method-tab">t-SNE</button>
        </div>
        <div class="method-tabs" id="method-tabs" style="margin-bottom:0">
          <button class="method-tab active" data-method="cosine">Cosine</button>
          <button class="method-tab" data-method="euclidean">Euclidean</button>
          <button class="method-tab" data-method="knn">kNN</button>
        </div>
      </div>
    </div>
    <div class="chart-box" style="height:450px"><canvas id="scatter-chart"></canvas></div>
  </div>

  <div class="panel" id="leaflet-map-panel" style="display:none">
    <h2 class="panel-title">Geographic Analog Connections & Clusters</h2>
    <div id="leaflet-map" style="height:400px;width:100%;border-radius:12px;border:1px solid var(--border);background:#0f172a"></div>
  </div>
</div> <!-- End Left Column -->

<div class="panel"> <!-- Right Column -->
  <h2 class="panel-title">Ecosystem Search Database</h2>
  <select id="patch-select" class="search-select">
    <option value="" disabled selected>Select a location...</option>
  </select>
  <div id="no-sel" class="no-sel">Select a location or click a point on the scatter plot.</div>

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

  <div id="comparison-card" class="detail-card" style="display:none;background:rgba(139,92,246,0.05);border-color:rgba(139,92,246,0.2);margin-top:15px">
    <div class="detail-title" style="color:var(--pri);font-size:1.05rem;border-bottom:1px solid rgba(139,92,246,0.1)">Ecological Comparison Report</div>
    <div id="comparison-desc" style="font-size:0.8rem;margin-bottom:12px;color:var(--t1);font-style:italic"></div>
    <div style="font-size:0.7rem;color:var(--t2);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;font-weight:600">Metric Comparison</div>
    <div id="comparison-metrics" style="display:flex;flex-direction:column;gap:6px;font-size:0.75rem"></div>

    <div style="margin-top:15px;height:240px;position:relative">
      <canvas id="radar-chart"></canvas>
    </div>

    <button id="set-query-btn" class="method-tab" style="width:100%;margin-top:12px;background:var(--pri);color:var(--t1);border:none;border-radius:6px;padding:8px;font-weight:600;box-shadow:0 0 10px var(--glow)">Set as Query Ecosystem</button>
  </div>

  <div id="sim-results" style="display:none">
    <h3 class="panel-title" style="font-size:.95rem;margin-bottom:10px">Top Similar Ecosystems</h3>
    <div id="rankings"></div>
  </div>
</div> <!-- End Right Column -->
</div>
<div class="footer"><p>EcoLens - Multi-Model Ecosystem Similarity Retrieval Framework | Prithvi-100M, ViT-Base, ResNet-50</p></div>
</div>

<script>
const ALL_MODEL_DATA={json.dumps(all_model_data)};
const EVAL={json.dumps(eval_data)};
const EXPLAIN_DATA={json.dumps(explain_data)};
const MODEL_LABELS={json.dumps(model_labels)};
const colors={{'forest':'#10b981','wetland':'#3b82f6','mangrove':'#06b6d4','agricultural':'#f59e0b','urban_green':'#ec4899'}};
const modelColors={{'prithvi':'#8b5cf6','vit':'#3b82f6','resnet':'#f59e0b'}};
let currentModel='{default_model}';
let currentMethod='cosine';
let currentProjection='pca';
let scatterChart=null;
let catChart=null;
let crossModelChart=null;

// Leaflet Map State
let map=null;
let markersLayer=null;
let mapLinesLayer=null;

function initMap() {{
  if(map)return;
  map=L.map('leaflet-map',{{zoomControl:true,attributionControl:false}}).setView([20,0],2);
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
    maxZoom:19
  }}).addTo(map);
  markersLayer=L.layerGroup().addTo(map);
  mapLinesLayer=L.layerGroup().addTo(map);
}}

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

function setProjection(proj) {{
  currentProjection=proj;
  document.getElementById('btn-pca').classList.toggle('active',proj==='pca');
  document.getElementById('btn-tsne').classList.toggle('active',proj==='tsne');
  
  if(scatterChart){{
    scatterChart.data.datasets.forEach(ds=>{{
      const m=ds.meta;
      if(proj==='pca'){{
        ds.data[0].x=m.x_pca;
        ds.data[0].y=m.y_pca;
      }}else{{
        ds.data[0].x=m.x_tsne;
        ds.data[0].y=m.y_tsne;
      }}
    }});
    scatterChart.options.scales.x.title.text=proj.toUpperCase()+' 1';
    scatterChart.options.scales.y.title.text=proj.toUpperCase()+' 2';
    scatterChart.update();
  }}
}}

function switchModel(mk){{
  currentModel=mk;
  const md=ALL_MODEL_DATA[mk];
  if(!md)return;
  const D=md.dashboard_data;
  
  // Re-populate Leaflet Map with only base locations initially
  initMap();
  markersLayer.clearLayers();
  mapLinesLayer.clearLayers();
  D.forEach(d=>{{
    if (!d.id.endsWith('_p0')) return; // Show only base locations initially
    const color=colors[d.ecosystem]||'#8b5cf6';
    const baseName=d.name.replace(' (Patch #1)', '');
    const marker=L.circleMarker([d.lat,d.lon],{{
      radius:6,
      fillColor:color,
      color:'#ffffff',
      weight:1,
      fillOpacity:0.8
    }});
    marker.bindPopup('<b>'+baseName+'</b>');
    marker.on('click',()=>{{
      document.getElementById('patch-select').value=d.id;
      doSearch(d.id);
    }});
    markersLayer.addLayer(marker);
  }});

  // Update stats (Count unique locations instead of raw sub-patches)
  const ev=EVAL[mk]||{{}};
  const uniqueLocCount = D.filter(d => d.id.endsWith('_p0')).length;
  document.getElementById('stat-patches').textContent=uniqueLocCount;
  document.getElementById('stat-map').textContent=(ev.cosine?.overall?.mAP||0).toFixed(3);
  document.getElementById('stat-mrr').textContent=(ev.cosine?.overall?.MRR||0).toFixed(3);
  document.getElementById('model-desc').textContent=(MODEL_LABELS[mk]||mk);

  // Rebuild dropdown with only base locations
  const sel=document.getElementById('patch-select');
  sel.innerHTML='<option value="" disabled selected>Select a location...</option>';
  D.forEach(d=>{{
    if (d.id.endsWith('_p0')) {{
      const baseName=d.name.replace(' (Patch #1)', '');
      const o=document.createElement('option');
      o.value=d.id;
      o.textContent=`[${{d.ecosystem.toUpperCase()}}] ${{baseName}}`;
      sel.appendChild(o);
    }}
  }});

  // Rebuild scatter chart (shows all 710 patches to display intra-class feature spread)
  const ctx=document.getElementById('scatter-chart').getContext('2d');
  if(scatterChart)scatterChart.destroy();
  const scatterData={{datasets:D.map(d=>({{label:d.id,data:[{{x:d.x_pca,y:d.y_pca}}],backgroundColor:colors[d.ecosystem]||'#8b5cf6',pointRadius:6,pointHoverRadius:9,borderColor:'rgba(255,255,255,.2)',borderWidth:1,meta:d}}))}}; 
  scatterChart=new Chart(ctx,{{type:'scatter',data:scatterData,options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>{{const m=c.dataset.meta;return[`${{m.id}}: ${{m.name}}`,`Ecosystem: ${{m.ecosystem.toUpperCase()}}`,`Climate: ${{m.climatic_region}}`]}}}}}}}},scales:{{x:{{grid:{{color:'rgba(255,255,255,.05)'}},ticks:{{display:false}},title:{{display:true,text:'PC1',color:'#94a3b8'}}}},y:{{grid:{{color:'rgba(255,255,255,.05)'}},ticks:{{display:false}},title:{{display:true,text:'PC2',color:'#94a3b8'}}}}}},onClick:(e,el)=>{{if(el.length){{const m=scatterChart.data.datasets[el[0].datasetIndex].meta;const baseId=m.id.split('_p')[0] + '_p0';sel.value=baseId;doSearch(baseId)}}}}}}}});

  // Reset projection display to PCA
  setProjection('pca');

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
  document.getElementById('leaflet-map-panel').style.display='none';
  document.getElementById('comparison-card').style.display='none';
  document.getElementById('sim-results').style.display='none';
}}

// Initialize with default model
switchModel(currentModel);

// Model selector change
modelSel.addEventListener('change',e=>switchModel(e.target.value));

// Method tabs
document.querySelectorAll('.method-tab').forEach(btn=>{{btn.addEventListener('click',()=>{{document.querySelectorAll('.method-tab').forEach(b=>b.classList.remove('active'));btn.classList.add('active');currentMethod=btn.dataset.method;const qid=document.getElementById('patch-select').value;if(qid)doSearch(qid)}})}});

// Projection Buttons click handlers
document.getElementById('btn-pca').addEventListener('click',()=>setProjection('pca'));
document.getElementById('btn-tsne').addEventListener('click',()=>setProjection('tsne'));

document.getElementById('patch-select').addEventListener('change',e=>doSearch(e.target.value));

function showComparison(qid, aid, element) {{
  const compCard=document.getElementById('comparison-card');
  const compDesc=document.getElementById('comparison-desc');
  const compMetrics=document.getElementById('comparison-metrics');
  const setQueryBtn=document.getElementById('set-query-btn');

  const qDesc=EXPLAIN_DATA.descriptors[qid];
  const aDesc=EXPLAIN_DATA.descriptors[aid];
  
  if(!qDesc || !aDesc) {{
    compCard.style.display='none';
    return;
  }}

  const expObj=(EXPLAIN_DATA.explanations[qid] && EXPLAIN_DATA.explanations[qid][aid]) || {{explanation: "No comparison details available."}};
  compDesc.textContent=expObj.explanation;

  compMetrics.innerHTML=`
    <div style="display:grid;grid-template-columns:1fr 80px 80px;border-bottom:1px solid rgba(255,255,255,0.05);padding-bottom:4px;font-weight:600">
      <span>Indicator</span>
      <span style="text-align:right;color:var(--t2)">Query</span>
      <span style="text-align:right;color:var(--t2)">Analog</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 80px 80px;padding-top:4px">
      <span>Forest Cover</span>
      <span style="text-align:right">${{qDesc.forest_cover}}%</span>
      <span style="text-align:right">${{aDesc.forest_cover}}%</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 80px 80px">
      <span>Water Cover</span>
      <span style="text-align:right">${{qDesc.water_cover}}%</span>
      <span style="text-align:right">${{aDesc.water_cover}}%</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 80px 80px">
      <span>Urban/Bare Soil</span>
      <span style="text-align:right">${{qDesc.urban_cover}}%</span>
      <span style="text-align:right">${{aDesc.urban_cover}}%</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 80px 80px">
      <span>Veg Health (NDVI)</span>
      <span style="text-align:right">${{qDesc.veg_health.toFixed(3)}}</span>
      <span style="text-align:right">${{aDesc.veg_health.toFixed(3)}}</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 80px 80px">
      <span>Altitude (m)</span>
      <span style="text-align:right">${{qDesc.elevation}}m</span>
      <span style="text-align:right">${{aDesc.elevation}}m</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 80px 80px">
      <span>Temperature (°C)</span>
      <span style="text-align:right">${{qDesc.temp}}°C</span>
      <span style="text-align:right">${{aDesc.temp}}°C</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 80px 80px">
      <span>Rainfall (mm)</span>
      <span style="text-align:right">${{qDesc.rainfall}}mm</span>
      <span style="text-align:right">${{aDesc.rainfall}}mm</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 80px 80px">
      <span>Soil Type</span>
      <span style="text-align:right;font-size:0.65rem;color:var(--t2)">${{qDesc.soil}}</span>
      <span style="text-align:right;font-size:0.65rem;color:var(--t2)">${{aDesc.soil}}</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 80px 80px;border-bottom:1px solid rgba(255,255,255,0.05);padding-bottom:4px">
      <span>Ecoregion</span>
      <span style="text-align:right;font-size:0.65rem;color:var(--t2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${{qDesc.ecoregion}}">${{qDesc.ecoregion}}</span>
      <span style="text-align:right;font-size:0.65rem;color:var(--t2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${{aDesc.ecoregion}}">${{aDesc.ecoregion}}</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 80px 80px;padding-top:4px">
      <span>Protected Area</span>
      <span style="text-align:right;color:${{qDesc.protected_area ? 'var(--green)' : 'var(--orange)'}}">${{qDesc.protected_area ? 'Yes' : 'No'}}</span>
      <span style="text-align:right;color:${{aDesc.protected_area ? 'var(--green)' : 'var(--orange)'}}">${{aDesc.protected_area ? 'Yes' : 'No'}}</span>
    </div>
  `;

  // Draw Radar Comparison Chart
  const radarCtx = document.getElementById('radar-chart').getContext('2d');
  if (window.radarChartInst) {{
    window.radarChartInst.destroy();
  }}
  
  const qVals = [
    qDesc.forest_cover || 0,
    qDesc.water_cover || 0,
    qDesc.urban_cover || 0,
    (qDesc.veg_health || 0) * 100,
    (qDesc.elevation || 0) / 30.0,
    (qDesc.temp || 0) * 2.8
  ];
  
  const aVals = [
    aDesc.forest_cover || 0,
    aDesc.water_cover || 0,
    aDesc.urban_cover || 0,
    (aDesc.veg_health || 0) * 100,
    (aDesc.elevation || 0) / 30.0,
    (aDesc.temp || 0) * 2.8
  ];

  window.radarChartInst = new Chart(radarCtx, {{
    type: 'radar',
    data: {{
      labels: ['Forest Cover %', 'Water Cover %', 'Urban/Bare %', 'Veg Health (NDVI x100)', 'Elevation (scaled)', 'Temperature (scaled)'],
      datasets: [
        {{
          label: 'Query (' + qid + ')',
          data: qVals,
          backgroundColor: 'rgba(139, 92, 246, 0.2)',
          borderColor: 'rgba(139, 92, 246, 0.8)',
          pointBackgroundColor: 'rgba(139, 92, 246, 1)',
          borderWidth: 2
        }},
        {{
          label: 'Analog (' + aid + ')',
          data: aVals,
          backgroundColor: 'rgba(16, 185, 129, 0.2)',
          borderColor: 'rgba(16, 185, 129, 0.8)',
          pointBackgroundColor: 'rgba(16, 185, 129, 1)',
          borderWidth: 2
        }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 9 }} }} }}
      }},
      scales: {{
        r: {{
          grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
          angleLines: {{ color: 'rgba(255, 255, 255, 0.05)' }},
          pointLabels: {{ color: '#94a3b8', font: {{ size: 8 }} }},
          ticks: {{ display: false }},
          suggestedMin: 0,
          suggestedMax: 100
        }}
      }}
    }}
  }});

  // Highlight selection in list
  document.querySelectorAll('.ranking-item').forEach(el => {{
    el.style.borderColor = 'var(--border)';
    el.style.background = 'rgba(255,255,255,.02)';
  }});
  element.style.borderColor = 'var(--pri)';
  element.style.background = 'rgba(139,92,246,.08)';

  compCard.style.display='block';

  setQueryBtn.onclick=()=>{{
    document.getElementById('patch-select').value=aid;
    doSearch(aid);
  }};
}}

function doSearch(qid){{
  const md=ALL_MODEL_DATA[currentModel];
  if(!md)return;
  const D=md.dashboard_data;
  const d=D.find(x=>x.id===qid);
  if(!d)return;

  const qBase = qid.split('_p')[0];
  const baseName = d.name.replace(' (Patch #1)', '');

  document.getElementById('no-sel').style.display='none';
  document.getElementById('detail-card').style.display='block';
  document.getElementById('comparison-card').style.display='none'; // Reset comparison
  document.getElementById('sim-results').style.display='block';
  document.getElementById('d-name').textContent=baseName;
  document.getElementById('d-id').textContent=d.id;
  document.getElementById('d-cat').textContent=d.ecosystem.toUpperCase();
  document.getElementById('d-climate').textContent=d.climatic_region;
  document.getElementById('d-prot').textContent=d.protected_area?'Protected':'Unprotected';
  document.getElementById('d-coords').textContent=`${{d.lon.toFixed(4)}}, ${{d.lat.toFixed(4)}}`;

  const simKey=currentMethod==='knn'?'sims_knn':currentMethod==='euclidean'?'sims_euc':'sims_cos';
  const sims=md[simKey]?.[qid]||{{}};

  // 1. Filter out patches from the query location, and group other patches by base location ID
  const grouped = {{}};
  Object.keys(sims).forEach(id => {{
    const cBase = id.split('_p')[0];
    if (cBase === qBase) return; // Skip sub-crops of same location
    
    const score = sims[id];
    const meta = D.find(x => x.id === id);
    if (!meta) return;
    
    if (!grouped[cBase] || score > grouped[cBase].score) {{
      grouped[cBase] = {{ id, score, meta }};
    }}
  }});

  // Sort locations by their best matching sub-patch score
  const sorted = Object.values(grouped).sort((a, b) => b.score - a.score);

  // Update Leaflet map with query flyTo and similarity line markers
  document.getElementById('leaflet-map-panel').style.display='block';
  initMap();
  if (map) {{
    map.invalidateSize();
  }}
  mapLinesLayer.clearLayers();
  
  // Fly to global view (zoom 2) to see global connections by default, allowing manual zoom in/out
  map.flyTo([d.lat, d.lon], 2, {{animate:true, duration:1.2}});
  
  // Plot all 10 sub-patches of the query location
  const qSubPatches = D.filter(x => x.id.split('_p')[0] === qBase);
  qSubPatches.forEach(sp => {{
    const isExact = sp.id === qid;
    L.circleMarker([sp.lat, sp.lon], {{
      radius: isExact ? 10 : 6,
      fillColor: colors[d.ecosystem] || '#8b5cf6',
      color: isExact ? '#ffd700' : '#ffffff',
      weight: isExact ? 3 : 1,
      fillOpacity: isExact ? 0.9 : 0.6
    }}).addTo(mapLinesLayer).bindPopup(`<b>${{baseName}}</b><br>Sub-Patch: ${{sp.id}}${{isExact ? ' (Selected)' : ''}}`);
  }});
  
  // Plot and connect top 5 closest location analogs
  sorted.slice(0, 5).forEach((m, idx) => {{
    const analogName = m.meta.name.split(' (Patch #')[0];
    L.circleMarker([m.meta.lat, m.meta.lon], {{
      radius: 8,
      fillColor: colors[m.meta.ecosystem] || '#3b82f6',
      color: '#ffffff',
      weight: 2,
      fillOpacity: 0.8
    }}).addTo(mapLinesLayer).bindPopup(`<b>Analog #${{idx+1}}: ${{analogName}}</b><br>Match Score: ${{m.score.toFixed(4)}}<br>(Via Sub-Patch ${{m.id}})`);
    
    L.polyline([[d.lat, d.lon], [m.meta.lat, m.meta.lon]], {{
      color: 'rgba(139,92,246,0.4)',
      weight: 1.5,
      dashArray: '4, 4'
    }}).addTo(mapLinesLayer);
  }});

  const rc=document.getElementById('rankings');rc.innerHTML='';
  sorted.slice(0,15).forEach(m=>{{
    const analogName = m.meta.name.split(' (Patch #')[0];
    const patchSuffix = m.id.split('_p')[1];
    const div=document.createElement('div');
    div.className='ranking-item';
    div.onclick=()=>showComparison(qid, m.id, div);
    div.innerHTML=`<div><span class="ranking-name">${{analogName}}</span><div class="ranking-meta"><span class="eco-badge badge-${{m.meta.ecosystem}}">${{m.meta.ecosystem}}</span><span>${{m.meta.climatic_region}} (p${{patchSuffix}})</span></div></div><span class="ranking-score">${{m.score.toFixed(4)}}</span>`;
    rc.appendChild(div);
  }});

  // Dynamic point highlight on scatter plot
  if(scatterChart){{
    scatterChart.data.datasets.forEach(ds=>{{
      const pid=ds.label;
      if(pid===qid){{
        ds.pointStyle='rectRot';
        ds.pointRadius=14;
        ds.borderWidth=3;
        ds.borderColor='#ffd700';
      }}else if(sorted.slice(0,5).some(item=>item.id===pid)){{
        ds.pointStyle='circle';
        ds.pointRadius=10;
        ds.borderWidth=2;
        ds.borderColor='#ffffff';
      }}else{{
        ds.pointStyle='circle';
        ds.pointRadius=6;
        ds.borderWidth=1;
        ds.borderColor='rgba(255,255,255,.2)';
      }}
    }});
    scatterChart.update();
  }}
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

    # Load explainable retrieval details
    explain_path = f"{RESULTS_DIR}/explainable_retrieval.json"
    explain_data = {}
    if os.path.exists(explain_path):
        with open(explain_path) as f:
            explain_data = json.load(f)
        print("Loaded explainable retrieval database.")
    else:
        print(f"Warning: {explain_path} not found. Dashboard will not show comparison explanations.")

    html = build_html(all_model_data, eval_data, explain_data, model_labels)

    out_path = "retrieval_dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nRetrieval dashboard saved to: {out_path}")
    print(f"Models included: {list(model_labels.values())}")
    print("Open in a browser to explore ecosystem similarity retrieval results.")


if __name__ == "__main__":
    main()
