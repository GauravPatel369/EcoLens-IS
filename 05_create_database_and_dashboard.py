"""
EcoLens Phase 1 — Step 5: Database & Interactive Visualization Dashboard

Achieves Objective 1 by:
  - Creating a searchable ecosystem embedding database interface.
  - Visualizing the 768D foundation model embeddings in a 2D projection (using PCA).
  - Computing average similarities and embedding distributions across categories.
  - Exporting a premium standalone interactive web dashboard.

Run:
    python 05_create_database_and_dashboard.py
"""

import json
import os
import numpy as np

from config import METADATA_CATALOG_PATH, EMBEDDINGS_DIR


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    if not os.path.exists(METADATA_CATALOG_PATH):
        print(f"Error: Catalog not found at {METADATA_CATALOG_PATH}. Run steps 1-3 first.")
        return

    with open(METADATA_CATALOG_PATH) as f:
        catalog = json.load(f)

    valid_entries = []
    for e in catalog:
        emb_path = e.get("prithvi_embedding") or e.get("embedding_path")
        if emb_path and os.path.exists(emb_path):
            e["_resolved_embedding_path"] = emb_path
            valid_entries.append(e)
    if len(valid_entries) < 2:
        print(f"Error: Need at least 2 patches with extracted embeddings. Found {len(valid_entries)}.")
        return

    print(f"Loading {len(valid_entries)} embeddings for PCA projection...")

    # Load vectors
    ids = []
    vectors = []
    for entry in valid_entries:
        vec = np.load(entry["_resolved_embedding_path"])
        ids.append(entry["id"])
        vectors.append(vec)

    X = np.stack(vectors)  # Shape: (N, 768)

    # 1. 2D Principal Component Analysis (PCA) projection using raw NumPy
    # Center the data
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    # Compute Covariance matrix
    cov = np.cov(X_centered, rowvar=False)

    # Solve Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Project to top 2 principal components
    X_2d = X_centered @ sorted_eigenvectors[:, :2]

    # 2. Compute similarity matrix
    sims = {}
    for i, id_a in enumerate(ids):
        sims[id_a] = {}
        for j, id_b in enumerate(ids):
            sims[id_a][id_b] = cosine_similarity(vectors[i], vectors[j])

    # 3. Calculate category distribution statistics
    same_eco_sims = []
    cross_eco_sims = []
    for i, entry_a in enumerate(valid_entries):
        for j, entry_b in enumerate(valid_entries):
            if i >= j:
                continue
            sim = sims[entry_a["id"]][entry_b["id"]]
            if entry_a["ecosystem"] == entry_b["ecosystem"]:
                same_eco_sims.append(sim)
            else:
                cross_eco_sims.append(sim)

    same_mean = np.mean(same_eco_sims) if same_eco_sims else 0.0
    cross_mean = np.mean(cross_eco_sims) if cross_eco_sims else 0.0
    gap = same_mean - cross_mean

    print("\nEmbedding Distribution Statistics:")
    print(f"  Same-ecosystem similarity mean: {same_mean:.4f}")
    print(f"  Cross-ecosystem similarity mean: {cross_mean:.4f}")
    print(f"  Ecosystem Separation Gap: {gap:.4f}")

    # Build dashboard datasets
    dashboard_data = []
    for i, entry in enumerate(valid_entries):
        dashboard_data.append({
            "id": entry["id"],
            "ecosystem": entry["ecosystem"],
            "name": entry["name"],
            "lon": entry["lon"],
            "lat": entry["lat"],
            "protected_area": entry.get("protected_area", False),
            "climatic_region": entry.get("climatic_region", "Unknown"),
            "x": float(X_2d[i, 0]),
            "y": float(X_2d[i, 1]),
            "similarities": sims[entry["id"]]
        })

    # 4. Generate stand-alone premium HTML dashboard file
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EcoLens: Geospatial Ecosystem Embedding Explorer</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <!-- ChartJS via CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --panel-bg: rgba(20, 27, 45, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --primary: #8b5cf6;
            --primary-glow: rgba(139, 92, 246, 0.3);
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-orange: #f59e0b;
            --accent-pink: #ec4899;
            --accent-cyan: #06b6d4;
            --font-family: 'Plus Jakarta Sans', sans-serif;
            --title-font: 'Outfit', sans-serif;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.05) 0%, transparent 40%);
            color: var(--text-primary);
            font-family: var(--font-family);
            line-height: 1.6;
            overflow-x: hidden;
            padding: 30px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
        }}

        .brand h1 {{
            font-family: var(--title-font);
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a78bfa 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }}

        .brand p {{
            color: var(--text-secondary);
            font-size: 0.95rem;
            font-weight: 300;
        }}

        .quick-stats {{
            display: flex;
            gap: 20px;
        }}

        .stat-card {{
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            backdrop-filter: blur(16px);
            border-radius: 12px;
            padding: 12px 20px;
            text-align: center;
            min-width: 140px;
        }}

        .stat-card .label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }}

        .stat-card .value {{
            font-family: var(--title-font);
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .stat-card.glow .value {{
            color: var(--accent-green);
        }}

        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 30px;
            align-items: start;
        }}

        @media (max-width: 1024px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .panel {{
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            backdrop-filter: blur(16px);
            border-radius: 16px;
            padding: 24px;
            height: 100%;
        }}

        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}

        .panel-title {{
            font-family: var(--title-font);
            font-size: 1.3rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        /* Plot Container */
        .chart-container {{
            position: relative;
            width: 100%;
            height: 500px;
            background: rgba(10, 14, 23, 0.4);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.03);
            padding: 15px;
        }}

        /* Filter Controls */
        .filters {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }}

        .filter-btn {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 6px 14px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s ease;
        }}

        .filter-btn:hover {{
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-primary);
        }}

        .filter-btn.active {{
            background: var(--primary);
            border-color: var(--primary);
            color: var(--text-primary);
            box-shadow: 0 0 12px var(--primary-glow);
        }}

        /* Side Panel: Search / Details */
        .search-select {{
            width: 100%;
            background: #141b2d;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 12px;
            border-radius: 8px;
            font-family: var(--font-family);
            font-size: 0.95rem;
            margin-bottom: 20px;
            cursor: pointer;
            outline: none;
            transition: border-color 0.2s ease;
        }}

        .search-select:focus {{
            border-color: var(--primary);
        }}

        .detail-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            display: none; /* Shown dynamically */
        }}

        .detail-title {{
            font-family: var(--title-font);
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 10px;
        }}

        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            font-size: 0.85rem;
        }}

        .meta-item .label {{
            color: var(--text-secondary);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }}

        .meta-item .val {{
            font-weight: 600;
            color: var(--text-primary);
        }}

        .ecosystem-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: capitalize;
        }}

        /* Badge color maps */
        .badge-forest {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-green); }}
        .badge-wetland {{ background: rgba(59, 130, 246, 0.15); color: var(--accent-blue); }}
        .badge-mangrove {{ background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); }}
        .badge-agricultural {{ background: rgba(245, 158, 11, 0.15); color: var(--accent-orange); }}
        .badge-urban_green {{ background: rgba(236, 72, 153, 0.15); color: var(--accent-pink); }}

        /* Similarity Rankings */
        .rankings-list {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .ranking-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 14px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            transition: transform 0.2s ease, background 0.2s ease;
            cursor: pointer;
        }}

        .ranking-item:hover {{
            transform: translateX(4px);
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(139, 92, 246, 0.3);
        }}

        .ranking-item.selected-item {{
            border-color: var(--primary);
            background: rgba(139, 92, 246, 0.06);
        }}

        .ranking-info {{
            display: flex;
            flex-direction: column;
        }}

        .ranking-name {{
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .ranking-meta {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            display: flex;
            gap: 8px;
            align-items: center;
        }}

        .ranking-score {{
            font-family: var(--title-font);
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--accent-green);
        }}

        .no-selection {{
            color: var(--text-secondary);
            text-align: center;
            font-size: 0.9rem;
            padding: 40px 0;
            border: 1px dashed var(--border-color);
            border-radius: 12px;
        }}
        
        .footer {{
            margin-top: 50px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.8rem;
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="brand">
                <h1>EcoLens Embedding Explorer</h1>
                <p>Interactive Searchable Ecosystem Database & Visualization using Prithvi-100M latent spaces</p>
            </div>
            <div class="quick-stats">
                <div class="stat-card">
                    <div class="label">Total Patches</div>
                    <div class="value">{len(valid_entries)}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Embedding Dim</div>
                    <div class="value">768</div>
                </div>
                <div class="stat-card glow">
                    <div class="label">Ecosystem Separation</div>
                    <div class="value">+{gap:.3f}</div>
                </div>
            </div>
        </header>

        <div class="dashboard-grid">
            <!-- Left Panel: 2D Projected Map -->
            <div class="panel">
                <div class="panel-header">
                    <h2 class="panel-title">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s-8-4.5-8-11.8A8 8 0 0 1 12 2a8 8 0 0 1 8 8.2c0 7.3-8 11.8-8 11.8z"/><circle cx="12" cy="10" r="3"/></svg>
                        2D PCA Latent Projection Space
                    </h2>
                    <div class="filters" id="category-filters">
                        <button class="filter-btn active" data-cat="all">All</button>
                        <button class="filter-btn" data-cat="forest">Forest</button>
                        <button class="filter-btn" data-cat="wetland">Wetland</button>
                        <button class="filter-btn" data-cat="mangrove">Mangrove</button>
                        <button class="filter-btn" data-cat="agricultural">Agri</button>
                        <button class="filter-btn" data-cat="urban_green">Urban Green</button>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="embeddings-chart"></canvas>
                </div>
            </div>

            <!-- Right Panel: Database Search & Cosine Similarities -->
            <div class="panel">
                <h2 class="panel-title" style="margin-bottom: 20px;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                    Ecosystem Search Database
                </h2>
                
                <select id="patch-select" class="search-select">
                    <option value="" disabled selected>Select an ecosystem patch to search...</option>
                </select>

                <!-- Search Status Placeholder -->
                <div id="no-selection-placeholder" class="no-selection">
                    Select a patch from the dropdown or click a point on the scatter plot map to load its metadata and perform a similarity search query.
                </div>

                <!-- Match Details -->
                <div id="detail-card" class="detail-card">
                    <div class="detail-title" id="detail-name">Patch Name</div>
                    <div class="meta-grid">
                        <div class="meta-item">
                            <div class="label">ID</div>
                            <div class="val" id="detail-id">forest_001</div>
                        </div>
                        <div class="meta-item">
                            <div class="label">Category</div>
                            <div class="val" id="detail-category">Forest</div>
                        </div>
                        <div class="meta-item">
                            <div class="label">Climatic Region</div>
                            <div class="val" id="detail-climate">Tropical</div>
                        </div>
                        <div class="meta-item">
                            <div class="label">Protected Status</div>
                            <div class="val" id="detail-protected">Protected</div>
                        </div>
                        <div class="meta-item" style="grid-column: span 2;">
                            <div class="label">Location Coords</div>
                            <div class="val" id="detail-coords">0.0, 0.0</div>
                        </div>
                    </div>
                </div>

                <!-- Similar Items Container -->
                <div id="similarity-results" style="display: none;">
                    <h3 class="panel-title" style="font-size: 1rem; margin-bottom: 12px;">Top Ecological Analogs (Similarity Search)</h3>
                    <div class="rankings-list" id="rankings-container">
                        <!-- Filled dynamically -->
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>EcoLens Phase 1 — Technical Proof of Concept Dashboard. Powered by IBM/NASA Prithvi-100M temporal Vision Transformer.</p>
        </div>
    </div>

    <script>
        // Embed the computed datasets from python script
        const rawDataset = {json.dumps(dashboard_data)};
        
        // Setup dropdown options
        const patchSelect = document.getElementById('patch-select');
        rawDataset.forEach(item => {{
            const opt = document.createElement('option');
            opt.value = item.id;
            opt.textContent = `[${{item.ecosystem.toUpperCase()}}] ${{item.name}} (${{item.id}})`;
            patchSelect.appendChild(opt);
        }});

        // Color maps for charts
        const colors = {{
            'forest': '#10b981',      // Emerald Green
            'wetland': '#3b82f6',     // Blue
            'mangrove': '#06b6d4',    // Cyan
            'agricultural': '#f59e0b',// Amber/Orange
            'urban_green': '#ec4899'  // Pink
        }};

        // Render Scatter Plot
        const ctx = document.getElementById('embeddings-chart').getContext('2d');
        
        const chartData = {{
            datasets: rawDataset.map(item => ({{
                label: item.id,
                data: [{{ x: item.x, y: item.y }}],
                backgroundColor: colors[item.ecosystem] || '#8b5cf6',
                pointRadius: 9,
                pointHoverRadius: 12,
                borderColor: 'rgba(255, 255, 255, 0.4)',
                borderWidth: 1,
                metadata: item
            }}))
        }};

        const chartConfig = {{
            type: 'scatter',
            data: chartData,
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const meta = context.dataset.metadata;
                                return [
                                    `ID: ${{meta.id}}`,
                                    `Name: ${{meta.name}}`,
                                    `Ecosystem: ${{meta.ecosystem.toUpperCase()}}`,
                                    `Region: ${{meta.climatic_region}}`,
                                    `Protected: ${{meta.protected_area ? 'Yes' : 'No'}}`
                                ];
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ display: false }},
                        title: {{ display: true, text: 'Principal Component 1', color: '#94a3b8' }}
                    }},
                    y: {{
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ display: false }},
                        title: {{ display: true, text: 'Principal Component 2', color: '#94a3b8' }}
                    }}
                }},
                onClick: (event, elements) => {{
                    if (elements.length > 0) {{
                        const index = elements[0].datasetIndex;
                        const meta = chartConfig.data.datasets[index].metadata;
                        patchSelect.value = meta.id;
                        triggerSearch(meta.id);
                    }}
                }}
            }}
        }};

        const chart = new Chart(ctx, chartConfig);

        // Filter button click logic
        const filterButtons = document.querySelectorAll('#category-filters .filter-btn');
        filterButtons.forEach(btn => {{
            btn.addEventListener('click', () => {{
                // Toggle active class
                filterButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const selectedCat = btn.getAttribute('data-cat');
                
                // Show/hide datasets on the plot
                chart.data.datasets.forEach(dataset => {{
                    if (selectedCat === 'all' || dataset.metadata.ecosystem === selectedCat) {{
                        dataset.hidden = false;
                    }} else {{
                        dataset.hidden = true;
                    }}
                }});
                chart.update();
            }});
        }});

        // Trigger similarity search logic
        patchSelect.addEventListener('change', (e) => {{
            triggerSearch(e.target.value);
        }});

        function triggerSearch(queryId) {{
            const selected = rawDataset.find(item => item.id === queryId);
            if (!selected) return;

            // Update details card UI
            document.getElementById('no-selection-placeholder').style.display = 'none';
            document.getElementById('detail-card').style.display = 'block';
            document.getElementById('similarity-results').style.display = 'block';

            document.getElementById('detail-name').textContent = selected.name;
            document.getElementById('detail-id').textContent = selected.id;
            document.getElementById('detail-category').textContent = selected.ecosystem.toUpperCase();
            document.getElementById('detail-climate').textContent = selected.climatic_region;
            document.getElementById('detail-protected').textContent = selected.protected_area ? 'Protected Area' : 'Unprotected Area';
            document.getElementById('detail-coords').textContent = `Lon: ${{selected.lon.toFixed(4)}}, Lat: ${{selected.lat.toFixed(4)}}`;

            // Change class of details card to match ecosystem badge colors
            const ecosystem = selected.ecosystem;
            const detailCard = document.getElementById('detail-card');
            detailCard.className = `detail-card`;
            
            // Build Similarity search query list
            const sims = selected.similarities;
            
            // Sort matches based on similarity descending, excluding query itself
            const sortedMatches = Object.keys(sims)
                .map(id => ({{
                    id: id,
                    score: sims[id],
                    meta: rawDataset.find(item => item.id === id)
                }}))
                .filter(item => item.id !== queryId)
                .sort((a, b) => b.score - a.score);

            const rankingsContainer = document.getElementById('rankings-container');
            rankingsContainer.innerHTML = '';

            sortedMatches.forEach(match => {{
                const itemDiv = document.createElement('div');
                itemDiv.className = 'ranking-item';
                itemDiv.onclick = () => {{
                    patchSelect.value = match.id;
                    triggerSearch(match.id);
                }};

                itemDiv.innerHTML = `
                    <div class="ranking-info">
                        <span class="ranking-name">${{match.meta.name}} (${{match.id}})</span>
                        <div class="ranking-meta">
                            <span class="ecosystem-badge badge-${{match.meta.ecosystem}}">${{match.meta.ecosystem.toUpperCase()}}</span>
                            <span>${{match.meta.climatic_region}}</span>
                            <span>${{match.meta.protected_area ? 'Protected' : 'Unprotected'}}</span>
                        </div>
                    </div>
                    <span class="ranking-score">${{match.score.toFixed(4)}}</span>
                `;
                rankingsContainer.appendChild(itemDiv);
            }});

            // Highlight queried point on the scatter plot
            chart.data.datasets.forEach(dataset => {{
                if (dataset.label === queryId) {{
                    dataset.pointRadius = 15;
                    dataset.borderWidth = 3;
                    dataset.borderColor = '#ffffff';
                }} else {{
                    dataset.pointRadius = 9;
                    dataset.borderWidth = 1;
                    dataset.borderColor = 'rgba(255, 255, 255, 0.4)';
                }}
            }});
            chart.update();
        }}
    </script>
</body>
</html>
"""

    output_html_path = "embedding_dashboard.html"
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nInteractive embedding database explorer dashboard saved to: {output_html_path}")
    print("Open this file in any web browser to view, filter, and search ecosystem embeddings.")


if __name__ == "__main__":
    main()
