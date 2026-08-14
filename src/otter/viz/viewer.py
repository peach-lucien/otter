"""Self-contained 3D HTML viewer for a cross-species coupling π.

Public API
----------
    build_viewer_data(pi, mouse_ad, human_ad, *, top_k=30, pi_label="custom")
        Build the embedded JSON payload (per-node records + top-K partners +
        entropies + col_mass).

    build_viewer_html(payload)
        Render a self-contained HTML+JS document that loads the embedded JSON
        and renders two side-by-side Plotly 3D scatters with click-to-highlight
        cross-species partners.

    write_viewer(pi, mouse_ad, human_ad, *, output_dir, top_k=30, pi_label=None)
        Convenience: build payload + HTML and save both as
        ``output_dir/viewer_data.json`` and ``output_dir/index.html``.

Architecture:
  - Two separate Plotly divs (no subplots), each rendering independently
  - Three traces per side: base (static, network-coloured), overlay (partners),
    selected (1 point in orange); clicks restyle the overlay only
  - Click handler uses curveNumber + pointNumber for scatter3d events
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from otter.data.anchors import get_anchor_index
from otter.data.networks import NETWORKS, assign_networks


# ---------------------------------------------------------------------------
# Top-K partners + entropy helpers
# ---------------------------------------------------------------------------
def topk_per_row(pi: np.ndarray, k: int) -> list[list[list[float]]]:
    """For each row, return top-k (col_idx, value) pairs sorted by value desc.
    Values are normalised by row sum (so they're in [0, 1])."""
    n_rows = pi.shape[0]
    out: list[list[list[float]]] = []
    row_sum = pi.sum(axis=1).clip(min=1e-12)
    for i in range(n_rows):
        row = pi[i] / row_sum[i]
        top = np.argpartition(-row, min(k, len(row) - 1))[:k]
        top = top[np.argsort(-row[top])]
        out.append([[int(j), float(round(row[j], 4))] for j in top])
    return out


def topk_per_col(pi: np.ndarray, k: int) -> list[list[list[float]]]:
    """Mirror for columns: top-k mouse rows for each human column."""
    n_cols = pi.shape[1]
    out: list[list[list[float]]] = []
    col_sum = pi.sum(axis=0).clip(min=1e-12)
    for j in range(n_cols):
        col = pi[:, j] / col_sum[j]
        top = np.argpartition(-col, min(k, len(col) - 1))[:k]
        top = top[np.argsort(-col[top])]
        out.append([[int(i), float(round(col[i], 4))] for i in top])
    return out


def row_entropy(pi: np.ndarray) -> np.ndarray:
    """Per-row Shannon entropy in nats. Higher = softer / more uncertain mapping."""
    row_sum = pi.sum(axis=1, keepdims=True).clip(min=1e-12)
    p = pi / row_sum
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def col_entropy(pi: np.ndarray) -> np.ndarray:
    """Per-column Shannon entropy in nats. Higher = softer / more uncertain mapping."""
    col_sum = pi.sum(axis=0, keepdims=True).clip(min=1e-12)
    p = pi / col_sum
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=0)


# ---------------------------------------------------------------------------
# Per-node record builders
# ---------------------------------------------------------------------------
def _build_node_records(ad, idx_anchor, top_partners, *, entropies=None) -> dict:
    """Build per-node records for one species."""
    var = ad.var
    nets_int = assign_networks(var, idx_anchor)
    nets_str = [NETWORKS[i] for i in nets_int]
    out = {
        "n_nodes":   int(len(var)),
        "ids":       var.index.astype(int).tolist(),
        "x":         var["x"].astype(float).tolist(),
        "y":         var["y"].astype(float).tolist(),
        "z":         var["z"].astype(float).tolist(),
        "regions":   var["region"].astype(str).tolist(),
        "networks":  nets_str,
        "is_anchor": var["garin_anchor"].astype(bool).tolist(),
        "pair_id":   var["pairid"].astype(int).tolist(),
        "anchor_pair_id": var["anchor_pair_id"].fillna(0).astype(int).tolist(),
        "top_partners": top_partners,
    }
    if entropies is not None:
        out["entropy"] = [float(round(e, 4)) for e in entropies]
    return out


def build_viewer_data(
    pi: np.ndarray,
    mouse_ad,
    human_ad,
    *,
    top_k: int = 30,
    pi_label: str = "custom",
    pi_source: str = "(in-memory)",
) -> dict:
    """Build the embedded JSON payload for the viewer.

    Parameters
    ----------
    pi : (n_m, n_h) ndarray
        The cross-species coupling.
    mouse_ad, human_ad : AnnData
    top_k : int, default 30
        How many top partners to embed per node.
    pi_label : str
        Short label shown in the viewer header (e.g. "fc_plus_SC").
    pi_source : str
        Origin tag for the sidecar (e.g. "pi_fc_plus_SC.npy").

    Returns
    -------
    dict ready to json.dumps and embed in build_viewer_html().
    """
    pi = np.asarray(pi, dtype=np.float64)
    if pi.ndim != 2:
        raise ValueError(f"pi must be 2-D, got shape {pi.shape}")

    idx_m = get_anchor_index(mouse_ad.var)
    idx_h = get_anchor_index(human_ad.var)

    top_h_per_m = topk_per_row(pi, top_k)
    top_m_per_h = topk_per_col(pi, top_k)
    ent_m = row_entropy(pi)
    ent_h = col_entropy(pi)

    mouse_data = _build_node_records(mouse_ad, idx_m, top_h_per_m, entropies=ent_m)
    human_data = _build_node_records(human_ad, idx_h, top_m_per_h, entropies=ent_h)
    human_data["col_mass"] = [float(round(v, 6)) for v in pi.sum(axis=0).tolist()]

    return {
        "version":       1,
        "pi_source":     pi_source,
        "pi_label":      pi_label,
        "n_mouse_nodes": int(pi.shape[0]),
        "n_human_nodes": int(pi.shape[1]),
        "top_k":         int(top_k),
        "networks":      list(NETWORKS),
        "mouse":         mouse_data,
        "human":         human_data,
    }


def write_viewer(
    pi: np.ndarray,
    mouse_ad,
    human_ad,
    *,
    output_dir: str | Path,
    top_k: int = 30,
    pi_label: str = "custom",
    pi_source: str = "(in-memory)",
) -> tuple[Path, Path]:
    """Convenience: build payload + HTML and save both files.

    Returns ``(json_path, html_path)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_viewer_data(pi, mouse_ad, human_ad,
                                 top_k=top_k, pi_label=pi_label, pi_source=pi_source)
    json_path = output_dir / "viewer_data.json"
    json_path.write_text(json.dumps(payload, separators=(",", ":")))
    html_path = output_dir / "index.html"
    html_path.write_text(build_viewer_html(payload))
    return json_path, html_path


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
def build_viewer_html(payload: dict) -> str:
    """Self-contained HTML+JS viewer with embedded data."""
    embedded = json.dumps(payload, separators=(",", ":"))
    return _HTML_TEMPLATE.replace("{{EMBEDDED_DATA}}", embedded)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Cross-species brain map viewer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          margin: 0; padding: 12px; background: #1e1e1e; color: #ddd; }
  h1   { margin: 0 0 6px; font-size: 18px; color: #fff; }
  .meta{ color: #aaa; font-size: 12px; margin-bottom: 8px; }
  .plotrow { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .plotbox { height: 720px; background: #111; border: 1px solid #333; border-radius: 4px; }
  #info{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
  .panel{ background: #2a2a2a; border: 1px solid #444; border-radius: 4px; padding: 10px;
          font-size: 13px; min-height: 90px; color: #ddd; }
  .panel h2 { margin: 0 0 6px; font-size: 14px; color: #fff; }
  .matches  { margin: 0; padding: 0; list-style: none; }
  .matches li { font-family: ui-monospace, "SF Mono", Menlo, monospace;
                 font-size: 12px; padding: 2px 0; }
  .controls { background: #2a2a2a; border: 1px solid #444; border-radius: 4px;
                padding: 10px; margin-bottom: 8px; display: flex; gap: 14px;
                align-items: center; flex-wrap: wrap; font-size: 13px; }
  select, input, button { padding: 4px 6px; font-size: 13px; background: #1a1a1a;
                color: #ddd; border: 1px solid #555; border-radius: 3px; }
  .legend { margin-left: auto; color: #888; font-size: 12px; }
  .anchor-mark { color: #ff8855; font-weight: 600; }
  .mode-group { display: inline-flex; border: 1px solid #555; border-radius: 3px; overflow: hidden; }
  .mode-group button { border: none; border-right: 1px solid #555; border-radius: 0;
                background: #1a1a1a; padding: 5px 10px; cursor: pointer; }
  .mode-group button:last-child { border-right: none; }
  .mode-group button.active { background: #4a7fc4; color: #fff; font-weight: 600; }
  kbd { background: #444; padding: 1px 4px; border-radius: 3px;
         font-family: ui-monospace, monospace; font-size: 11px; color: #fff; }
  code { color: #6cf; }
</style>
</head>
<body>
<h1>Cross-species brain region map, interactive viewer</h1>
<div class="meta">
  Production π = <code id="pilabel"></code> · 1864 mouse nodes (CCFv3) ↔ 2094 human nodes (MNI152) ·
  larger dot = Garin anchor · <span style="color:#FFAA00">orange</span> = currently selected ·
  <a href="#" id="toggleHelp" style="color:#6cf">show/hide quick guide ▾</a>
</div>

<div id="helpBox" style="background:#222a3a; border:1px solid #4a7fc4; border-radius:5px;
                          padding:12px 14px; margin-bottom:8px; font-size:13px; color:#ddd;">
  <h2 style="margin:0 0 6px; font-size:15px; color:#fff;">Quick guide</h2>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px;">
    <div>
      <b style="color:#fff">What this shows.</b><br>
      Two 3D point clouds: each dot is a <i>parcel</i> (small brain region) in either the mouse or human atlas.
      The model maps each mouse parcel to a probability distribution over human parcels.
      Larger dots are the 42 Garin anchors, known cross-species correspondences.
      <br><br>
      <b style="color:#fff">Cameras.</b>
      Drag = rotate · scroll = zoom · <kbd>shift</kbd>+drag = pan.
    </div>
    <div>
      <b style="color:#fff">Click a node</b> on either side: the partners on the OTHER side light up
      on a viridis scale (yellow = high probability, purple = low). Info panels below show region +
      top partners with percentages.
      <br><br>
      <b style="color:#fff">View modes</b> (top-left toggle):
      <ul style="margin:4px 0 0; padding-left:18px;">
        <li><b>By network</b>, color by the 11 functional networks.</li>
        <li><b>Mapping entropy</b>, bright = soft mapping; dark = one-hot.</li>
        <li><b>Coverage</b>, for the human side, brightness = mouse mass landed there.</li>
      </ul>
    </div>
  </div>
</div>

<div class="controls">
  <span>View:</span>
  <span class="mode-group" id="modegroup">
    <button data-mode="network" class="active">By network</button>
    <button data-mode="entropy">Mapping entropy</button>
    <button data-mode="coverage">Coverage (col mass)</button>
  </span>
  <label>Network filter:
    <select id="netfilter"><option value="">(all)</option></select>
  </label>
  <label>Hemi:
    <select id="hemifilter"><option value="">both</option><option value="L">L</option><option value="R">R</option></select>
  </label>
  <label>Search region:
    <input id="regionsearch" type="text" placeholder="visual, hippocampus, ..." size="18" />
  </label>
  <button id="reset">Reset</button>
  <span class="legend">Drag = rotate · scroll = zoom · <kbd>shift</kbd>+drag = pan</span>
</div>

<div class="plotrow">
  <div id="plot-mouse" class="plotbox"></div>
  <div id="plot-human" class="plotbox"></div>
</div>

<div id="info">
  <div class="panel" id="mouse-panel">
    <h2>🐭 Mouse, click a node</h2>
    <div>Click a node on the left side.</div>
  </div>
  <div class="panel" id="human-panel">
    <h2>🧠 Human, click a node</h2>
    <div>Click a node on the right side.</div>
  </div>
</div>

<script>
const DATA = {{EMBEDDED_DATA}};
document.getElementById('pilabel').textContent = DATA.pi_label;

const NET_COLORS = {
  auditory:        '#FF6B6B', brainstem:    '#A78BFA', frontal_dmn:    '#38BDF8',
  frontoparietal:  '#34D399', limbic:       '#FBBF24', olfactory:      '#FACC15',
  salience:        '#F472B6', sensorimotor: '#67E8F9', subcortical:    '#94A3B8',
  temporal_dmn:    '#10B981', visual:       '#FB923C'
};
const VIRIDIS = ['#fde725', '#b5de2b', '#6ece58', '#35b779', '#1f9e89',
                 '#26828e', '#31688e', '#3e4a89', '#482878', '#440154'];
function viridisColor(t) {
  const n = VIRIDIS.length;
  const i = Math.max(0, Math.min(n-1, Math.floor(t * (n-1))));
  return VIRIDIS[i];
}

let currentMode = 'network';
let selected = { species: null, idx: null };
const PLOT_IDS = { mouse: 'plot-mouse', human: 'plot-human' };

function computeBaseStyle(species, mode) {
  const d = DATA[species];
  const n = d.x.length;
  let colors, sizes, opacities;
  if (mode === 'entropy') {
    const ent = d.entropy || new Array(n).fill(0);
    let emax = 0; for (let k=0; k<n; k++) if (ent[k] > emax) emax = ent[k];
    colors = new Array(n); sizes = new Array(n); opacities = new Array(n);
    for (let k=0; k<n; k++) {
      const t = emax > 0 ? ent[k] / emax : 0;
      colors[k]    = viridisColor(1 - t);
      sizes[k]     = (d.is_anchor[k] ? 12 : 6) + 8 * t;
      opacities[k] = 0.45 + 0.5 * t;
    }
  } else if (mode === 'coverage' && species === 'human') {
    const cm = DATA.human.col_mass;
    let cmax = 0; for (let k=0; k<n; k++) if (cm[k] > cmax) cmax = cm[k];
    colors = new Array(n); sizes = new Array(n); opacities = new Array(n);
    for (let k=0; k<n; k++) {
      const v = cm[k];
      if (v < 1e-5) {
        colors[k] = 'rgba(80,80,80,0.18)'; sizes[k] = 2; opacities[k] = 0.15;
      } else {
        const t = Math.sqrt(v / cmax);
        colors[k]    = viridisColor(1 - t);
        sizes[k]     = (d.is_anchor[k] ? 12 : 6) + 10 * t;
        opacities[k] = 0.5 + 0.5 * t;
      }
    }
  } else if (mode === 'coverage' && species === 'mouse') {
    colors    = d.networks.map(nm => NET_COLORS[nm] || '#999');
    sizes     = d.is_anchor.map(a => a ? 8 : 4);
    opacities = new Array(n).fill(0.30);
  } else {
    colors    = d.networks.map(nm => NET_COLORS[nm] || '#999');
    sizes     = d.is_anchor.map(a => a ? 12 : 6);
    opacities = new Array(n).fill(0.65);
  }
  return [colors, sizes, opacities];
}

function applyFilters(species, colors, sizes, opacities) {
  const d = DATA[species];
  const net  = document.getElementById('netfilter').value;
  const hemi = document.getElementById('hemifilter').value;
  const re   = document.getElementById('regionsearch').value.trim().toLowerCase();
  if (!net && !hemi && !re) return [colors, sizes, opacities];
  for (let k = 0; k < d.x.length; k++) {
    const matchNet  = !net  || d.networks[k] === net;
    const matchHemi = !hemi ||
      (hemi === 'L' && d.regions[k].startsWith('L_')) ||
      (hemi === 'R' && d.regions[k].startsWith('R_'));
    const matchRe   = !re   || d.regions[k].toLowerCase().includes(re);
    if (!(matchNet && matchHemi && matchRe)) {
      colors[k] = 'rgba(120,120,120,0.04)'; sizes[k] = 1; opacities[k] = 0.04;
    }
  }
  return [colors, sizes, opacities];
}

function makeBaseTrace(species) {
  const d = DATA[species];
  let [c, s, o] = computeBaseStyle(species, currentMode);
  [c, s, o]     = applyFilters(species, c, s, o);
  return {
    type: 'scatter3d', mode: 'markers',
    x: d.x, y: d.y, z: d.z,
    marker: { size: s, color: c, opacity: 0.85, line: { width: 0 } },
    text: d.regions.map((r, i) => `${r} <i>(${d.networks[i]})</i>`),
    hovertemplate: '%{text}<extra></extra>',
    name: 'all-nodes', showlegend: false,
  };
}
function makeOverlayTrace() {
  return {
    type: 'scatter3d', mode: 'markers',
    x: [], y: [], z: [],
    marker: { size: [], color: [], opacity: 0.95, line: { color: '#fff', width: 1 } },
    text: [], hovertemplate: '%{text}<extra></extra>',
    name: 'partners', showlegend: false,
  };
}
function makeSelectedTrace() {
  return {
    type: 'scatter3d', mode: 'markers',
    x: [], y: [], z: [],
    marker: { size: [22], color: ['#FFAA00'], opacity: 1.0, line: { color: '#fff', width: 2 } },
    text: [], hovertemplate: '%{text}<extra></extra>',
    name: 'selected', showlegend: false,
  };
}

const layoutFor = (species) => ({
  showlegend: false,
  paper_bgcolor: '#111', plot_bgcolor: '#111',
  font: { color: '#ccc' },
  margin: { l: 0, r: 0, t: 28, b: 0 },
  scene: {
    aspectmode: 'data',
    bgcolor: '#111',
    xaxis: { title: 'x', backgroundcolor: '#1a1a1a', gridcolor: '#333' },
    yaxis: { title: 'y', backgroundcolor: '#1a1a1a', gridcolor: '#333' },
    zaxis: { title: 'z', backgroundcolor: '#1a1a1a', gridcolor: '#333' },
  },
  annotations: [{
    text: species === 'mouse' ? '🐭 Mouse, 1864 nodes (CCFv3)' : '🧠 Human, 2094 nodes (MNI152)',
    x: 0.5, y: 1.0, xref: 'paper', yref: 'paper',
    showarrow: false, font: { size: 13, color: '#fff' },
  }],
});

const PLOT_CONFIG = { responsive: true, displayModeBar: false, doubleClick: 'reset' };

Plotly.newPlot('plot-mouse',
  [makeBaseTrace('mouse'), makeOverlayTrace(), makeSelectedTrace()],
  layoutFor('mouse'), PLOT_CONFIG);
Plotly.newPlot('plot-human',
  [makeBaseTrace('human'), makeOverlayTrace(), makeSelectedTrace()],
  layoutFor('human'), PLOT_CONFIG);

function refreshBase(species) {
  let [c, s, o] = computeBaseStyle(species, currentMode);
  [c, s, o]     = applyFilters(species, c, s, o);
  Plotly.restyle(PLOT_IDS[species], { 'marker.color': [c], 'marker.size': [s] }, [0]);
}
function refreshAllBases() {
  refreshBase('mouse');
  refreshBase('human');
}

function bindClick(species) {
  const div = document.getElementById(PLOT_IDS[species]);
  div.on('plotly_click', (ev) => {
    if (!ev || !ev.points || !ev.points.length) return;
    const pt = ev.points[0];
    if (pt.curveNumber !== 0) return;
    const idx = (pt.pointNumber != null) ? pt.pointNumber
              : (pt.pointIndex  != null) ? pt.pointIndex
              : (pt.i           != null) ? pt.i : null;
    if (idx == null) {
      console.warn('Plotly click: could not extract point index', pt);
      return;
    }
    showSelection(species, idx);
  });
}
bindClick('mouse'); bindClick('human');

function showSelection(species, i) {
  selected = { species, idx: i };
  const partners = DATA[species].top_partners[i];
  const otherSp  = species === 'mouse' ? 'human' : 'mouse';
  const dSrc     = DATA[species];
  const dOther   = DATA[otherSp];

  Plotly.restyle(PLOT_IDS[species], {
    x: [[dSrc.x[i]]], y: [[dSrc.y[i]]], z: [[dSrc.z[i]]],
    text: [[`SELECTED · ${dSrc.regions[i]} (${dSrc.networks[i]})`]],
  }, [2]);
  Plotly.restyle(PLOT_IDS[otherSp], { x: [[]], y: [[]], z: [[]], text: [[]] }, [2]);

  const xs = [], ys = [], zs = [], sizes = [], colors = [], texts = [];
  const maxVal = partners.length ? partners[0][1] : 1.0;
  for (const [idx, val] of partners) {
    if (val < 0.001) continue;
    const t = Math.sqrt(val / Math.max(maxVal, 1e-9));
    xs.push(dOther.x[idx]); ys.push(dOther.y[idx]); zs.push(dOther.z[idx]);
    sizes.push(8 + 22 * t);
    colors.push(viridisColor(1 - t));
    texts.push(`${(val*100).toFixed(1)}% → ${dOther.regions[idx]} (${dOther.networks[idx]})`);
  }
  Plotly.restyle(PLOT_IDS[otherSp], {
    x: [xs], y: [ys], z: [zs],
    'marker.size': [sizes], 'marker.color': [colors],
    text: [texts],
  }, [1]);
  Plotly.restyle(PLOT_IDS[species], {
    x: [[]], y: [[]], z: [[]], 'marker.size': [[]], 'marker.color': [[]], text: [[]],
  }, [1]);

  document.getElementById(species + '-panel').innerHTML = fmtPanel(species, i);
  if (partners.length) {
    document.getElementById(otherSp + '-panel').innerHTML = fmtPanel(otherSp, partners[0][0]);
  }
}

function fmtPanel(species, i) {
  const d = DATA[species];
  const partners = d.top_partners[i];
  const other = species === 'mouse' ? DATA.human : DATA.mouse;
  const direction = species === 'mouse' ? 'top human partners' : 'top mouse partners';
  const ringInfo = d.is_anchor[i]
    ? `<span class="anchor-mark">⚓ Garin anchor pair_id=${d.anchor_pair_id[i]}</span>` : '';
  const ent = d.entropy ? d.entropy[i] : null;
  const maxEnt = d.entropy ? Math.log(other.n_nodes) : null;
  const entropyStr = ent != null
    ? `<span style="color:#888"> · entropy ${ent.toFixed(2)} / ${maxEnt.toFixed(2)} max (${((ent/maxEnt)*100).toFixed(0)}% softness)</span>`
    : '';
  let lines = [];
  for (const [idx, val] of partners) {
    if (val < 0.001) continue;
    const r = other.regions[idx];
    const net = other.networks[idx];
    const isA = other.is_anchor[idx] ? '⚓' : ' ';
    const bar = '█'.repeat(Math.max(1, Math.round(val * 12)));
    lines.push(`<li><span style="color:#${val > 0.5 ? 'fde725' : (val > 0.1 ? '34d399' : '888')};">${bar}</span> <b>${(val*100).toFixed(1)}%</b> ${isA} → ${r} <span style="color:#888">(${net})</span></li>`);
  }
  if (!lines.length) lines.push('<li><i>(no concentrated partners, diffuse mapping)</i></li>');
  return `
    <h2>${species === 'mouse' ? '🐭' : '🧠'} ${species[0].toUpperCase()+species.slice(1)} node #${d.ids[i]}</h2>
    <div><b>Region:</b> ${d.regions[i]}</div>
    <div><b>Network:</b> <span style="color:${NET_COLORS[d.networks[i]] || '#999'}">${d.networks[i]}</span> · <b>Hemi:</b> ${d.regions[i].startsWith('L_') ? 'L' : (d.regions[i].startsWith('R_') ? 'R' : '?')} ${ringInfo}${entropyStr}</div>
    <div style="margin-top:6px;"><b>${direction}:</b></div>
    <ul class="matches">${lines.join('')}</ul>`;
}

document.querySelectorAll('#modegroup button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#modegroup button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMode = btn.dataset.mode;
    refreshAllBases();
  });
});

const netFilter = document.getElementById('netfilter');
DATA.networks.forEach(n => {
  const opt = document.createElement('option'); opt.value = n; opt.textContent = n;
  netFilter.appendChild(opt);
});
netFilter.addEventListener('change', refreshAllBases);
document.getElementById('hemifilter').addEventListener('change', refreshAllBases);
let searchTimer = null;
document.getElementById('regionsearch').addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(refreshAllBases, 200);
});
document.getElementById('reset').addEventListener('click', () => {
  netFilter.value = ''; document.getElementById('hemifilter').value = '';
  document.getElementById('regionsearch').value = '';
  selected = { species: null, idx: null };
  ['mouse', 'human'].forEach(sp => {
    Plotly.restyle(PLOT_IDS[sp], {
      x: [[]], y: [[]], z: [[]], 'marker.size': [[]], 'marker.color': [[]], text: [[]],
    }, [1]);
    Plotly.restyle(PLOT_IDS[sp], { x: [[]], y: [[]], z: [[]], text: [[]] }, [2]);
  });
  document.getElementById('mouse-panel').innerHTML = '<h2>🐭 Mouse, click a node</h2><div>Click a node on the left side.</div>';
  document.getElementById('human-panel').innerHTML = '<h2>🧠 Human, click a node</h2><div>Click a node on the right side.</div>';
  refreshAllBases();
});

document.getElementById('toggleHelp').addEventListener('click', (e) => {
  e.preventDefault();
  const box = document.getElementById('helpBox');
  box.style.display = (box.style.display === 'none') ? 'block' : 'none';
});

</script>
</body>
</html>
"""
