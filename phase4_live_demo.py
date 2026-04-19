import argparse
import re
import random
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request


MAX_EDGES = 3000
MAX_NODES_FOR_LAYOUT = 2000
RECENT_TABLE_SIZE = 20


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _safe_token(value: str, fallback: str = 'unk') -> str:
  token = re.sub(r'[^a-zA-Z0-9]+', '_', str(value or '')).strip('_').lower()
  return token or fallback


def _safe_int(value, default: int = 0) -> int:
  if pd.isna(value):
    return default
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def load_ieee_demo_records(project_root: Path, max_rows: int = 120000) -> List[Dict[str, str]]:
    """Load replay records from a joined IEEE test dataset (transaction + identity)."""
    tx_path = project_root / 'data' / 'test_transaction.csv'
    id_path = project_root / 'data' / 'test_identity.csv'
    if not tx_path.exists():
        return []

    tx_cols = {
        'TransactionID',
        'TransactionAmt',
        'ProductCD',
        'card1',
        'card2',
        'addr1',
    }
    tx_df = pd.read_csv(tx_path, usecols=lambda c: c in tx_cols, nrows=max_rows)
    if tx_df.empty:
        return []

    if id_path.exists():
        id_cols = {'TransactionID', 'DeviceType', 'DeviceInfo'}
        id_df = pd.read_csv(id_path, usecols=lambda c: c in id_cols, nrows=max_rows)
        joined_df = tx_df.merge(id_df, on='TransactionID', how='left')
    else:
        joined_df = tx_df

    product_to_channel = {
        'W': 'web',
        'C': 'mobile',
        'R': 'atm',
        'H': 'mobile',
        'S': 'web',
    }

    records: List[Dict[str, str]] = []
    for row in joined_df.itertuples(index=False):
        amount = float(getattr(row, 'TransactionAmt', 0.0) or 0.0)
        if amount <= 0:
            continue

        tx_id = _safe_int(getattr(row, 'TransactionID', 0), 0)
        product = str(getattr(row, 'ProductCD', 'U') or 'U')
        card1 = _safe_int(getattr(row, 'card1', 0), 0)
        card2 = _safe_int(getattr(row, 'card2', 0), 0)
        addr1 = _safe_int(getattr(row, 'addr1', 0), 0)
        dtype = _safe_token(getattr(row, 'DeviceType', 'unk'))
        dinfo = _safe_token(getattr(row, 'DeviceInfo', 'unk'))

        src = f"card_{card1}"
        dst = f"merchant_{product}_{addr1}_{card2}"
        bank = f"Bank-{(addr1 % 3) + 1}"
        device_id = f"dev_{dtype}_{dinfo}" if dinfo != 'unk' or dtype != 'unk' else f"dev_unknown_{tx_id % 1000}"
        channel = product_to_channel.get(product, 'web')

        records.append(
            {
                'src': src,
                'dst': dst,
                'amount': amount,
                'bank': bank,
                'device_id': device_id,
                'channel': channel,
            }
        )

    return records


class LiveFraudGraph:
    """In-memory transaction graph with lightweight fraud scoring for live demos."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tx_counter = 0
        self.fraud_counter = 0
        self.edges: Deque[Dict] = deque(maxlen=MAX_EDGES)
        self.node_stats: Dict[str, Dict] = {}
        self.last_seen_by_src: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=30))
        self.device_seen: Dict[str, set] = defaultdict(set)
        self.amount_window: Deque[float] = deque(maxlen=1000)

    def _ensure_node(self, node_id: str, bank: str) -> None:
        if node_id not in self.node_stats:
            self.node_stats[node_id] = {
                'id': node_id,
                'bank': bank,
                'total_tx': 0,
                'fraud_hits': 0,
                'last_seen': 0.0,
                'ever_flagged': 0,
            }

    def _score_transaction(self, src: str, amount: float, device_id: str, ts_unix: float) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        score = 0.0

        if len(self.amount_window) >= 30:
            mu = float(np.mean(self.amount_window))
            sigma = float(np.std(self.amount_window)) + 1e-6
            z = (amount - mu) / sigma
            if z > 3.0:
                score += 0.45
                reasons.append('amount_z>3')
            elif z > 2.0:
                score += 0.25
                reasons.append('amount_z>2')

        src_times = self.last_seen_by_src[src]
        recent = [t for t in src_times if ts_unix - t <= 90.0]
        if len(recent) >= 3:
            score += 0.30
            reasons.append('velocity_spike')

        if device_id and device_id not in self.device_seen[src]:
            score += 0.15
            reasons.append('new_device_for_source')

        if amount >= 5000:
            score += 0.20
            reasons.append('high_amount')

        return min(score, 0.99), reasons

    def add_transaction(self, src: str, dst: str, amount: float, bank: str, device_id: str, channel: str) -> Dict:
        ts_unix = time.time()
        with self.lock:
            self.tx_counter += 1
            tx_id = self.tx_counter

            self._ensure_node(src, bank)
            self._ensure_node(dst, bank)

            fraud_score, reasons = self._score_transaction(src, amount, device_id, ts_unix)
            is_fraud = int(fraud_score >= 0.60)

            self.node_stats[src]['total_tx'] += 1
            self.node_stats[dst]['total_tx'] += 1
            self.node_stats[src]['last_seen'] = ts_unix
            self.node_stats[dst]['last_seen'] = ts_unix
            if is_fraud:
                self.fraud_counter += 1
                self.node_stats[src]['fraud_hits'] += 1
                self.node_stats[dst]['fraud_hits'] += 1
                self.node_stats[src]['ever_flagged'] = 1
                self.node_stats[dst]['ever_flagged'] = 1

            self.amount_window.append(float(amount))
            self.last_seen_by_src[src].append(ts_unix)
            if device_id:
                self.device_seen[src].add(device_id)

            edge = {
                'tx_id': tx_id,
                'source': src,
                'target': dst,
                'amount': float(amount),
                'bank': bank,
                'device_id': device_id,
                'channel': channel,
                'score': float(fraud_score),
                'fraud': int(is_fraud),
                'reasons': reasons,
                'timestamp': utc_now_iso(),
                'ts_unix': ts_unix,
            }
            self.edges.append(edge)
            return edge

    def _build_graph(self, edges: List[Dict] | None = None) -> nx.Graph:
      g = nx.Graph()
      source_edges = edges if edges is not None else list(self.edges)
      for edge in source_edges:
        u, v = edge['source'], edge['target']
        if g.has_edge(u, v):
          g[u][v]['count'] += 1
          g[u][v]['fraud_count'] += edge['fraud']
          g[u][v]['max_score'] = max(g[u][v]['max_score'], edge['score'])
        else:
          g.add_edge(
            u,
            v,
            count=1,
            fraud_count=edge['fraud'],
            max_score=edge['score'],
          )
      return g

    def _fraud_components(self, g: nx.Graph) -> List[Dict]:
        fraud_sub = nx.Graph()
        for u, v, attrs in g.edges(data=True):
            if attrs.get('fraud_count', 0) > 0:
                fraud_sub.add_edge(u, v, **attrs)

        components = []
        for i, nodes in enumerate(nx.connected_components(fraud_sub), start=1):
            sub = fraud_sub.subgraph(nodes)
            edge_count = sub.number_of_edges()
            fraud_edges = sum(1 for _u, _v, a in sub.edges(data=True) if a.get('fraud_count', 0) > 0)
            max_score = 0.0
            for _u, _v, a in sub.edges(data=True):
                max_score = max(max_score, float(a.get('max_score', 0.0)))

            components.append(
                {
                    'component_id': i,
                    'node_count': len(nodes),
                    'edge_count': edge_count,
                    'fraud_edge_count': fraud_edges,
                    'max_score': round(max_score, 3),
                    'nodes': sorted(nodes),
                }
            )

        components.sort(key=lambda c: (c['fraud_edge_count'], c['max_score']), reverse=True)
        return components[:8]

    def state(self) -> Dict:
        with self.lock:
            edges = list(self.edges)
            node_snapshot = {k: v.copy() for k, v in self.node_stats.items()}
        total_tx_seen = int(self.tx_counter)
        total_fraud_seen = int(self.fraud_counter)

        g = nx.Graph()
        for edge in edges:
            g.add_edge(edge['source'], edge['target'])

        # Keep historically flagged fraud nodes visible even if their edges age out.
        for node_id, stats in node_snapshot.items():
            if int(stats.get('ever_flagged', 0)) > 0:
                g.add_node(node_id)

        if g.number_of_nodes() > MAX_NODES_FOR_LAYOUT:
            fraud_nodes = {
                node_id
                for node_id, stats in node_snapshot.items()
                if int(stats.get('ever_flagged', 0)) > 0 and node_id in g
            }
            ranked_nodes = [n for n, _d in sorted(g.degree, key=lambda x: x[1], reverse=True)]
            keep_nodes = set(list(fraud_nodes)[:MAX_NODES_FOR_LAYOUT])
            for node_id in ranked_nodes:
                if len(keep_nodes) >= MAX_NODES_FOR_LAYOUT:
                    break
                keep_nodes.add(node_id)
            g = g.subgraph(keep_nodes).copy()

        if g.number_of_nodes() > 0:
            pos = nx.spring_layout(g, seed=42, k=0.8 / np.sqrt(max(g.number_of_nodes(), 2)), iterations=40)
        else:
            pos = {}

        degree_map = dict(g.degree()) if g.number_of_nodes() > 0 else {}
        last_seen_values = [
          float(node_snapshot.get(node_id, {}).get('last_seen', 0.0) or 0.0)
          for node_id in g.nodes()
        ]
        non_zero_seen = [v for v in last_seen_values if v > 0.0]
        min_seen = min(non_zero_seen) if non_zero_seen else 0.0
        max_seen = max(non_zero_seen) if non_zero_seen else 1.0

        nodes_payload = []
        now_unix = time.time()
        for node_id in g.nodes():
            stats = node_snapshot.get(node_id, {'bank': 'unknown', 'total_tx': 0, 'fraud_hits': 0, 'last_seen': 0.0, 'ever_flagged': 0})
            tx = max(int(stats.get('total_tx', 0)), 1)
            fraud_hits = int(stats.get('fraud_hits', 0))
            risk = fraud_hits / tx
            x, y = pos[node_id]
            last_seen = float(stats.get('last_seen', 0.0) or 0.0)
            age_seconds = max(0.0, now_unix - last_seen) if last_seen > 0 else 9999.0
            if last_seen > 0 and max_seen > min_seen:
              time_norm = (last_seen - min_seen) / (max_seen - min_seen)
            elif last_seen > 0:
              time_norm = 0.5
            else:
              time_norm = 0.0
            recency_boost = float(np.exp(-age_seconds / 180.0))
            z = (2.2 * time_norm) + (0.8 * recency_boost) + (0.6 * risk)
            nodes_payload.append(
                {
                    'id': node_id,
                    'bank': stats.get('bank', 'unknown'),
                    'x': float(x),
                    'y': float(y),
                    'z': round(float(z), 3),
                    'degree': int(degree_map.get(node_id, 0)),
                    'total_tx': int(tx),
                    'fraud_hits': fraud_hits,
                    'ever_flagged': int(stats.get('ever_flagged', 0)),
                    'risk': round(float(risk), 3),
                    'last_seen_age_s': round(age_seconds, 1),
                }
            )

        graph_with_attrs = self._build_graph(edges)
        fraud_components = self._fraud_components(graph_with_attrs)

        edge_payload = []
        visible_nodes = {n['id'] for n in nodes_payload}
        for edge in edges[-200:]:
            if edge['source'] in visible_nodes and edge['target'] in visible_nodes:
                edge_payload.append(
                    {
                        'source': edge['source'],
                        'target': edge['target'],
                        'amount': edge['amount'],
                        'fraud': edge['fraud'],
                        'score': round(edge['score'], 3),
                        'tx_id': edge['tx_id'],
                        'timestamp': edge['timestamp'],
                        'reasons': edge['reasons'],
                    }
                )

        total_tx = total_tx_seen
        fraud_tx = total_fraud_seen

        return {
            'metrics': {
                'total_transactions': total_tx,
                'fraud_transactions': fraud_tx,
                'fraud_rate': round((fraud_tx / total_tx) if total_tx else 0.0, 4),
                'active_nodes': len(node_snapshot),
                'visible_nodes': len(nodes_payload),
            },
            'nodes': nodes_payload,
            'edges': edge_payload,
            'fraud_components': fraud_components,
            'latest_transactions': edges[-RECENT_TABLE_SIZE:][::-1],
        }


def build_app() -> Flask:
  app = Flask(__name__)
  store = LiveFraudGraph()
  project_root = Path(__file__).resolve().parent
  dataset_records = load_ieee_demo_records(project_root)
  dataset_state = {'cursor': 0}

  @app.get('/')
  def dashboard():
    return render_template_string(DASHBOARD_HTML)

  @app.get('/submit')
  def submit_page():
    return render_template_string(SENDER_HTML)

  @app.post('/api/transaction')
  def api_transaction():
    payload = request.get_json(silent=True) or request.form.to_dict()

    src = str(payload.get('src', '')).strip() or f"acct_{random.randint(1000, 9999)}"
    dst = str(payload.get('dst', '')).strip() or f"merchant_{random.randint(100, 999)}"
    bank = str(payload.get('bank', 'Bank-A')).strip() or 'Bank-A'
    device_id = str(payload.get('device_id', payload.get('device', 'device-web'))).strip()
    channel = str(payload.get('channel', 'mobile')).strip() or 'mobile'

    try:
      amount = float(payload.get('amount', 100.0))
    except (TypeError, ValueError):
      amount = 100.0

    amount = max(amount, 1.0)
    event = store.add_transaction(src=src, dst=dst, amount=amount, bank=bank, device_id=device_id, channel=channel)
    return jsonify({'ok': True, 'event': event})

  @app.get('/api/state')
  def api_state():
    return jsonify(store.state())

  @app.post('/api/simulate')
  def api_simulate():
    payload = request.get_json(silent=True) or {}
    count = int(payload.get('count', 15))
    count = max(1, min(count, 10000))

    banks = ['Bank-A', 'Bank-B', 'Bank-C']
    channels = ['mobile', 'web', 'atm']

    if dataset_records:
      start = dataset_state['cursor']
      total = len(dataset_records)
      for i in range(count):
        rec = dataset_records[(start + i) % total]
        store.add_transaction(
          src=rec['src'],
          dst=rec['dst'],
          amount=float(rec['amount']),
          bank=rec['bank'],
          device_id=rec['device_id'],
          channel=rec['channel'],
        )
      dataset_state['cursor'] = (start + count) % total
      return jsonify({'ok': True, 'added': count, 'source': 'data/test_transaction.csv + data/test_identity.csv'})

    for _ in range(count):
      src = f"acct_{random.randint(1000, 1025)}"
      dst = f"merchant_{random.randint(300, 340)}"
      amount = float(max(10, np.random.lognormal(mean=6.0, sigma=0.8)))
      bank = random.choice(banks)
      device = f"dev_{random.randint(1, 8)}"
      channel = random.choice(channels)
      store.add_transaction(src, dst, amount, bank, device, channel)

    return jsonify({'ok': True, 'added': count, 'source': 'synthetic-fallback'})

  return app


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Live Fraud Graph Demo</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    :root {
      --bg: #f8f7f4;
      --ink: #1f2a2e;
      --card: #ffffff;
      --accent: #0b6e4f;
      --warn: #c1121f;
      --muted: #607d8b;
      --line: #d6d9dc;
    }
    body {
      margin: 0;
      font-family: 'Trebuchet MS', 'Gill Sans', sans-serif;
      background: radial-gradient(circle at 20% 10%, #fff2cf 0%, var(--bg) 55%);
      color: var(--ink);
    }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 16px; }
    .hero { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; }
    .title { font-size: 28px; font-weight: 700; letter-spacing: 0.2px; }
    .subtitle { color: var(--muted); margin-top: 4px; }
    .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 14px; margin-top: 14px; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 6px 16px rgba(0,0,0,0.05);
      padding: 10px;
    }
    .stats { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 8px; margin-top: 12px; }
    .stat { background: #f7fbfc; border: 1px solid #dce8ec; border-radius: 10px; padding: 8px; }
    .stat b { display: block; font-size: 20px; }
    .pill { display: inline-block; background: #e9f7f1; color: var(--accent); border-radius: 999px; padding: 4px 10px; }
    .btn {
      background: var(--accent);
      color: white;
      border: 0;
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
    }
    .btn.warn { background: var(--warn); }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid #eceff1; padding: 6px; text-align: left; }
    .fraud { color: var(--warn); font-weight: 700; }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      .stats { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"hero\">
      <div>
        <div class=\"title\">Live Fraud Network (GNN-style View)</div>
        <div class=\"subtitle\">Share <span class=\"pill\">/submit</span> to phones. This dashboard auto-refreshes every 2s.</div>
      </div>
      <div>
        <button class=\"btn\" onclick=\"seedData(10)\">10 Tx</button>
        <button class=\"btn\" onclick=\"seedData(20)\">20 Tx</button>
        <button class=\"btn\" onclick=\"seedData(50)\">50 Tx</button>
        <button class=\"btn\" onclick=\"seedData(100)\">100 Tx</button>
        <button class=\"btn\" onclick=\"seedData(1000)\">1000 Tx</button>
        <button class=\"btn\" onclick=\"seedData(5000)\">5000 Tx</button>
        <button class=\"btn\" onclick=\"seedData(10000)\">10000 Tx</button>
      </div>
    </div>
    <div id=\"seed_status\" style=\"margin-top:8px;color:#64748b;font-size:13px;\">Seed source: waiting...</div>

    <div class=\"stats\">
      <div class=\"stat\"><span>Total Tx</span><b id=\"s_total\">0</b></div>
      <div class=\"stat\"><span>Fraud Tx</span><b id=\"s_fraud\">0</b></div>
      <div class=\"stat\"><span>Fraud Rate</span><b id=\"s_rate\">0.00%</b></div>
      <div class=\"stat\"><span>Active Nodes</span><b id=\"s_nodes\">0</b></div>
    </div>

    <div class=\"grid\">
      <div class=\"card\">
        <div id=\"graph\" style=\"height:560px;\"></div>
      </div>
      <div class=\"card\">
        <h3 style=\"margin:6px 0 8px;\">Node Inspector</h3>
        <div id=\"node_inspector\" style=\"font-size:13px;margin-bottom:12px;color:#475569;\">Click any node in the 3D graph to inspect details.</div>
        <h3 style=\"margin:6px 0 8px;\">Fraud Components</h3>
        <div id=\"fraud_components\" style=\"font-size:13px;\"></div>
        <h3 style=\"margin:16px 0 8px;\">Latest Transactions</h3>
        <div style=\"max-height:280px;overflow:auto\">
          <table>
            <thead><tr><th>ID</th><th>Src→Dst</th><th>Amt</th><th>Score</th></tr></thead>
            <tbody id=\"tx_rows\"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

<script>
let selectedNodeId = null;
const stickyFraudNodes = new Set();
let currentCamera = null;
let graphInitialized = false;
let latestGraphData = null;
const frozenNodePositions = new Map();

function loadSavedCamera() {
  try {
    const raw = window.localStorage.getItem('fraud_graph_camera');
    return raw ? JSON.parse(raw) : null;
  } catch (_e) {
    return null;
  }
}

function saveCamera(cameraObj) {
  try {
    window.localStorage.setItem('fraud_graph_camera', JSON.stringify(cameraObj));
  } catch (_e) {
    // Ignore storage errors in private/incognito mode.
  }
}

function registerStickyFraud(data) {
  (data.nodes || []).forEach(n => {
    if (Number(n.ever_flagged) > 0 || Number(n.fraud_hits) > 0 || Number(n.risk) >= 0.40) {
      stickyFraudNodes.add(n.id);
    }
  });
}

function getFrozenNode(node) {
  if (!frozenNodePositions.has(node.id)) {
    frozenNodePositions.set(node.id, {x: node.x, y: node.y});
  }
  return frozenNodePositions.get(node.id);
}

async function seedData(count) {
  const res = await fetch('/api/simulate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({count: count})});
  const out = await res.json();
  const source = out.source || 'unknown';
  document.getElementById('seed_status').textContent = `Seed source: ${source} | added ${out.added || count} tx`;
  await refresh();
}

function renderGraph(data) {
  latestGraphData = data;
  registerStickyFraud(data);
  const edgeX = [];
  const edgeY = [];
  const edgeZ = [];
  const nodeIds = data.nodes.map(n => n.id);

  const frozenNodes = data.nodes.map(n => {
    const p = getFrozenNode(n);
    return { ...n, x: p.x, y: p.y, z: n.z };
  });

  const frozenMap = new Map(frozenNodes.map(n => [n.id, n]));
  data.edges.forEach(e => {
    const s = frozenMap.get(e.source);
    const t = frozenMap.get(e.target);
    if (!s || !t) return;
    edgeX.push(s.x, t.x, null);
    edgeY.push(s.y, t.y, null);
    edgeZ.push(s.z, t.z, null);
  });

  const nodeX = frozenNodes.map(n => n.x);
  const nodeY = frozenNodes.map(n => n.y);
  const nodeZ = frozenNodes.map(n => n.z);
  const riskColorValues = frozenNodes.map(n => {
    const r = Number(n.risk) || 0.0;
    // Keep historically flagged nodes visually in high-risk region.
    return stickyFraudNodes.has(n.id) ? Math.max(r, 0.85) : r;
  });
  const sizes = frozenNodes.map(n => 10 + Math.min(n.degree * 1.5, 18));
  const hoverLabels = frozenNodes.map(n => `${n.id}<br>bank=${n.bank}<br>risk=${n.risk}<br>fraud_hits=${n.fraud_hits}<br>age=${n.last_seen_age_s}s`);
  const nodeText = frozenNodes.map(n => n.risk > 0.25 ? n.id : '');

  const edgeTrace = {
    x: edgeX,
    y: edgeY,
    z: edgeZ,
    mode: 'lines',
    hoverinfo: 'none',
    line: { width: 1, color: '#94a3b8' },
    type: 'scatter3d',
  };

  const nodeTrace = {
    x: nodeX,
    y: nodeY,
    z: nodeZ,
    mode: 'markers+text',
    type: 'scatter3d',
    text: nodeText,
    textposition: 'top center',
    textfont: {size: 10, color: '#0f172a'},
    hovertext: hoverLabels,
    hovertemplate: '%{hovertext}<extra></extra>',
    customdata: nodeIds,
    marker: {
      color: riskColorValues,
      colorscale: 'YlOrRd_r',
      cmin: 0,
      cmax: 1,
      colorbar: {
        title: 'Risk',
        thickness: 12,
      },
      size: sizes,
      line: { color: '#1f2937', width: 0.4 }
    }
  };

  const layout = {
    margin: {l: 10, r: 10, b: 10, t: 10},
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    uirevision: 'graph-view-lock',
    scene: {
      uirevision: 'graph-view-lock',
      xaxis: {visible: false, range: [-1.2, 1.2], autorange: false},
      yaxis: {visible: false, range: [-1.2, 1.2], autorange: false},
      zaxis: {
        visible: true,
        title: 'Temporal-Risk Axis',
        range: [0.0, 4.0],
        autorange: false,
        showgrid: true,
        gridcolor: '#e2e8f0',
      },
      camera: {
        eye: {x: 1.35, y: 1.35, z: 1.1}
      },
      aspectmode: 'cube',
    },
    showlegend: false,
  };

  if (!currentCamera) {
    currentCamera = loadSavedCamera();
  }

  if (currentCamera) {
    layout.scene.camera = currentCamera;
  }

  if (!graphInitialized) {
    Plotly.newPlot('graph', [edgeTrace, nodeTrace], layout, {displayModeBar:false, responsive:true});
    graphInitialized = true;
  } else {
    Plotly.update('graph', {
      x: [edgeX, nodeX],
      y: [edgeY, nodeY],
      z: [edgeZ, nodeZ],
      text: [null, nodeText],
      hovertext: [null, hoverLabels],
      customdata: [null, nodeIds],
      'marker.color': [null, riskColorValues],
      'marker.size': [null, sizes],
    }, {}, [0, 1]);

    if (currentCamera) {
      Plotly.relayout('graph', {'scene.camera': currentCamera});
    }
  }

  const graphEl = document.getElementById('graph');
  const liveCamera = graphEl?.layout?.scene?.camera;
  if (!currentCamera && liveCamera) {
    currentCamera = liveCamera;
    saveCamera(currentCamera);
  }

  if (!graphEl.__handlersBound) {
    graphEl.on('plotly_relayout', () => {
      const cam = graphEl?.layout?.scene?.camera;
      if (cam) {
        currentCamera = cam;
        saveCamera(currentCamera);
      }
    });

    graphEl.on('plotly_click', (evt) => {
      const point = evt && evt.points && evt.points[0];
      if (!point || !point.customdata) return;
      selectedNodeId = point.customdata;
      renderNodeInspector(selectedNodeId, latestGraphData || data);
    });

    graphEl.__handlersBound = true;
  }
}

function renderComponents(comps) {
  const container = document.getElementById('fraud_components');
  if (!comps || comps.length === 0) {
    container.innerHTML = '<div>No fraud components detected yet.</div>';
    return;
  }

  container.innerHTML = comps.map(c => {
    const preview = c.nodes.slice(0,5).join(', ');
    return `<div style=\"border:1px solid #f1d5d8;border-radius:8px;padding:8px;margin-bottom:8px;background:#fff8f8\">
      <b class=\"fraud\">Component #${c.component_id}</b><br>
      nodes=${c.node_count}, fraud-edges=${c.fraud_edge_count}, max-score=${c.max_score}<br>
      <span style=\"color:#6b7280\">${preview}${c.nodes.length > 5 ? ' ...' : ''}</span>
    </div>`;
  }).join('');
}

function renderLatest(latest) {
  const rows = document.getElementById('tx_rows');
  rows.innerHTML = latest.map(tx => {
    const fraudClass = tx.fraud ? 'fraud' : '';
    return `<tr>
      <td>${tx.tx_id}</td>
      <td>${tx.source}→${tx.target}</td>
      <td>${tx.amount.toFixed(0)}</td>
      <td class=\"${fraudClass}\">${tx.score.toFixed(2)}</td>
    </tr>`;
  }).join('');
}

function renderNodeInspector(nodeId, data) {
  const el = document.getElementById('node_inspector');
  if (!nodeId) {
    el.innerHTML = 'Click any node in the 3D graph to inspect details.';
    return;
  }

  const node = (data.nodes || []).find(n => n.id === nodeId);
  if (!node) {
    el.innerHTML = 'Selected node is not visible in current graph window.';
    return;
  }

  const related = (data.edges || [])
    .filter(e => e.source === nodeId || e.target === nodeId)
    .sort((a, b) => b.tx_id - a.tx_id)
    .slice(0, 6);

  const relatedHtml = related.length > 0
    ? related.map(tx => {
        const other = tx.source === nodeId ? tx.target : tx.source;
        const cls = tx.fraud ? 'fraud' : '';
        return `<div style=\"padding:4px 0;border-bottom:1px solid #eef2f7\">tx#${tx.tx_id} with <b>${other}</b>, amt=${Number(tx.amount).toFixed(0)}, <span class=\"${cls}\">score=${Number(tx.score).toFixed(2)}</span></div>`;
      }).join('')
    : '<div>No recent connected transactions in the visible window.</div>';

  el.innerHTML = `<div style=\"border:1px solid #cbd5e1;border-radius:8px;padding:8px;background:#f8fafc\">\
    <div><b>${node.id}</b> (${node.bank})</div>\
    <div style=\"margin-top:4px\">risk=${node.risk}, total_tx=${node.total_tx}, fraud_hits=${node.fraud_hits}, degree=${node.degree}</div>\
    <div>last_seen_age=${node.last_seen_age_s}s, temporal_z=${node.z}</div>\
    <div style=\"margin-top:8px;color:#475569\"><b>Connected recent transactions</b></div>\
    ${relatedHtml}\
  </div>`;
}

async function refresh() {
  const res = await fetch('/api/state');
  const data = await res.json();
  document.getElementById('s_total').textContent = data.metrics.total_transactions;
  document.getElementById('s_fraud').textContent = data.metrics.fraud_transactions;
  document.getElementById('s_rate').textContent = (100 * data.metrics.fraud_rate).toFixed(2) + '%';
  document.getElementById('s_nodes').textContent = data.metrics.active_nodes;

  renderGraph(data);
  renderComponents(data.fraud_components);
  renderLatest(data.latest_transactions);
  renderNodeInspector(selectedNodeId, data);
}

refresh();
setInterval(refresh, 2500);
</script>
</body>
</html>
"""


SENDER_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Send Transaction</title>
  <style>
    body {
      margin: 0;
      font-family: 'Trebuchet MS', 'Gill Sans', sans-serif;
      background: linear-gradient(140deg, #d9f8ef, #f4e6c9);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 16px;
    }
    .card {
      width: min(460px, 100%);
      background: rgba(255,255,255,0.95);
      border-radius: 14px;
      box-shadow: 0 12px 24px rgba(0,0,0,0.08);
      padding: 16px;
    }
    h2 { margin: 4px 0 12px; }
    label { font-size: 13px; color: #374151; }
    input, select {
      width: 100%;
      border: 1px solid #cfd8dc;
      border-radius: 8px;
      padding: 10px;
      margin: 6px 0 10px;
      box-sizing: border-box;
      font-size: 14px;
    }
    button {
      width: 100%;
      border: 0;
      background: #0b6e4f;
      color: white;
      padding: 11px;
      border-radius: 8px;
      font-size: 15px;
    }
    .ok { margin-top: 10px; color: #166534; font-size: 13px; }
  </style>
</head>
<body>
  <div class=\"card\">
    <h2>Send Demo Transaction</h2>
    <label>Source Account</label>
    <input id=\"src\" value=\"acct_1001\" />

    <label>Destination / Merchant</label>
    <input id=\"dst\" value=\"merchant_301\" />

    <label>Amount</label>
    <input id=\"amount\" type=\"number\" value=\"750\" min=\"1\" />

    <label>Bank</label>
    <select id=\"bank\">
      <option>Bank-A</option>
      <option>Bank-B</option>
      <option>Bank-C</option>
    </select>

    <label>Device ID</label>
    <input id=\"device\" value=\"dev_1\" />

    <label>Channel</label>
    <select id=\"channel\">
      <option>mobile</option>
      <option>web</option>
      <option>atm</option>
    </select>

    <button onclick=\"submitTx()\">Send Transaction</button>
    <div class=\"ok\" id=\"status\"></div>
  </div>

<script>
async function submitTx() {
  const payload = {
    src: document.getElementById('src').value,
    dst: document.getElementById('dst').value,
    amount: Number(document.getElementById('amount').value),
    bank: document.getElementById('bank').value,
    device_id: document.getElementById('device').value,
    channel: document.getElementById('channel').value,
  };

  const res = await fetch('/api/transaction', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });

  const out = await res.json();
  if (out.ok) {
    const e = out.event;
    document.getElementById('status').textContent = `Sent tx #${e.tx_id} | score=${e.score.toFixed(2)} | fraud=${e.fraud}`;
  }
}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description='Live multi-device fraud graph demo')
    parser.add_argument('--host', default='0.0.0.0', help='Host bind address')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable Flask debug mode')
    args = parser.parse_args()

    app = build_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
