# ============================================================
#  PHASE 3 — Federated Learning Simulation (Flower)
#  Simulates 3 banks training collaboratively
#  without sharing raw data — all in one Kaggle session
#  Install first: pip install flwr torch-geometric
# ============================================================

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
import pickle
import copy
import os
from collections import OrderedDict
from sklearn.metrics import (average_precision_score,
                              recall_score,
                              precision_score, roc_auc_score)
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
# Enable CUDA optimizations when available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def resolve_artifacts_dir():
    """Use Kaggle artifacts dir when available, otherwise local ./artifacts."""
    kaggle_dir = '/kaggle/working/artifacts'
    local_dir = os.path.join('.', 'artifacts')
    if os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir, exist_ok=True)
        return kaggle_dir
    os.makedirs(local_dir, exist_ok=True)
    return local_dir


ARTIFACTS_DIR = resolve_artifacts_dir()
PREPROCESSED_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessed.pkl')
CENTRALISED_PATH = os.path.join(ARTIFACTS_DIR, 'centralised_metrics.pkl')
FL_HISTORY_PATH = os.path.join(ARTIFACTS_DIR, 'fl_history.pkl')


# ── 1. Re-use same model architecture from Phase 2 ───────────
class FraudSTGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32,
                 gru_hidden=16, dropout=0.5):
        super().__init__()
        self.sage1      = SAGEConv(in_dim, hidden_dim)
        self.sage2      = SAGEConv(hidden_dim, out_dim)
        self.bn1        = nn.BatchNorm1d(hidden_dim)
        self.bn2        = nn.BatchNorm1d(out_dim)
        self.h1_proj    = nn.Linear(hidden_dim, out_dim)
        self.gru        = nn.GRU(out_dim, gru_hidden,
                                 batch_first=True)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x, edge_index):
        h1 = F.normalize(F.relu(self.bn1(
                self.sage1(x, edge_index))), p=2, dim=1)
        h1 = self.dropout(h1)
        h1_proj = F.normalize(self.h1_proj(h1), p=2, dim=1)
        h2 = F.normalize(F.relu(self.bn2(
                self.sage2(h1, edge_index))), p=2, dim=1)
        h2 = self.dropout(h2)
        seq     = torch.stack([h1_proj, h2], dim=1)
        _, h_n  = self.gru(seq)
        out     = self.classifier(h_n.squeeze(0))
        return out.squeeze(1)


# ── 2. Load & Partition Data into 3 "Banks" ──────────────────
print("Partitioning data into 3 federated clients (banks)...")

if not os.path.exists(PREPROCESSED_PATH):
    raise FileNotFoundError(
        f"Missing preprocessed data at {PREPROCESSED_PATH}. "
        "Run phase1_preprocessing.py first."
    )

with open(PREPROCESSED_PATH, 'rb') as f:
    artifacts = pickle.load(f)
edge_list = artifacts['edge_list']

# Shuffle and split edge list into 3 non-overlapping partitions
# This simulates non-IID data across banks
np.random.seed(42)
idx       = np.random.permutation(len(edge_list))
splits    = np.array_split(idx, 3)

def build_pyg_graph(edges_subset):
    """Convert a subset of edges into a PyG graph"""
    all_ids    = set()
    for e in edges_subset:
        all_ids.add(e['src'])
        all_ids.add(e['dst'])

    id_map   = {nid: i for i, nid in enumerate(sorted(all_ids))}
    n_nodes  = len(id_map)
    feat_dim = edges_subset[0]['feats'].shape[0]

    node_feats  = np.zeros((n_nodes, feat_dim), dtype=np.float32)
    node_counts = np.ones(n_nodes,  dtype=np.int32)
    node_labels = np.zeros(n_nodes, dtype=np.float32)
    src_list, dst_list = [], []

    for e in edges_subset:
        s, d = id_map[e['src']], id_map[e['dst']]
        src_list.append(s)
        dst_list.append(d)
        node_feats[s]  += e['feats']
        node_counts[s] += 1
        node_feats[d]  += e['feats']
        node_counts[d] += 1
        if e['label'] == 1:
            node_labels[s] = 1
            node_labels[d] = 1

    node_feats /= node_counts[:, None]

    # Bidirectional
    edge_index = torch.tensor(
        [src_list + dst_list, dst_list + src_list], dtype=torch.long
    )

    n      = n_nodes
    t_mask = torch.zeros(n, dtype=torch.bool)
    v_mask = torch.zeros(n, dtype=torch.bool)
    t_mask[:int(n * 0.8)] = True
    v_mask[int(n * 0.8):] = True

    return Data(
        x          = torch.tensor(node_feats, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(node_labels, dtype=torch.float),
        train_mask = t_mask,
        val_mask   = v_mask
    )

# Build one graph per bank
bank_graphs = []
for bank_id, split_idx in enumerate(splits):
    subset = [edge_list[i] for i in split_idx]
    g      = build_pyg_graph(subset)
    bank_graphs.append(g)
    fraud_rate = g.y.mean().item() * 100
    print(f"  Bank {bank_id+1}: {g.num_nodes} nodes | "
          f"{g.num_edges} edges | Fraud rate: {fraud_rate:.2f}%")

IN_DIM = bank_graphs[0].num_node_features


# ── 3. Flower Client Definition ───────────────────────────────
class FraudDetectionClient(fl.client.NumPyClient):
    """
    One Flower client = one bank
    Trains local STGNN, shares model weights
    In real deployment: replace weights with Local Sufficient
    Statistics (μ, Σ) + Laplace noise for privacy
    """

    def __init__(self, client_id: int, graph: Data):
        self.client_id = client_id
        self.graph     = graph.to(DEVICE)
        self.model     = FraudSTGNN(IN_DIM).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([3.0]).to(DEVICE)
        )

    def get_parameters(self, config) -> List[np.ndarray]:
        """Extract model weights as numpy arrays"""
        return [np.asarray(val.detach().cpu().tolist())
                for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Load aggregated global weights into local model"""
        state_dict = OrderedDict()
        for (k, old_tensor), v in zip(self.model.state_dict().items(), parameters):
            v_arr = np.asarray(v)
            state_dict[k] = torch.tensor(
                v_arr.tolist(),
                dtype=old_tensor.dtype,
                device=old_tensor.device
            )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Local training — runs E epochs on private bank data
        Returns updated weights to aggregation server
        """
        self.set_parameters(parameters)

        local_epochs = config.get('local_epochs', 5)
        lr           = config.get('lr', 1e-3)
        optimizer    = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )

        self.model.train()
        for epoch in range(local_epochs):
            optimizer.zero_grad()
            logits = self.model(self.graph.x, self.graph.edge_index)
            loss   = self.criterion(
                logits[self.graph.train_mask],
                self.graph.y[self.graph.train_mask]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            optimizer.step()

        # Return weights, num samples, and metrics
        return (self.get_parameters(config={}),
                int(self.graph.train_mask.sum()),
                {'loss': float(loss.item()),
                 'client_id': self.client_id})

    def evaluate(self, parameters, config):
        """Evaluate global model on local validation data"""
        self.set_parameters(parameters)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(self.graph.x, self.graph.edge_index)
            probs  = torch.sigmoid(logits)
            loss   = self.criterion(
                logits[self.graph.val_mask],
                self.graph.y[self.graph.val_mask]
            )

        y_true = np.asarray(
            self.graph.y[self.graph.val_mask].detach().cpu().tolist(),
            dtype=np.float32
        )
        y_prob = np.asarray(
            probs[self.graph.val_mask].detach().cpu().tolist(),
            dtype=np.float32
        )
        y_pred = (y_prob > 0.5).astype(int)

        metrics = {}
        if len(np.unique(y_true)) > 1:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            beta_sq = 4.0
            f2 = ((1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
                  if (beta_sq * precision + recall) > 0 else 0.0)
            metrics = {
                'auc_pr': float(average_precision_score(y_true, y_prob)),
                'f2': float(f2),
                'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            }

        return (float(loss.item()),
                int(self.graph.val_mask.sum()),
                metrics)


# ── 4. Aggregation Strategy ───────────────────────────────────
def weighted_average(metrics):
    """
    Custom aggregation: weighted average of client metrics
    Weight by number of validation samples per client
    """
    total = sum(num for num, _ in metrics)
    auc_prs = [num * m.get('auc_pr', 0) for num, m in metrics]
    f2s     = [num * m.get('f2', 0)     for num, m in metrics]
    recalls = [num * m.get('recall', 0) for num, m in metrics]

    return {
        'auc_pr':  sum(auc_prs) / total if total > 0 else 0,
        'f2':      sum(f2s)     / total if total > 0 else 0,
        'recall':  sum(recalls) / total if total > 0 else 0,
    }


# ── 5. PSI Drift Detection ────────────────────────────────────
def compute_psi(expected: np.ndarray,
                actual: np.ndarray,
                bins: int = 10) -> float:
    """
    Population Stability Index
    PSI = Σ (actual% - expected%) × ln(actual% / expected%)
    PSI < 0.10 → no change
    PSI 0.10–0.25 → slight change, monitor
    PSI > 0.25 → significant shift → trigger retraining
    """
    expected = np.clip(expected, 1e-6, None)
    actual   = np.clip(actual,   1e-6, None)

    breakpoints = np.linspace(0, 1, bins + 1)
    exp_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual,   breakpoints)[0] / len(actual)

    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


# ── 6. Run Federated Simulation ───────────────────────────────
print("\nStarting Flower federated simulation...")
print(f"  Clients (banks): 3")
print(f"  Rounds: 20")
print(f"  Local epochs per round: 5")
print()

fl_metrics_history = []

def client_fn(cid: str) -> fl.client.Client:
    client_id = int(cid)
    return FraudDetectionClient(
        client_id = client_id,
        graph     = bank_graphs[client_id]
    ).to_client()

strategy = fl.server.strategy.FedAvg(
    fraction_fit          = 1.0,   # use all 3 clients every round
    fraction_evaluate     = 1.0,
    min_fit_clients       = 3,
    min_evaluate_clients  = 3,
    min_available_clients = 3,
    evaluate_metrics_aggregation_fn = weighted_average,
    on_fit_config_fn = lambda rnd: {
        'local_epochs': 5,
        'lr': max(1e-4, 1e-3 * (0.95 ** rnd))  # decay lr per round
    }
)

history = fl.simulation.start_simulation(
    client_fn        = client_fn,
    num_clients      = 3,
    config           = fl.server.ServerConfig(num_rounds=20),
    strategy         = strategy,
    client_resources = {'num_cpus': 1,
                        'num_gpus': 0.3 if torch.cuda.is_available()
                                    else 0}
)

# ── 7. Print Results & Compare with Centralised ──────────────
print("\n" + "="*55)
print("FEDERATED LEARNING RESULTS (Round 20)")
print("="*55)

# Get final round metrics
if history.metrics_distributed:
    final = history.metrics_distributed
    for metric, values in final.items():
        if values:
            last_val = values[-1][1]
            print(f"  {metric:12s}: {last_val:.4f}")

# Load centralised baseline from Phase 2
try:
    with open(CENTRALISED_PATH, 'rb') as f:
        central = pickle.load(f)
    print("\n" + "-"*55)
    print("COMPARISON: Federated vs Centralised")
    print("-"*55)
    print(f"{'Metric':<15} {'Centralised':>15} {'Federated':>15} {'Δ':>10}")
    print("-"*55)

    fed_final = {k: v[-1][1] for k, v in
                 history.metrics_distributed.items() if v}

    for m in ['auc_pr', 'f2', 'recall']:
        c_val = central.get(m, 0)
        f_val = fed_final.get(m, 0)
        delta = f_val - c_val
        print(f"  {m:<13} {c_val:>15.4f} {f_val:>15.4f} "
              f"{'↑' if delta >= 0 else '↓'}{abs(delta):>8.4f}")

except FileNotFoundError:
    print("  (Run Phase 2 first for centralised comparison)")

# ── 8. PSI Drift Check ────────────────────────────────────────
print("\n" + "-"*55)
print("DRIFT DETECTION (PSI) — Bank 1 vs Bank 2")
print("-"*55)

# Compare feature distributions between two banks (simulates drift)
feats_bank1 = np.asarray(bank_graphs[0].x[:, 0].detach().cpu().tolist())  # TransactionAmt proxy
feats_bank2 = np.asarray(bank_graphs[1].x[:, 0].detach().cpu().tolist())

psi = compute_psi(feats_bank1, feats_bank2)
print(f"  PSI Score: {psi:.4f}")
if psi < 0.10:
    print("  Status: ✓ STABLE — no retraining needed")
elif psi < 0.25:
    print("  Status: ⚠ MONITOR — slight distribution shift")
else:
    print("  Status: ✗ RETRAIN — significant drift detected (PSI > 0.25)")

# Save history
with open(FL_HISTORY_PATH, 'wb') as f:
    pickle.dump(history, f)

print("\n✓ Phase 3 complete.")
print("  Federated simulation done — all 3 banks trained collaboratively")
print("  without sharing raw transaction data.")
print("\n  Next step for real deployment: replace Flower simulation")
print("  with 3 separate machines connected via ngrok (see phase4_ngrok.md)")
