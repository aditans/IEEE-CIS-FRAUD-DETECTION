# ============================================================
#  PHASE 3 — Federated Learning Simulation (Flower)
#  Simulates 3 banks training collaboratively
#  without sharing raw data — all in one Kaggle/local session
# ============================================================

import os
import pickle
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
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
FED_METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'federated_metrics.pkl')
PSI_PATH = os.path.join(ARTIFACTS_DIR, 'psi_details.pkl')


class FraudSTGNN(nn.Module):
    """Same model capacity as phase2 for fairer comparison."""

    def __init__(self, in_dim, hidden_dim=128, out_dim=64,
                 gru_hidden=32, dropout=0.5):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.h1_proj = nn.Linear(hidden_dim, out_dim)
        self.gru = nn.GRU(out_dim, gru_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x, edge_index):
        h1 = F.normalize(F.relu(self.bn1(self.sage1(x, edge_index))), p=2, dim=1)
        h1 = self.dropout(h1)
        h1_proj = F.normalize(self.h1_proj(h1), p=2, dim=1)

        h2 = F.normalize(F.relu(self.bn2(self.sage2(h1, edge_index))), p=2, dim=1)
        h2 = self.dropout(h2)

        seq = torch.stack([h1_proj, h2], dim=1)
        _, h_n = self.gru(seq)
        out = self.classifier(h_n.squeeze(0))
        return out.squeeze(1)


def build_pyg_graph(edges_subset, train_ratio: float = 0.8) -> Data:
    """Convert edge subset into a node-level graph and train/val masks."""
    if len(edges_subset) == 0:
        raise ValueError('edges_subset is empty')

    all_ids = set()
    for e in edges_subset:
        all_ids.add(e['src'])
        all_ids.add(e['dst'])

    id_map = {nid: i for i, nid in enumerate(sorted(all_ids))}
    n_nodes = len(id_map)
    feat_dim = edges_subset[0]['feats'].shape[0]

    node_feats = np.zeros((n_nodes, feat_dim), dtype=np.float32)
    node_counts = np.zeros(n_nodes, dtype=np.int32)
    node_labels = np.zeros(n_nodes, dtype=np.float32)
    src_list, dst_list = [], []

    for e in edges_subset:
        s, d = id_map[e['src']], id_map[e['dst']]
        src_list.append(s)
        dst_list.append(d)

        node_feats[s] += e['feats']
        node_counts[s] += 1
        node_feats[d] += e['feats']
        node_counts[d] += 1

        if e['label'] == 1:
            node_labels[s] = 1
            node_labels[d] = 1

    node_counts = np.maximum(node_counts, 1)
    node_feats = node_feats / node_counts[:, None]

    edge_index = torch.tensor([src_list + dst_list, dst_list + src_list], dtype=torch.long)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)

    if train_ratio <= 0.0:
        val_mask[:] = True
    else:
        split = int(n_nodes * train_ratio)
        train_mask[:split] = True
        val_mask[split:] = True

    return Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(node_labels, dtype=torch.float),
        train_mask=train_mask,
        val_mask=val_mask,
    )


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute consistent metric dictionary including confusion matrix and FPR."""
    y_pred = (y_prob > threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    beta_sq = 4.0
    f2 = ((1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
          if (beta_sq * precision + recall) > 0 else 0.0)

    if len(np.unique(y_true)) > 1:
        auc_pr = float(average_precision_score(y_true, y_prob))
        auc_roc = float(roc_auc_score(y_true, y_prob))
    else:
        auc_pr = 0.0
        auc_roc = 0.0

    return {
        'auc_pr': auc_pr,
        'auc_roc': auc_roc,
        'f2': float(f2),
        'recall': float(recall),
        'precision': float(precision),
        'fpr': float(fpr),
        'tn': float(tn),
        'fp': float(fp),
        'fn': float(fn),
        'tp': float(tp),
    }


def evaluate_on_mask(model: nn.Module, graph: Data, criterion: nn.Module, mask_name: str = 'val_mask') -> Tuple[float, Dict[str, float]]:
    """Evaluate a model on a graph mask."""
    model.eval()
    mask = getattr(graph, mask_name)
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        probs = torch.sigmoid(logits)
        loss = criterion(logits[mask], graph.y[mask])

    y_true = np.asarray(graph.y[mask].detach().cpu().tolist(), dtype=np.float32)
    y_prob = np.asarray(probs[mask].detach().cpu().tolist(), dtype=np.float32)
    metrics = compute_binary_metrics(y_true, y_prob)
    return float(loss.item()), metrics


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile-based bins for robustness."""
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)

    q = np.linspace(0, 1, bins + 1)
    breakpoints = np.quantile(expected, q)
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        breakpoints = np.linspace(float(np.min(expected)), float(np.max(expected)), bins + 1)

    exp_pct = np.histogram(expected, breakpoints)[0] / max(len(expected), 1)
    act_pct = np.histogram(actual, breakpoints)[0] / max(len(actual), 1)

    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


print('Partitioning data into 3 federated clients (banks)...')
if not os.path.exists(PREPROCESSED_PATH):
    raise FileNotFoundError(
        f"Missing preprocessed data at {PREPROCESSED_PATH}. "
        'Run phase1_preprocessing.py first.'
    )

with open(PREPROCESSED_PATH, 'rb') as f:
    artifacts = pickle.load(f)
edge_list = artifacts['edge_list']

np.random.seed(42)
perm_idx = np.random.permutation(len(edge_list))

# Hold out 20% edges for global server-side evaluation to avoid comparing
# weighted client AUC-PR against a single centralised test score.
n_eval = max(1, int(0.2 * len(edge_list)))
eval_idx = perm_idx[-n_eval:]
fed_idx = perm_idx[:-n_eval]

client_splits = np.array_split(fed_idx, 3)
bank_graphs: List[Data] = []

for bank_id, split_idx in enumerate(client_splits):
    subset = [edge_list[i] for i in split_idx]
    g = build_pyg_graph(subset, train_ratio=0.8)
    bank_graphs.append(g)
    fraud_rate = g.y.mean().item() * 100
    print(f"  Bank {bank_id+1}: {g.num_nodes} nodes | {g.num_edges} edges | Fraud rate: {fraud_rate:.2f}%")

global_eval_edges = [edge_list[i] for i in eval_idx]
global_eval_graph = build_pyg_graph(global_eval_edges, train_ratio=0.0).to(DEVICE)
print(f"  Global eval graph: {global_eval_graph.num_nodes} nodes | {global_eval_graph.num_edges} edges")

IN_DIM = bank_graphs[0].num_node_features


class FraudDetectionClient(fl.client.NumPyClient):
    """One Flower client = one bank with private local graph data."""

    def __init__(self, client_id: int, graph: Data):
        self.client_id = client_id
        self.graph = graph.to(DEVICE)
        self.model = FraudSTGNN(IN_DIM).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE))

    def get_parameters(self, config) -> List[np.ndarray]:
        return [np.asarray(val.detach().cpu().tolist()) for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = OrderedDict()
        for (k, old_tensor), v in zip(self.model.state_dict().items(), parameters):
            v_arr = np.asarray(v)
            state_dict[k] = torch.tensor(v_arr.tolist(), dtype=old_tensor.dtype, device=old_tensor.device)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        local_epochs = config.get('local_epochs', 5)
        lr = config.get('lr', 1e-3)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        self.model.train()
        for _ in range(local_epochs):
            optimizer.zero_grad()
            logits = self.model(self.graph.x, self.graph.edge_index)
            loss = self.criterion(logits[self.graph.train_mask], self.graph.y[self.graph.train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

        return (
            self.get_parameters(config={}),
            int(self.graph.train_mask.sum()),
            {'loss': float(loss.item()), 'client_id': float(self.client_id)}
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = evaluate_on_mask(self.model, self.graph, self.criterion, mask_name='val_mask')
        return (loss, int(self.graph.val_mask.sum()), metrics)


def weighted_average(metrics):
    """Aggregate client metrics by summing confusion counts and deriving rates."""
    total = float(sum(num for num, _ in metrics))

    # AUC-PR is non-linear; keep weighted average only as a diagnostic trend.
    auc_pr_weighted = float(sum(num * m.get('auc_pr', 0.0) for num, m in metrics) / total) if total > 0 else 0.0

    tp = float(sum(m.get('tp', 0.0) for _, m in metrics))
    fp = float(sum(m.get('fp', 0.0) for _, m in metrics))
    fn = float(sum(m.get('fn', 0.0) for _, m in metrics))
    tn = float(sum(m.get('tn', 0.0) for _, m in metrics))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    beta_sq = 4.0
    f2 = ((1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
          if (beta_sq * precision + recall) > 0 else 0.0)

    return {
        'auc_pr': auc_pr_weighted,
        'auc_pr_weighted': auc_pr_weighted,
        'f2': float(f2),
        'recall': float(recall),
        'precision': float(precision),
        'fpr': float(fpr),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }


def build_model_from_parameters(parameters) -> FraudSTGNN:
    # Flower may pass either a Parameters object or ndarrays depending on version.
    if isinstance(parameters, list):
        ndarrays = parameters
    else:
        ndarrays = fl.common.parameters_to_ndarrays(parameters)
    model = FraudSTGNN(IN_DIM).to(DEVICE)

    state_dict = OrderedDict()
    for (k, old_tensor), v in zip(model.state_dict().items(), ndarrays):
        v_arr = np.asarray(v)
        state_dict[k] = torch.tensor(v_arr.tolist(), dtype=old_tensor.dtype, device=old_tensor.device)

    model.load_state_dict(state_dict, strict=True)
    return model


def server_evaluate(server_round: int, parameters: fl.common.Parameters, config: Dict[str, float]):
    """Server-side global evaluation on one shared holdout graph."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE))
    model = build_model_from_parameters(parameters)
    loss, metrics = evaluate_on_mask(model, global_eval_graph, criterion, mask_name='val_mask')

    out = {
        'auc_pr_global': float(metrics['auc_pr']),
        'auc_roc_global': float(metrics['auc_roc']),
        'f2_global': float(metrics['f2']),
        'recall_global': float(metrics['recall']),
        'precision_global': float(metrics['precision']),
        'fpr_global': float(metrics['fpr']),
        'tn_global': float(metrics['tn']),
        'fp_global': float(metrics['fp']),
        'fn_global': float(metrics['fn']),
        'tp_global': float(metrics['tp']),
    }
    return loss, out


print('\nStarting Flower federated simulation...')
print('  Clients (banks): 3')
print('  Rounds: 20')
print('  Local epochs per round: 5')
print()


def client_fn(cid: str) -> fl.client.Client:
    client_id = int(cid)
    return FraudDetectionClient(client_id=client_id, graph=bank_graphs[client_id]).to_client()


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average,
    evaluate_fn=server_evaluate,
    on_fit_config_fn=lambda rnd: {
        'local_epochs': 5,
        'lr': max(1e-4, 1e-3 * (0.95 ** rnd)),
    }
)

history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=3,
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
    client_resources={
        'num_cpus': 1,
        'num_gpus': 0.3 if torch.cuda.is_available() else 0,
    }
)

print('\n' + '='*60)
print('FEDERATED LEARNING RESULTS (ROUND 20)')
print('='*60)

fed_final = {}
if history.metrics_distributed:
    for metric, values in history.metrics_distributed.items():
        if values:
            fed_final[metric] = float(values[-1][1])

global_final = {}
if history.metrics_centralized:
    for metric, values in history.metrics_centralized.items():
        if values:
            global_final[metric] = float(values[-1][1])

if fed_final:
    print('\nClient-aggregated validation metrics:')
    for k in ['auc_pr', 'f2', 'recall', 'precision', 'fpr']:
        if k in fed_final:
            print(f"  {k:12s}: {fed_final[k]:.4f}")

if global_final:
    print('\nServer global holdout metrics (recommended for comparison):')
    for k in ['auc_pr_global', 'auc_roc_global', 'f2_global', 'recall_global', 'precision_global', 'fpr_global']:
        if k in global_final:
            print(f"  {k:18s}: {global_final[k]:.4f}")
    cm = (
        int(global_final.get('tn_global', 0.0)),
        int(global_final.get('fp_global', 0.0)),
        int(global_final.get('fn_global', 0.0)),
        int(global_final.get('tp_global', 0.0)),
    )
    print(f"  confusion_matrix: TN={cm[0]} FP={cm[1]} FN={cm[2]} TP={cm[3]}")

try:
    with open(CENTRALISED_PATH, 'rb') as f:
        central = pickle.load(f)

    print('\n' + '-'*60)
    print('COMPARISON: CENTRALISED VS FEDERATED (GLOBAL HOLDOUT)')
    print('-'*60)
    print(f"{'Metric':<18} {'Centralised':>14} {'Federated':>14} {'Delta':>12}")

    compare_map = {
        'auc_pr': 'auc_pr_global',
        'auc_roc': 'auc_roc_global',
        'f2': 'f2_global',
        'recall': 'recall_global',
        'precision': 'precision_global',
        'fpr': 'fpr_global',
    }

    for c_key, f_key in compare_map.items():
        c_val = float(central.get(c_key, 0.0))
        f_val = float(global_final.get(f_key, 0.0))
        delta = f_val - c_val
        print(f"{c_key:<18} {c_val:>14.4f} {f_val:>14.4f} {delta:>+12.4f}")
except FileNotFoundError:
    print('\n(Run phase2_model.py first for centralised comparison.)')


print('\n' + '-'*60)
print('DRIFT DETECTION (PSI) — BANK 1 VS BANK 2')
print('-'*60)

feats_bank1 = np.asarray(bank_graphs[0].x[:, 0].detach().cpu().tolist(), dtype=np.float32)
feats_bank2 = np.asarray(bank_graphs[1].x[:, 0].detach().cpu().tolist(), dtype=np.float32)

psi = compute_psi(feats_bank1, feats_bank2, bins=10)
print(f"  PSI score: {psi:.4f}")
if psi < 0.10:
    print('  Status: STABLE (no retraining trigger)')
elif psi < 0.25:
    print('  Status: MONITOR (moderate drift)')
else:
    print('  Status: RETRAIN (significant drift)')

with open(PSI_PATH, 'wb') as f:
    pickle.dump({
        'bank1_feature0': feats_bank1,
        'bank2_feature0': feats_bank2,
        'psi': float(psi),
    }, f)

with open(FL_HISTORY_PATH, 'wb') as f:
    pickle.dump(history, f)

with open(FED_METRICS_PATH, 'wb') as f:
    pickle.dump({
        'distributed_final': fed_final,
        'global_eval_final': global_final,
        'notes': {
            'auc_pr_distributed_is_weighted_client_average': True,
            'recommended_for_comparison': 'global_eval_final',
        },
    }, f)

print('\nSaved:')
print(f'  {FL_HISTORY_PATH}')
print(f'  {FED_METRICS_PATH}')
print(f'  {PSI_PATH}')
print('\nPhase 3 complete.')
