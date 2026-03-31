# ============================================================
#  PHASE 2 — GraphSAGE + GRU Model Training
#  Run AFTER phase1_preprocessing.py
#  Enable GPU on Kaggle: Settings → Accelerator → GPU T4
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
import numpy as np
import pickle
import os
from sklearn.metrics import (roc_auc_score,
                              average_precision_score,
                              precision_score, recall_score)
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
# Ensure deterministic CUDA behavior for reproducibility if needed
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
BEST_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best_model.pt')
METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'centralised_metrics.pkl')


# ── 1. Load Preprocessed Data ────────────────────────────────
print("Loading preprocessed artifacts...")
if not os.path.exists(PREPROCESSED_PATH):
    raise FileNotFoundError(
        f"Missing preprocessed data at {PREPROCESSED_PATH}. "
        "Run phase1_preprocessing.py first."
    )

with open(PREPROCESSED_PATH, 'rb') as f:
    data = pickle.load(f)

X_train    = data['X_train']
y_train    = data['y_train']
X_val      = data['X_val']
y_val      = data['y_val']
X_test     = data['X_test']
y_test     = data['y_test']
edge_list  = data['edge_list']

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── 2. Build PyTorch Geometric Graph ─────────────────────────
print("\nBuilding PyG graph...")

# Map raw node IDs to consecutive integers
all_node_ids = set()
for e in edge_list:
    all_node_ids.add(e['src'])
    all_node_ids.add(e['dst'])

node_id_map = {nid: idx for idx, nid in enumerate(sorted(all_node_ids))}
num_nodes   = len(node_id_map)
feat_dim    = edge_list[0]['feats'].shape[0]

print(f"  Nodes: {num_nodes} | Feature dim: {feat_dim}")

# Node features: aggregate edge features per node (mean)
node_feats  = np.zeros((num_nodes, feat_dim), dtype=np.float32)
node_counts = np.zeros(num_nodes, dtype=np.int32)

edge_src, edge_dst, edge_labels = [], [], []

for e in edge_list:
    s = node_id_map[e['src']]
    d = node_id_map[e['dst']]
    edge_src.append(s)
    edge_dst.append(d)
    edge_labels.append(e['label'])
    # Accumulate features into both endpoint nodes
    node_feats[s]  += e['feats']
    node_counts[s] += 1
    node_feats[d]  += e['feats']
    node_counts[d] += 1

# Average
node_counts = np.maximum(node_counts, 1)
node_feats  = node_feats / node_counts[:, None]

# Build bidirectional edge index (undirected graph)
edge_index = torch.tensor(
    [edge_src + edge_dst,
     edge_dst + edge_src], dtype=torch.long
)

# Node labels: fraud if ANY connected transaction is fraud
node_labels = np.zeros(num_nodes, dtype=np.float32)
for i, (s, d, lbl) in enumerate(zip(edge_src, edge_dst, edge_labels)):
    if lbl == 1:
        node_labels[s] = 1
        node_labels[d] = 1

x = torch.tensor(node_feats, dtype=torch.float)
y = torch.tensor(node_labels, dtype=torch.float)

# Train/val/test masks (chronological — first 70% nodes as train)
n           = num_nodes
train_mask  = torch.zeros(n, dtype=torch.bool)
val_mask    = torch.zeros(n, dtype=torch.bool)
test_mask   = torch.zeros(n, dtype=torch.bool)

train_mask[:int(n * 0.70)]                          = True
val_mask[int(n * 0.70):int(n * 0.85)]               = True
test_mask[int(n * 0.85):]                            = True

graph = Data(
    x          = x,
    edge_index = edge_index,
    y          = y,
    train_mask = train_mask,
    val_mask   = val_mask,
    test_mask  = test_mask
).to(DEVICE)

print(f"  Graph built: {graph.num_nodes} nodes, "
      f"{graph.num_edges} edges")
print(f"  Fraud node rate: {y.mean()*100:.2f}%")

# ── 3. Define Hybrid GraphSAGE + GRU Model ───────────────────
class FraudSTGNN(nn.Module):
    """
    Spatial:  2-layer GraphSAGE (inductive, drift-robust)
    Temporal: GRU on top of node embeddings
    Output:   Binary fraud/legitimate classification
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=64,
                 gru_hidden=32, dropout=0.5):
        super().__init__()

        # ── Spatial: GraphSAGE layers ──────────────────────
        # Layer 1: aggregate 1-hop neighbourhood
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        # Layer 2: aggregate 2-hop neighbourhood
        self.sage2 = SAGEConv(hidden_dim, out_dim)

        # Batch norm for training stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        # Project h1 to out_dim so h1 and h2 can be stacked for GRU.
        self.h1_proj = nn.Linear(hidden_dim, out_dim)

        # ── Temporal: GRU ─────────────────────────────────
        # Treats the 2 GraphSAGE layers as a "sequence"
        # h¹_v → h²_v fed as timesteps into GRU
        # reset gate: r_t = σ(W_r x_t + U_r h_{t-1})
        # update gate: z_t = σ(W_z x_t + U_z h_{t-1})
        # candidate:  h̃_t = tanh(W_h x_t + U_h(r_t ⊙ h_{t-1}))
        # final:       h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t
        self.gru = nn.GRU(
            input_size  = out_dim,
            hidden_size = gru_hidden,
            num_layers  = 1,
            batch_first = True
        )

        self.dropout = nn.Dropout(dropout)

        # ── Classifier head ────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x, edge_index):
        # ── Spatial pass ──────────────────────────────────
        # Layer 1: h¹_v = σ(W¹ · [h⁰_v ‖ AGGREGATE({h⁰_u})])
        h1 = self.sage1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        # ℓ2 normalise
        h1 = F.normalize(h1, p=2, dim=1)
        h1_proj = self.h1_proj(h1)
        h1_proj = F.normalize(h1_proj, p=2, dim=1)

        # Layer 2: h²_v = σ(W² · [h¹_v ‖ AGGREGATE({h¹_u})])
        h2 = self.sage2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)
        h2 = F.normalize(h2, p=2, dim=1)

        # ── Temporal pass (GRU) ───────────────────────────
        # Stack h1 and h2 as a 2-timestep sequence per node
        # Shape: (num_nodes, seq_len=2, out_dim)
        seq      = torch.stack([h1_proj, h2], dim=1)
        _, h_n   = self.gru(seq)   # h_n: (1, num_nodes, gru_hidden)
        temporal = h_n.squeeze(0)  # (num_nodes, gru_hidden)

        # ── Classification ────────────────────────────────
        out = self.classifier(temporal)
        return out.squeeze(1)


# ── 4. Training Setup ─────────────────────────────────────────
print("\nInitialising model...")

model = FraudSTGNN(
    in_dim     = graph.num_node_features,
    hidden_dim = 128,
    out_dim    = 64,
    gru_hidden = 32,
    dropout    = 0.5
).to(DEVICE)

print(f"  Model parameters: "
      f"{sum(p.numel() for p in model.parameters()):,}")

# Class-weighted loss to handle remaining imbalance after SMOTE
fraud_weight = torch.tensor([3.0]).to(DEVICE)  # weight fraud class higher
criterion    = nn.BCEWithLogitsLoss(pos_weight=fraud_weight)
optimizer    = torch.optim.Adam(model.parameters(),
                                lr=1e-3, weight_decay=1e-4)
scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.5
)

# ── 5. Evaluation Function ───────────────────────────────────
def evaluate(model, graph, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        probs  = torch.sigmoid(logits)

        # Convert through Python lists to avoid torch<->numpy ABI issues.
        y_true = np.asarray(graph.y[mask].detach().cpu().tolist(), dtype=np.float32)
        y_prob = np.asarray(probs[mask].detach().cpu().tolist(), dtype=np.float32)
        y_pred = (y_prob > 0.5).astype(int)

        # Guard against all-one-class in small masks
        if len(np.unique(y_true)) < 2:
            return {'auc_pr': 0, 'auc_roc': 0,
                    'f2': 0, 'recall': 0, 'precision': 0}

        # F2-score: weights recall 2x over precision
        # F2 = (1+beta^2) * P * R / (beta^2 * P + R), beta=2
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta_sq = 4.0  # beta=2
        f2 = ((1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
              if (beta_sq * precision + recall) > 0 else 0.0)

        return {
            'auc_pr':    average_precision_score(y_true, y_prob),
            'auc_roc':   roc_auc_score(y_true, y_prob),
            'f2':        f2,
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0)
        }

# ── 6. Training Loop ──────────────────────────────────────────
print("\nTraining...")
EPOCHS       = 100
best_auc_pr  = 0
best_epoch   = 0
patience     = 15
no_improve   = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(graph.x, graph.edge_index)
    loss   = criterion(logits[graph.train_mask],
                       graph.y[graph.train_mask])
    loss.backward()

    # Gradient clipping — important for GRU stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % 5 == 0:
        val_metrics = evaluate(model, graph, graph.val_mask)
        scheduler.step(val_metrics['auc_pr'])

        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
              f"Val AUC-PR: {val_metrics['auc_pr']:.4f} | "
              f"Val F2: {val_metrics['f2']:.4f} | "
              f"Val Recall: {val_metrics['recall']:.4f}")

        # Save best model
        if val_metrics['auc_pr'] > best_auc_pr:
            best_auc_pr = val_metrics['auc_pr']
            best_epoch  = epoch
            no_improve  = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best epoch: {best_epoch})")
            break

# ── 7. Final Evaluation on Test Set ──────────────────────────
print(f"\nLoading best model (epoch {best_epoch})...")
model.load_state_dict(
    torch.load(BEST_MODEL_PATH)
)

test_metrics = evaluate(model, graph, graph.test_mask)

print("\n" + "="*50)
print("FINAL TEST SET RESULTS")
print("="*50)
print(f"  AUC-PR    (primary):  {test_metrics['auc_pr']:.4f}")
print(f"  AUC-ROC:              {test_metrics['auc_roc']:.4f}")
print(f"  F2-Score  (primary):  {test_metrics['f2']:.4f}")
print(f"  Recall:               {test_metrics['recall']:.4f}")
print(f"  Precision:            {test_metrics['precision']:.4f}")
print("="*50)

# Save metrics for Phase 3 comparison
with open(METRICS_PATH, 'wb') as f:
    pickle.dump(test_metrics, f)
print("\n✓ Phase 2 complete. Best model saved.")
print("  Next: run phase3_federated.py")
