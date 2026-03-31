# ============================================================
#  PHASE 1 — Preprocessing
#  Run this on Kaggle Notebooks (dataset already mounted)
#  Path: /kaggle/input/ieee-fraud-detection/
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def downcast_numeric(df):
    """Reduce DataFrame memory usage by downcasting numeric dtypes."""
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols = df.select_dtypes(include=['int64']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype(np.float32)
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype(np.int32)
    return df


def quick_balance(X, y, max_rows=250000, target_fraud_ratio=0.20, seed=42):
    """Fast, memory-safe balancing for local runs."""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y)

    fraud_idx = np.where(y_arr == 1)[0]
    legit_idx = np.where(y_arr == 0)[0]

    if len(fraud_idx) == 0 or len(legit_idx) == 0:
        return X, y_arr

    max_rows = min(max_rows, len(y_arr))
    target_fraud = int(max_rows * target_fraud_ratio)
    target_legit = max_rows - target_fraud

    sampled_fraud = rng.choice(
        fraud_idx,
        size=target_fraud,
        replace=(len(fraud_idx) < target_fraud)
    )
    sampled_legit = rng.choice(
        legit_idx,
        size=min(target_legit, len(legit_idx)),
        replace=False
    )

    sampled_idx = np.concatenate([sampled_fraud, sampled_legit])
    rng.shuffle(sampled_idx)

    X_bal = X.iloc[sampled_idx].reset_index(drop=True)
    y_bal = y_arr[sampled_idx]
    return X_bal, y_bal

# ── 1. Load Data ─────────────────────────────────────────────
print("Loading data...")

# Locate dataset files: prefer Kaggle path, else search ./data recursively
kaggle_path = '/kaggle/input/ieee-fraud-detection'
required_files = [
    'train_transaction.csv',
    'train_identity.csv',
    'test_transaction.csv',
    'test_identity.csv'
]

if os.path.exists(os.path.join(kaggle_path, required_files[0])):
    BASE = kaggle_path + os.sep
    TRAIN_TXN_PATH = TRAIN_ID_PATH = TEST_TXN_PATH = TEST_ID_PATH = None
else:
    # Search ./data recursively for the required files
    found = {}
    for root, _, files in os.walk('./data'):
        for f in files:
            if f in required_files:
                found[f] = os.path.join(root, f)

    # If all required files are found in arbitrary subfolders, use explicit paths
    if len(found) == len(required_files):
        TRAIN_TXN_PATH = found['train_transaction.csv']
        TRAIN_ID_PATH  = found['train_identity.csv']
        TEST_TXN_PATH  = found['test_transaction.csv']
        TEST_ID_PATH   = found['test_identity.csv']
        BASE = None
    # Fallback: files directly under ./data/
    elif all(os.path.exists(os.path.join('./data', f)) for f in required_files):
        BASE = './data' + os.sep
        TRAIN_TXN_PATH = TRAIN_ID_PATH = TEST_TXN_PATH = TEST_ID_PATH = None
    else:
        raise FileNotFoundError(
            "Dataset not found. Place IEEE files in ./data/ (or subfolders) or run on Kaggle.\n"
            "Expected files: train_transaction.csv, train_identity.csv, test_transaction.csv, test_identity.csv"
        )

if BASE is not None:
    train_txn   = pd.read_csv(os.path.join(BASE, 'train_transaction.csv'))
    train_id    = pd.read_csv(os.path.join(BASE, 'train_identity.csv'))
    test_txn    = pd.read_csv(os.path.join(BASE, 'test_transaction.csv'))
    test_id     = pd.read_csv(os.path.join(BASE, 'test_identity.csv'))
else:
    train_txn   = pd.read_csv(TRAIN_TXN_PATH)
    train_id    = pd.read_csv(TRAIN_ID_PATH)
    test_txn    = pd.read_csv(TEST_TXN_PATH)
    test_id     = pd.read_csv(TEST_ID_PATH)

# Merge transaction + identity on TransactionID
train = train_txn.merge(train_id, on='TransactionID', how='left')
test  = test_txn.merge(test_id,  on='TransactionID', how='left')

print(f"Train shape: {train.shape} | Fraud rate: {train['isFraud'].mean()*100:.2f}%")
print(f"Test shape:  {test.shape}")

# Keep chronological order before splitting so features and labels stay aligned.
train = train.sort_values('TransactionDT').reset_index(drop=True)
y     = train['isFraud'].values
train = train.drop(columns=['isFraud'])

# ── 3. Missing Value Handling ─────────────────────────────────
print("\nHandling missing values...")

# Count nulls per column
null_rates = train.isnull().mean()
print(f"  Columns with >90% nulls: {(null_rates > 0.9).sum()}")
print(f"  Columns with any nulls:  {(null_rates > 0).sum()}")

# Strategy: replace NaN with -999 (sentinel outside normal range)
# This is preferred over mean imputation for tree/GNN models
# because it preserves the "missingness" as a signal
train = train.fillna(-999)
test  = test.fillna(-999)

train = downcast_numeric(train)
test = downcast_numeric(test)

# ── 4. Encode Categorical Features ───────────────────────────
print("\nEncoding categoricals...")

cat_cols = [c for c in train.columns
            if train[c].dtype == 'object']
print(f"  Categorical columns found: {len(cat_cols)}")

le = LabelEncoder()
for col in cat_cols:
    # Ensure column exists in test; if missing, create placeholder so LabelEncoder fits
    if col not in test.columns:
        test[col] = '-999'
    # Fit on combined train+test to avoid unseen labels
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

# ── 5. Feature Engineering ────────────────────────────────────
print("\nEngineering features...")

# Create UID (unique client fingerprint) from card + address + email
# This becomes the node identifier in the graph
train['uid'] = (train['card1'].astype(str) + '_' +
                train['card2'].astype(str) + '_' +
                train['addr1'].astype(str))
test['uid']  = (test['card1'].astype(str)  + '_' +
                test['card2'].astype(str)  + '_' +
                test['addr1'].astype(str))

# Transaction velocity: count transactions per uid in last N rows
# Proxy for "burst" fraud behaviour
uid_counts = train['uid'].value_counts().to_dict()
train['uid_count'] = train['uid'].map(uid_counts).fillna(1)
test['uid_count']  = test['uid'].map(uid_counts).fillna(1)

# Log-transform TransactionAmt to handle skew
train['TransactionAmt_log'] = np.log1p(train['TransactionAmt'])
test['TransactionAmt_log']  = np.log1p(test['TransactionAmt'])

print(f"  uid unique values (train): {train['uid'].nunique()}")

# ── 6. Chronological Split ────────────────────────────────────
print("\nCreating chronological split...")

n        = len(train)
n_train  = int(n * 0.70)
n_val    = int(n * 0.15)
# remaining 15% = test

X_train = train.iloc[:n_train]
y_train = y[:n_train]

X_val   = train.iloc[n_train:n_train+n_val]
y_val   = y[n_train:n_train+n_val]

X_test  = train.iloc[n_train+n_val:]
y_test  = y[n_train+n_val:]

print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"  Train fraud rate: {y_train.mean()*100:.2f}%")

# ── 7. SMOTE — Balance Classes ────────────────────────────────
print("\nBalancing training set...")

# Drop uid (string) before SMOTE — add back after
uid_train = X_train['uid'].values
X_train_num = X_train.drop(columns=['uid'])

USE_SMOTE = os.getenv('USE_SMOTE', '0') == '1'
if USE_SMOTE:
    # Full-data SMOTE is extremely slow and RAM-heavy locally, so cap size.
    smote_cap = min(len(X_train_num), 120000)
    sample_idx = np.random.choice(len(X_train_num), size=smote_cap, replace=False)
    X_sm = X_train_num.iloc[sample_idx].reset_index(drop=True)
    y_sm = y_train[sample_idx]
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_sm, y_sm)
    print(f"  Mode: SMOTE (capped at {smote_cap:,} rows)")
else:
    X_train_bal, y_train_bal = quick_balance(
        X_train_num,
        y_train,
        max_rows=250000,
        target_fraud_ratio=0.20,
        seed=42
    )
    print("  Mode: quick_balance (recommended for local CPU runs)")

print(f"  Before balancing: {len(X_train_num)} samples | "
      f"Fraud: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"  After  balancing: {len(X_train_bal)} samples | "
      f"Fraud: {y_train_bal.sum()} ({y_train_bal.mean()*100:.2f}%)")

# ── 8. Build Graph Structure ──────────────────────────────────
print("\nBuilding transaction graph...")

# Nodes: unique entities (cards, devices, merchants)
# Edges: transactions between them
# We build an edge list: (src_node, dst_node, features, label)

def build_edge_list(df, labels=None):
    """
    Each row = one transaction = one edge
    src = card1 (account node)
    dst = DeviceInfo encoded (device node)
    edge features = transaction features
    """
    edges = []
    feature_cols = [c for c in df.columns
                    if c not in ['TransactionID', 'uid',
                                 'TransactionDT']]

    src_vals = pd.to_numeric(df['card1'], errors='coerce').fillna(-999).to_numpy(dtype=np.int32)
    dst_vals = pd.to_numeric(df['DeviceInfo'], errors='coerce').fillna(-999).to_numpy(dtype=np.int32) + 100000

    feat_df = df[feature_cols].copy()
    obj_cols = feat_df.select_dtypes(exclude=[np.number]).columns
    for c in obj_cols:
        feat_df[c] = pd.factorize(feat_df[c].astype(str))[0]
    feat_vals = feat_df.to_numpy(dtype=np.float32)

    for idx in range(len(df)):
        src = int(src_vals[idx])
        dst = int(dst_vals[idx])

        feats = feat_vals[idx]
        label = labels[idx] if labels is not None else -1

        edges.append({
            'src':   src,
            'dst':   dst,
            'feats': feats,
            'label': label
        })

    return edges

print("  Building training graph edges...")
# Use a sample for speed — full dataset on GPU Kaggle session
sample_idx = np.random.choice(len(X_train_bal),
                               size=min(50000, len(X_train_bal)),
                               replace=False)
X_sample = X_train_bal.iloc[sample_idx].reset_index(drop=True)
y_sample = y_train_bal[sample_idx]

edge_list = build_edge_list(X_sample, y_sample)
print(f"  Graph edges (sample): {len(edge_list)}")

# Unique nodes
all_nodes = set()
for e in edge_list:
    all_nodes.add(e['src'])
    all_nodes.add(e['dst'])
print(f"  Graph nodes: {len(all_nodes)}")

# ── 9. Save Artifacts ─────────────────────────────────────────
print("\nSaving preprocessed artifacts...")

os.makedirs('artifacts', exist_ok=True)

X_val_num = X_val.drop(columns=['uid'])
X_test_num = X_test.drop(columns=['uid'])

pickle.dump({
    'X_train': X_train_bal,
    'y_train': y_train_bal,
    'X_val':   X_val_num.values,
    'y_val':   y_val,
    'X_test':  X_test_num.values,
    'y_test':  y_test,
    'edge_list': edge_list,
    'feature_cols': [c for c in X_train_bal.columns]
}, open('artifacts/preprocessed.pkl', 'wb'))

print("\n✓ Phase 1 complete. Artifacts saved to /kaggle/working/artifacts/")
print("  Next: run phase2_model.py")
