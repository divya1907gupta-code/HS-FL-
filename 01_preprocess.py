"""
Stage 1 - Preprocessing NF-BoT-IoT-v2 into per-device FL clients.

Design decisions (documented for the paper's methodology section):
- 10 clients = 10 real device IP addresses with >=500 flows in the sample.
  This gives GENUINE non-IID structure (not synthetic Dirichlet):
    - 4 "bot" devices (.147-.150): traffic dominated by DDoS/DoS/Reconnaissance
    - 6 "normal" devices: traffic dominated by Benign, with sparse anomalies
- Features: drop IP addresses (used only to define clients -> would leak
  client identity trivially if kept as a feature) and L4 ports (device/
  session-specific, do not generalize). Keep the 39 remaining NetFlow
  statistical features.
- Target: multiclass Attack label (Benign, DoS, DDoS, Reconnaissance, Theft)
  -> lets us report Macro-F1 meaningfully under real class imbalance.
- Scaler: StandardScaler fit ONLY on pooled training data (no test leakage).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json, os

np.random.seed(42)

RAW = "/home/claude/data/bot_iot_sample_raw.csv"
OUT_DIR = "/home/claude/pipeline/clients"
os.makedirs(OUT_DIR, exist_ok=True)

CLIENT_IPS = [
    "192.168.100.148", "192.168.100.147", "192.168.100.149", "192.168.100.150",
    "192.168.100.3", "192.168.100.6", "192.168.100.7", "192.168.100.5",
    "192.168.100.46", "192.168.100.55",
]

DROP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "Label"]

print("Loading sampled CSV...")
df = pd.read_csv(RAW)
df = df[df["IPV4_SRC_ADDR"].isin(CLIENT_IPS)].reset_index(drop=True)
print("Rows after restricting to 10 device clients:", len(df))

client_col = df["IPV4_SRC_ADDR"].copy()
attack_col = df["Attack"].copy()

classes = sorted(attack_col.unique().tolist())
class_to_idx = {c: i for i, c in enumerate(classes)}
print("Classes:", class_to_idx)

y_all = attack_col.map(class_to_idx).values.astype(np.int64)

X_df = df.drop(columns=DROP_COLS + ["Attack"])
feature_names = X_df.columns.tolist()
print("Num features:", len(feature_names))
X_all = X_df.values.astype(np.float64)

# ---- global train/test split, stratified by class, done WITHIN each client ----
train_idx_all, test_idx_all = [], []
for ip in CLIENT_IPS:
    idx = np.where(client_col.values == ip)[0]
    y_local = y_all[idx]
    # stratify only if every class has >=2 samples for that client
    vals, counts = np.unique(y_local, return_counts=True)
    can_stratify = np.all(counts >= 2) and len(vals) > 1
    tr, te = train_test_split(
        idx, test_size=0.2, random_state=42,
        stratify=y_local if can_stratify else None
    )
    train_idx_all.append(tr)
    test_idx_all.append(te)

train_idx_all = np.concatenate(train_idx_all)
test_idx_all = np.concatenate(test_idx_all)

scaler = StandardScaler()
scaler.fit(X_all[train_idx_all])

X_all = scaler.transform(X_all)
# clip extreme outlier flow stats after scaling for numerical stability
X_all = np.clip(X_all, -8, 8).astype(np.float32)

meta = {"classes": classes, "class_to_idx": class_to_idx,
        "feature_names": feature_names, "clients": []}

for ip in CLIENT_IPS:
    idx_client = np.where(client_col.values == ip)[0]
    tr = np.intersect1d(idx_client, train_idx_all)
    te = np.intersect1d(idx_client, test_idx_all)
    Xtr, ytr = X_all[tr], y_all[tr]
    Xte, yte = X_all[te], y_all[te]
    fname = ip.replace(".", "_")
    np.savez_compressed(f"{OUT_DIR}/client_{fname}.npz",
                         Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    vals, counts = np.unique(ytr, return_counts=True)
    dist = {classes[v]: int(c) for v, c in zip(vals, counts)}
    meta["clients"].append({"ip": ip, "n_train": len(tr), "n_test": len(te),
                             "train_class_dist": dist})
    print(ip, "train:", len(tr), "test:", len(te), "dist:", dist)

with open(f"{OUT_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nDone. Saved", len(CLIENT_IPS), "client files to", OUT_DIR)
print("Total train:", len(train_idx_all), "Total test:", len(test_idx_all))
