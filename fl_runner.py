import numpy as np, json, time
from sklearn.metrics import accuracy_score, f1_score
from fl_core import (MLP, local_train, adaptive_gradient_compression,
                      Paillier, secure_aggregate_fast, gradient_norm_filter,
                      direction_filter, behavioral_filter)

META = json.load(open("/home/claude/pipeline/clients/meta.json"))
CLASSES = META["classes"]
N_CLASSES = len(CLASSES)
IN_DIM = len(META["feature_names"])
CLIENT_IPS = [c["ip"] for c in META["clients"]]

_DATA_CACHE = None
def load_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        data = {}
        for ip in CLIENT_IPS:
            fname = ip.replace(".", "_")
            d = np.load(f"/home/claude/pipeline/clients/client_{fname}.npz")
            data[ip] = (d["Xtr"], d["ytr"], d["Xte"], d["yte"])
        _DATA_CACHE = data
    return _DATA_CACHE


def build_global_test():
    data = load_data()
    Xte = np.concatenate([data[ip][2] for ip in CLIENT_IPS], axis=0)
    yte = np.concatenate([data[ip][3] for ip in CLIENT_IPS], axis=0)
    return Xte, yte


FLOAT32_BYTES = 4


def run_federated(method="fedavg", T=15, local_epochs=1, batch_size=256, lr=0.05,
                   participation=1.0, malicious_fraction=0.0, malicious_kind="scale",
                   beta=1.5, key_bits=256, z_thresh=3.0, fedprox_mu=0.0, seed=0,
                   verbose=False):
    """
    method in {"fedavg","fedprox","compression_only","encryption_only","hsfl"}
    Returns dict with per-round history + final metrics + communication bytes.
    """
    data = load_data()
    Xte_global, yte_global = build_global_test()
    rng = np.random.default_rng(seed)

    use_compression = method in ("compression_only", "hsfl")
    use_encryption = method in ("encryption_only", "hsfl")
    use_threat_detect = method in ("hsfl",)
    use_fedprox = method == "fedprox"
    mu = fedprox_mu if use_fedprox else 0.0

    n_clients = len(CLIENT_IPS)
    n_malicious = int(round(malicious_fraction * n_clients))
    malicious_ids = set(rng.choice(n_clients, size=n_malicious, replace=False).tolist()) if n_malicious > 0 else set()

    model = MLP(IN_DIM, out_dim=N_CLASSES, seed=seed)
    global_flat = model.get_flat()

    paillier = Paillier(key_bits=key_bits, seed=seed) if use_encryption else None

    history = {"round": [], "acc": [], "macro_f1": [], "n_participants": [],
               "flagged_clients": []}
    client_behavior_history = {}  # persisted across rounds for behavioral_filter
    total_bytes_transmitted = 0
    total_bytes_baseline = 0  # what FedAvg would have sent (uncompressed, unencrypted)
    n_enc_ops_total = 0

    for t in range(T):
        # sample participating clients this round
        n_part = max(1, int(round(participation * n_clients)))
        participants = rng.choice(n_clients, size=n_part, replace=False)

        deltas = []
        raw_deltas = []
        flagged_this_round = []
        for ci in participants:
            ip = CLIENT_IPS[ci]
            Xtr, ytr, _, _ = data[ip]
            mal = None
            if ci in malicious_ids:
                mal = {"kind": malicious_kind, "factor": -5.0}
            d = local_train(global_flat, Xtr, ytr, IN_DIM, N_CLASSES,
                             epochs=local_epochs, batch_size=batch_size, lr=lr,
                             seed=seed * 100 + t * 10 + ci, fedprox_mu=mu, malicious=mal)
            raw_deltas.append(d)

            total_bytes_baseline += len(d) * FLOAT32_BYTES

            if use_compression:
                comp, mask, ratio = adaptive_gradient_compression(d, beta=beta)
                deltas.append(comp)
                total_bytes_transmitted += mask.sum() * (FLOAT32_BYTES + 4)  # value + index
            else:
                deltas.append(d)
                total_bytes_transmitted += len(d) * FLOAT32_BYTES

        # ---- threat detection (server-side, before aggregation) ----
        # IMPORTANT: run detection on the RAW (pre-compression, dense) updates,
        # not the AGC-compressed ones. AGC gives each client a different random
        # sparse support (~8% of coords each), so a coordinate-wise median across
        # compressed vectors has almost no coordinate overlap and collapses to
        # near-zero noise -- verified empirically to misclassify clients. The
        # dense raw update is what actually carries a meaningful direction.
        if use_threat_detect:
            if z_thresh >= 1e8:
                # explicit "detection disabled" ablation path
                keep_mask = np.ones(len(raw_deltas), dtype=bool)
            else:
                keep_mask, client_behavior_history = behavioral_filter(
                    list(participants), raw_deltas, client_behavior_history,
                    warmup_rounds=3, z_thresh=z_thresh, cos_thresh=0.3)
                # Safety cap: never exclude more than 40% of this round's
                # participants, even if flagged, so the server always keeps
                # enough legitimate signal to make progress and recover
                # (prevents the cascading-lockout failure mode).
                max_exclude = int(np.floor(0.4 * len(participants)))
                n_flagged = (~keep_mask).sum()
                if n_flagged > max_exclude:
                    # keep the (max_exclude) MOST anomalous excluded, re-admit the rest
                    norms_this_round = np.array([np.linalg.norm(d) for d in raw_deltas])
                    flagged_idx = np.where(~keep_mask)[0]
                    # rank flagged clients by how far their norm is from their own history mean
                    scores = []
                    for i in flagged_idx:
                        cid = participants[i]
                        hist = client_behavior_history.get(cid, {"norms": [norms_this_round[i]]})
                        mu = np.mean(hist["norms"][:-1]) if len(hist["norms"]) > 1 else norms_this_round[i]
                        scores.append(abs(norms_this_round[i] - mu))
                    order = flagged_idx[np.argsort(scores)[::-1]]
                    re_admit = order[max_exclude:]
                    keep_mask[re_admit] = True
            flagged_this_round = [int(participants[i]) for i in range(len(participants)) if not keep_mask[i]]
            deltas_kept = [d for d, k in zip(deltas, keep_mask) if k]
            if len(deltas_kept) == 0:
                deltas_kept = deltas  # safety net: don't stall training
        else:
            deltas_kept = deltas

        # ---- aggregation ----
        if use_encryption:
            agg_sum, n_ops, n_idx = secure_aggregate_fast(deltas_kept, paillier, scale=1e4)
            n_enc_ops_total += n_ops
            avg_delta = agg_sum / len(deltas_kept)
        else:
            avg_delta = np.mean(deltas_kept, axis=0)

        global_flat = global_flat + avg_delta

        # ---- evaluation ----
        model.set_flat(global_flat)
        preds = model.predict(Xte_global)
        acc = accuracy_score(yte_global, preds)
        mf1 = f1_score(yte_global, preds, average="macro", zero_division=0)

        history["round"].append(t)
        history["acc"].append(acc)
        history["macro_f1"].append(mf1)
        history["n_participants"].append(n_part)
        history["flagged_clients"].append(flagged_this_round)

        if verbose:
            print(f"[{method}] round {t+1}/{T} acc={acc:.4f} macroF1={mf1:.4f} "
                  f"flagged={flagged_this_round}")

    result = {
        "method": method,
        "final_acc": history["acc"][-1],
        "final_macro_f1": history["macro_f1"][-1],
        "history": history,
        "total_bytes_transmitted": int(total_bytes_transmitted),
        "total_bytes_baseline_fedavg_equiv": int(total_bytes_baseline),
        "n_encryption_ops_total": n_enc_ops_total,
        "malicious_ids": sorted(list(malicious_ids)),
        "params": dict(T=T, local_epochs=local_epochs, participation=participation,
                        malicious_fraction=malicious_fraction, beta=beta,
                        key_bits=key_bits, z_thresh=z_thresh, seed=seed),
    }
    return result
