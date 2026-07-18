"""
Core building blocks for the HS-FL pipeline, implemented from scratch
(no torch available in this offline sandbox -> numpy autograd-free MLP).

Contains:
  - MLP: small feed-forward net w/ manual forward/backward (softmax CE)
  - flatten/unflatten helpers (needed for AGC + Paillier, which both
    operate on the FLAT update vector, matching Eq. 2-3 / Eq. 4-6 of paper)
  - adaptive_gradient_compression(): Eq. 2-3
  - Paillier: minimal, real (not simulated) additive-homomorphic cryptosystem
  - secure_aggregate(): Eq. 4-6, using real Paillier encrypt/add/decrypt
  - gradient_norm_filter(): server-side layered threat detection
"""
import numpy as np
import random as pyrandom
import sympy

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. Small MLP, manual forward/backward, numpy only
# ----------------------------------------------------------------------
class MLP:
    def __init__(self, in_dim, hidden1=32, hidden2=16, out_dim=5, seed=0):
        r = np.random.default_rng(seed)
        self.shapes = [(in_dim, hidden1), (hidden1,), (hidden1, hidden2),
                        (hidden2,), (hidden2, out_dim), (out_dim,)]
        self.W1 = r.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden1)).astype(np.float32)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = r.normal(0, np.sqrt(2.0 / hidden1), (hidden1, hidden2)).astype(np.float32)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.W3 = r.normal(0, np.sqrt(2.0 / hidden2), (hidden2, out_dim)).astype(np.float32)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

    def get_flat(self):
        return np.concatenate([self.W1.ravel(), self.b1.ravel(),
                                self.W2.ravel(), self.b2.ravel(),
                                self.W3.ravel(), self.b3.ravel()])

    def set_flat(self, flat):
        i = 0
        for name, shape in zip(["W1", "b1", "W2", "b2", "W3", "b3"], self.shapes):
            n = int(np.prod(shape))
            setattr(self, name, flat[i:i + n].reshape(shape).astype(np.float32))
            i += n

    def n_params(self):
        return sum(int(np.prod(s)) for s in self.shapes)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        z3s = self.z3 - self.z3.max(axis=1, keepdims=True)
        expz = np.exp(z3s)
        self.probs = expz / expz.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y_onehot, l2=0.0):
        N = X.shape[0]
        dz3 = (self.probs - y_onehot) / N
        dW3 = self.a2.T @ dz3 + l2 * self.W3
        db3 = dz3.sum(axis=0)
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (self.z2 > 0)
        dW2 = self.a1.T @ dz2 + l2 * self.W2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T @ dz1 + l2 * self.W1
        db1 = dz1.sum(axis=0)
        return np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(),
                                db2.ravel(), dW3.ravel(), db3.ravel()])

    def predict(self, X, batch=4096):
        preds = []
        for i in range(0, len(X), batch):
            preds.append(self.forward(X[i:i + batch]).argmax(axis=1))
        return np.concatenate(preds)


def one_hot(y, n_classes):
    oh = np.zeros((len(y), n_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def local_train(global_flat, Xtr, ytr, in_dim, n_classes, epochs=1, batch_size=256,
                 lr=0.05, seed=0, fedprox_mu=0.0, malicious=None):
    """Runs `epochs` local SGD epochs starting from global_flat.
    Returns the UPDATE (delta = trained_flat - global_flat), i.e. g_i^t.
    malicious: None, or dict(kind='scale', factor=-5.0) / dict(kind='flip')
    """
    model = MLP(in_dim, out_dim=n_classes, seed=seed)
    model.set_flat(global_flat.copy())
    n = len(Xtr)
    idx = np.arange(n)
    r = np.random.default_rng(seed + 1000)

    y_use = ytr.copy()
    if malicious is not None and malicious.get("kind") == "flip":
        # label-flipping attack: shift every label by +1 (mod n_classes)
        y_use = (y_use + 1) % n_classes

    for ep in range(epochs):
        r.shuffle(idx)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            Xb, yb = Xtr[b], y_use[b]
            oh = one_hot(yb, n_classes)
            model.forward(Xb)
            grad = model.backward(Xb, oh)
            if fedprox_mu > 0:
                grad = grad + fedprox_mu * (model.get_flat() - global_flat)
            new_flat = model.get_flat() - lr * grad
            model.set_flat(new_flat)

    delta = model.get_flat() - global_flat

    if malicious is not None and malicious.get("kind") == "scale":
        delta = delta * malicious.get("factor", -5.0)

    return delta


# ----------------------------------------------------------------------
# 2. Adaptive Gradient Compression  (paper Eq. 2-3)
# ----------------------------------------------------------------------
def adaptive_gradient_compression(delta, beta=1.5):
    """theta = mean(|g|) + beta*std(|g|); keep entries with |g_j| >= theta."""
    mag = np.abs(delta)
    theta = mag.mean() + beta * mag.std()
    mask = mag >= theta
    compressed = np.where(mask, delta, 0.0).astype(np.float32)
    ratio_kept = mask.mean()
    return compressed, mask, ratio_kept


# ----------------------------------------------------------------------
# 3. Real (minimal) Paillier additive homomorphic cryptosystem
#    Reduced key size (demo-scale, NOT production-secure) so that
#    encrypting thousands of gradient components per round is tractable
#    in pure Python. This is explicitly flagged in the write-up.
# ----------------------------------------------------------------------
class Paillier:
    def __init__(self, key_bits=256, seed=0):
        pyrandom.seed(seed)
        p = sympy.randprime(2 ** (key_bits // 2 - 1), 2 ** (key_bits // 2))
        q = sympy.randprime(2 ** (key_bits // 2 - 1), 2 ** (key_bits // 2))
        while p == q:
            q = sympy.randprime(2 ** (key_bits // 2 - 1), 2 ** (key_bits // 2))
        self.n = p * q
        self.n2 = self.n * self.n
        self.g = self.n + 1  # standard simplification (g = n+1)
        lam = (p - 1) * (q - 1)
        self.lam = lam
        self.mu = pow(lam, -1, self.n)
        # sample a fresh r per encryption for semantic security
        self._r_cache = None

    def encrypt(self, m_int):
        m_int = m_int % self.n
        r = pyrandom.randrange(1, self.n)
        # g^m * r^n mod n^2 ; since g = n+1, g^m mod n^2 = 1 + m*n (fast form)
        gm = (1 + m_int * self.n) % self.n2
        rn = pow(r, self.n, self.n2)
        c = (gm * rn) % self.n2
        return c

    def add(self, c1, c2):
        return (c1 * c2) % self.n2

    def decrypt(self, c):
        u = pow(c, self.lam, self.n2)
        l = (u - 1) // self.n
        m = (l * self.mu) % self.n
        # map back from Z_n to signed integer range
        if m > self.n // 2:
            m -= self.n
        return m


def quantize(x, scale=1e4, clip=5e4):
    xi = np.clip(x, -clip, clip)
    return np.round(xi * scale).astype(np.int64)


def dequantize(xi, scale=1e4):
    return xi.astype(np.float64) / scale


def secure_aggregate(client_deltas, paillier, only_nonzero=True, scale=1e4):
    """Real Paillier-based secure aggregation (Eq. 4-6).
    client_deltas: list of flat np.float32 arrays (already compressed, i.e.
                   sparse with many exact zeros from AGC).
    Returns: aggregated (summed) delta vector (float), and bytes-equivalent
             ciphertext count actually encrypted (for comm-cost accounting).
    """
    dim = len(client_deltas[0])
    agg_cipher = None
    n_encrypted_total = 0

    # union of nonzero positions across clients that we must encrypt in each
    # client's vector individually (a real client only encrypts ITS nonzero
    # entries + sends zero-placeholders elsewhere, but to homomorphically
    # sum we must align on a common index set -> use union mask).
    if only_nonzero:
        union_mask = np.zeros(dim, dtype=bool)
        for d in client_deltas:
            union_mask |= (d != 0)
        idx = np.where(union_mask)[0]
    else:
        idx = np.arange(dim)

    qvecs = [quantize(d[idx], scale=scale) for d in client_deltas]
    n_encrypted_total = len(idx) * len(client_deltas)

    agg_q = np.zeros(len(idx), dtype=object)
    for qv in qvecs:
        for j in range(len(idx)):
            c = paillier.encrypt(int(qv[j]))
            agg_q[j] = c if isinstance(agg_q[j], int) is False and agg_q[j] == 0 else agg_q[j]
    return idx, qvecs, n_encrypted_total


def secure_aggregate_fast(client_deltas, paillier, scale=1e4):
    """Efficient version: encrypt each client's nonzero entries, homomorphically
    sum ciphertexts at matching positions, then decrypt the sums once.
    Returns aggregated float delta vector (sum across clients) and the
    number of Paillier encryption operations actually performed (comm/compute
    cost proxy)."""
    dim = len(client_deltas[0])
    union_mask = np.zeros(dim, dtype=bool)
    for d in client_deltas:
        union_mask |= (d != 0)
    idx = np.where(union_mask)[0]

    sums_cipher = {}
    n_ops = 0
    for d in client_deltas:
        qv = quantize(d[idx], scale=scale)
        for pos, val in zip(idx, qv):
            c = paillier.encrypt(int(val))
            n_ops += 1
            if pos in sums_cipher:
                sums_cipher[pos] = paillier.add(sums_cipher[pos], c)
            else:
                sums_cipher[pos] = c

    agg = np.zeros(dim, dtype=np.float64)
    for pos, c in sums_cipher.items():
        m = paillier.decrypt(c)
        agg[pos] = m / scale
    return agg, n_ops, len(idx)


# ----------------------------------------------------------------------
# 4. Server-side layered threat detection
# ----------------------------------------------------------------------
def gradient_norm_filter(client_deltas, z_thresh=2.0):
    """NAIVE VERSION - kept for the write-up as a documented negative result.
    Flags clients whose update L2 norm is a statistical outlier
    (|norm - median| > z_thresh * MAD-based robust std).
    FAILURE MODE (found empirically): in genuinely non-IID CIoT settings,
    the clients with the MOST local data naturally produce larger-magnitude
    updates (more local SGD steps per round), which this raw-magnitude
    filter conflates with malicious behavior -> it can systematically
    exclude the most informative legitimate clients every round, collapsing
    the global model. Superseded below by direction_filter()."""
    norms = np.array([np.linalg.norm(d) for d in client_deltas])
    med = np.median(norms)
    mad = np.median(np.abs(norms - med)) * 1.4826 + 1e-8  # robust std estimate
    z = np.abs(norms - med) / mad
    keep = z <= z_thresh
    return keep, norms


def behavioral_filter(client_ids, client_deltas, history, warmup_rounds=3,
                       z_thresh=3.0, cos_thresh=0.3):
    """Per-client BEHAVIORAL anomaly detection: compares each client's update
    only against ITS OWN history (running mean/std of norm, and direction
    consistency vs its own past average direction), never against other
    clients. This sidesteps the cross-client non-IID confound found with
    both gradient_norm_filter and direction_filter (a legitimately different
    -but honest- device is consistent with ITSELF over time; a sign-flip or
    scaling attack is a sudden deviation from that client's own established
    baseline, regardless of how different that baseline is from everyone
    else's).
    `history` is a dict keyed by client_id, persisted across rounds by the
    caller: {"norms": [...], "mean_dir": np.array, "n_seen": int}
    Returns keep_mask (aligned with client_ids order) and updated history.
    """
    keep = []
    for cid, d in zip(client_ids, client_deltas):
        h = history.get(cid, {"norms": [], "mean_dir": None, "n_seen": 0})
        norm = float(np.linalg.norm(d))
        n_seen = h["n_seen"]

        if n_seen < warmup_rounds or h["mean_dir"] is None:
            ok = True  # not enough history yet -> trust and learn baseline
        else:
            hist_norms = np.array(h["norms"])
            mu, sigma = hist_norms.mean(), hist_norms.std() + 1e-8
            z = abs(norm - mu) / sigma
            cos_to_own_history = float(np.dot(d, h["mean_dir"]) /
                                        (norm * np.linalg.norm(h["mean_dir"]) + 1e-12))
            ok = (z <= z_thresh) and (cos_to_own_history >= cos_thresh)

        keep.append(ok)

        # Always update the baseline with a SLOW EMA, regardless of the flag
        # outcome. FAILURE MODE FOUND without this: once one bad round shifts
        # the global model, every client's next honest update looks anomalous
        # vs. its now-stale pre-drift baseline -> cascading lockout where
        # eventually all clients get flagged, the safety net re-admits
        # everyone including the attacker, and the system never recovers.
        # A slow-decaying baseline lets legitimate drift be absorbed while
        # still reacting to a genuine sudden attack within a round or two.
        h["norms"].append(norm)
        h["norms"] = h["norms"][-10:]
        unit_d = d / (norm + 1e-12)
        if h["mean_dir"] is None:
            h["mean_dir"] = unit_d.copy()
        else:
            decay = 0.3 if ok else 0.05  # drift slower when currently flagged
            h["mean_dir"] = (1 - decay) * h["mean_dir"] + decay * unit_d
        h["n_seen"] = n_seen + 1
        history[cid] = h

    return np.array(keep), history


def direction_filter(client_deltas, cos_thresh=0.0, norm_mult=8.0):
    """Scale-invariant layered threat detection (Byzantine-robust style).
    Two checks, combined:
      1. Direction check: cosine similarity between each client's update
         and the coordinate-wise MEDIAN direction across clients. A
         gradient-scaling attack with a negative factor flips the sign of
         the update (cos ~ -1), which this catches regardless of the
         client's legitimate data volume (fixes the norm_filter failure
         mode above, since cosine similarity does not scale with |delta|).
      2. Magnitude safety-net: only flags on norm if a client's update is
         an extreme (norm_mult x) outlier vs the median -- loose enough
         that legitimate high-data clients are not penalized for producing
         bigger (but well-directed) updates.
    Returns boolean keep-mask and diagnostic dict.
    """
    deltas = np.stack(client_deltas, axis=0)
    median_dir = np.median(deltas, axis=0)
    med_norm = np.linalg.norm(median_dir) + 1e-12
    median_dir_unit = median_dir / med_norm

    norms = np.linalg.norm(deltas, axis=1)
    med_norm_all = np.median(norms) + 1e-12

    cos_sims = np.array([
        float(np.dot(d, median_dir_unit) / (np.linalg.norm(d) + 1e-12))
        for d in deltas
    ])

    direction_ok = cos_sims >= cos_thresh
    magnitude_ok = norms <= norm_mult * med_norm_all
    keep = direction_ok & magnitude_ok
    return keep, {"cos_sims": cos_sims, "norms": norms}
