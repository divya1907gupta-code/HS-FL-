"""
05_generate_figures.py

Generates every figure used in the paper/report from checkpoint.json:
  fig0_architecture.png        - pipeline diagram (static, hand-drawn boxes)
  fig1_accuracy_f1.png         - Figure 2: accuracy & macro-F1 bar chart
  fig2_comm_cost.png           - Figure 4: communication cost bar chart
  fig3_adversarial_robustness.png - Figure 5: robustness to malicious clients
  fig4_dropout_robustness.png  - Figure 6: robustness to client dropout
  fig5_convergence.png         - Figure 3: convergence curves

Run after 03_resumable_runner.py has completed all jobs.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

CKPT_PATH = "/home/pipeline/checkpoint.json"
OUT_DIR = "/home/pipeline/figs"
os.makedirs(OUT_DIR, exist_ok=True)

ckpt = json.load(open(CKPT_PATH))
results = ckpt["results"]
expA = [r for r in results if r["job"]["exp"] == "A"]
expB = [r for r in results if r["job"]["exp"] == "B"]
expC = [r for r in results if r["job"]["exp"] == "C"]

methods = ["fedavg", "fedprox", "compression_only", "encryption_only", "hsfl"]
labels = ["FedAvg", "FedProx", "Compression-only", "Encryption-only", "Proposed HS-FL"]
colors = ["#7f8c8d", "#95a5a6", "#3498db", "#e67e22", "#27ae60"]
by_method = {m: sorted([r for r in expA if r["job"]["method"] == m],
                        key=lambda x: x["job"]["seed"]) for m in methods}

# ---------------------------------------------------------------------------
# Figure 0: pipeline architecture diagram (static illustration, no data)
# ---------------------------------------------------------------------------
def make_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, color="#eaf2f8", edge="#2c3e50"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                            linewidth=1.4, edgecolor=edge, facecolor=color)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9.5, wrap=True)

    def arrow(x1, y1, x2, y2, text=None):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                             linewidth=1.3, color="#2c3e50")
        ax.add_patch(a)
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, text, ha="center", fontsize=8, color="#555555")

    box(0.3, 4.3, 2.0, 1.0, "Client i\nLocal SGD training\n(gt_i = wt+1 - wt)")
    box(0.3, 2.9, 2.0, 1.0, "Adaptive Gradient\nCompression (AGC)\n\u03b8=\u03bc+\u03b2\u03c3")
    box(0.3, 1.5, 2.0, 1.0, "Paillier Encryption\n(client-side, per\nnonzero component)")
    arrow(1.3, 4.3, 1.3, 3.9)
    arrow(1.3, 2.9, 1.3, 2.5)

    box(3.7, 3.4, 2.2, 1.0, "Aggregation Server\nHomomorphic Sum\n(Eq. 4-6)")
    box(3.7, 1.9, 2.2, 1.0, "Decrypt aggregate\n(single decryption\nper round)")
    box(6.6, 3.4, 2.2, 1.0, "Behavioral Threat\nDetection\n(per-client baseline)")
    box(6.6, 1.9, 2.2, 1.0, "Capped exclusion\n(max 40%/round)\n+ EMA baseline update")

    arrow(2.3, 2.0, 3.7, 3.7, "enc. updates")
    arrow(4.8, 3.4, 4.8, 2.9)
    arrow(5.9, 3.9, 6.6, 3.9)
    arrow(7.7, 3.4, 7.7, 2.9)

    box(3.7, 0.3, 5.1, 1.0, "Global Model Update  wt+1 = wt - \u03b7 \u00b7 mean(kept updates)")
    arrow(7.7, 1.9, 6.25, 1.3)
    arrow(4.8, 1.9, 5.0, 1.3)
    arrow(6.25, 0.8, 1.3, 5.3, "broadcast wt+1")

    ax.set_title("Revised HS-FL Pipeline (as implemented and evaluated)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig0_architecture.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 1: accuracy & macro-F1 bar chart
# ---------------------------------------------------------------------------
def make_accuracy_f1_fig():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    acc_means = [np.mean([r["final_acc"] for r in by_method[m]]) * 100 for m in methods]
    acc_stds = [np.std([r["final_acc"] for r in by_method[m]]) * 100 for m in methods]
    f1_means = [np.mean([r["final_macro_f1"] for r in by_method[m]]) * 100 for m in methods]
    f1_stds = [np.std([r["final_macro_f1"] for r in by_method[m]]) * 100 for m in methods]

    axes[0].bar(labels, acc_means, yerr=acc_stds, capsize=4, color=colors)
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Classification Accuracy (mean \u00b1 std, 5 seeds)")
    axes[0].set_ylim(0, 100)
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")

    axes[1].bar(labels, f1_means, yerr=f1_stds, capsize=4, color=colors)
    axes[1].set_ylabel("Macro-F1 (%)")
    axes[1].set_title("Macro-F1 Score (mean \u00b1 std, 5 seeds)")
    axes[1].set_ylim(0, 100)
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig1_accuracy_f1.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: communication cost bar chart
# ---------------------------------------------------------------------------
def make_comm_cost_fig():
    bytes_means = [np.mean([r["bytes"] for r in by_method[m]]) for m in methods]
    baseline = np.mean([r["bytes_baseline"] for r in by_method["fedavg"]])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, [b / 1024 for b in bytes_means], color=colors)
    ax.axhline(baseline / 1024, color="red", linestyle="--", label="Uncompressed FedAvg baseline")
    ax.set_ylabel("Cumulative uplink (KB, log scale)")
    ax.set_yscale("log")
    ax.set_title("Communication Cost per Client Over 10 Rounds")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig2_comm_cost.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: robustness to malicious clients
# ---------------------------------------------------------------------------
def make_adversarial_robustness_fig():
    fracs = sorted(set(r["job"]["frac"] for r in expB))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, lab, c in [("fedavg", "FedAvg", "#7f8c8d"),
                       ("hsfl", "HS-FL (detection ON)", "#27ae60"),
                       ("hsfl_no_detect", "HS-FL (detection OFF)", "#e74c3c")]:
        means, stds = [], []
        for frac in fracs:
            subset = [r for r in expB if r["job"]["frac"] == frac and r["job"]["method"] == m]
            accs = np.array([r["acc"] for r in subset])
            means.append(accs.mean() * 100); stds.append(accs.std() * 100)
        ax.errorbar([f * 100 for f in fracs], means, yerr=stds, marker="o", label=lab, color=c, capsize=4)
    ax.set_xlabel("Malicious client fraction (%)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Robustness to Gradient-Scaling Attacks (10-client federation)")
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig3_adversarial_robustness.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: dropout robustness
# ---------------------------------------------------------------------------
def make_dropout_robustness_fig():
    parts = sorted(set(r["job"]["participation"] for r in expC))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, lab, c in [("fedavg", "FedAvg", "#7f8c8d"), ("hsfl", "Proposed HS-FL", "#27ae60")]:
        means, stds = [], []
        for part in parts:
            subset = [r for r in expC if r["job"]["participation"] == part and r["job"]["method"] == m]
            accs = np.array([r["acc"] for r in subset])
            means.append(accs.mean() * 100); stds.append(accs.std() * 100)
        ax.errorbar([p * 100 for p in parts], means, yerr=stds, marker="o", label=lab, color=c, capsize=4)
    ax.set_xlabel("Client participation rate (%)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Robustness to Client Dropout")
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig4_dropout_robustness.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5: convergence curves
# ---------------------------------------------------------------------------
def make_convergence_fig():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, lab, c in zip(methods, labels, colors):
        curves = np.array([r["history_acc"] for r in by_method[m]])
        mean_curve = curves.mean(axis=0) * 100
        ax.plot(range(1, len(mean_curve) + 1), mean_curve, marker="o", label=lab, color=c)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Convergence over Federated Rounds (mean of 5 seeds)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig5_convergence.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    make_architecture_diagram()
    make_accuracy_f1_fig()
    make_comm_cost_fig()
    make_adversarial_robustness_fig()
    make_dropout_robustness_fig()
    make_convergence_fig()
    print("All figures written to", OUT_DIR)
    print(os.listdir(OUT_DIR))
