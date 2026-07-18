"""
04_analyze_results.py

Loads checkpoint.json (raw per-seed results from 03_resumable_runner.py)
and reproduces every statistic reported in the paper:
  - Table IV: main comparison (accuracy / macro-F1 / communication cost)
  - Table V:  paired t-test + Wilcoxon significance (HS-FL vs each baseline)
  - Table VI: robustness to malicious clients (Experiment B)
  - Table VII: robustness to client dropout (Experiment C)

Run after 03_resumable_runner.py has completed all jobs.
"""
import json
import numpy as np
from scipy import stats

CKPT_PATH = "/home/claude/pipeline/checkpoint.json"

def load_results():
    ckpt = json.load(open(CKPT_PATH))
    return ckpt["results"]


def analyze_experiment_a(results):
    expA = [r for r in results if r["job"]["exp"] == "A"]
    methods = ["fedavg", "fedprox", "compression_only", "encryption_only", "hsfl"]
    by_method = {m: sorted([r for r in expA if r["job"]["method"] == m],
                            key=lambda x: x["job"]["seed"]) for m in methods}

    print("=" * 78)
    print("TABLE IV: Main comparison (mean +/- std over 5 seeds, T=10 rounds)")
    print("=" * 78)
    summary = {}
    for m in methods:
        accs = np.array([r["final_acc"] for r in by_method[m]])
        f1s = np.array([r["final_macro_f1"] for r in by_method[m]])
        bytes_tx = np.array([r["bytes"] for r in by_method[m]])
        bytes_base = np.array([r["bytes_baseline"] for r in by_method[m]])
        reduction = 100 * (1 - bytes_tx.mean() / bytes_base.mean())
        summary[m] = dict(acc_mean=float(accs.mean()), acc_std=float(accs.std()),
                           f1_mean=float(f1s.mean()), f1_std=float(f1s.std()),
                           bytes_mean=float(bytes_tx.mean()), reduction_pct=float(reduction))
        print(f"{m:20s} Acc={accs.mean()*100:6.2f}+/-{accs.std()*100:4.2f}%  "
              f"MacroF1={f1s.mean()*100:6.2f}+/-{f1s.std()*100:4.2f}%  "
              f"CommBytes={bytes_tx.mean():8.0f} (reduction: {reduction:5.1f}%)")

    print()
    print("=" * 78)
    print("TABLE V: Statistical significance -- HS-FL vs each baseline (paired, 5 seeds)")
    print("=" * 78)
    hsfl_accs = np.array([r["final_acc"] for r in by_method["hsfl"]])
    hsfl_f1s = np.array([r["final_macro_f1"] for r in by_method["hsfl"]])
    sig_results = {}
    for m in methods:
        if m == "hsfl":
            continue
        accs = np.array([r["final_acc"] for r in by_method[m]])
        f1s = np.array([r["final_macro_f1"] for r in by_method[m]])
        t_acc, p_acc = stats.ttest_rel(hsfl_accs, accs)
        t_f1, p_f1 = stats.ttest_rel(hsfl_f1s, f1s)
        try:
            _, wp_acc = stats.wilcoxon(hsfl_accs, accs)
        except Exception:
            wp_acc = float("nan")
        sig_results[m] = dict(t_acc=float(t_acc), p_acc=float(p_acc),
                               wp_acc=float(wp_acc), t_f1=float(t_f1), p_f1=float(p_f1))
        print(f"HS-FL vs {m:18s} | Acc: t={t_acc:6.2f} p={p_acc:.4f} Wilcoxon-p={wp_acc:.4f} | "
              f"F1: t={t_f1:6.2f} p={p_f1:.4f}")

    return summary, sig_results


def analyze_experiment_b(results):
    expB = [r for r in results if r["job"]["exp"] == "B"]
    print()
    print("=" * 78)
    print("TABLE VI: Robustness to malicious clients (gradient-scaling attack)")
    print("=" * 78)
    fracs = sorted(set(r["job"]["frac"] for r in expB))
    table = {}
    for frac in fracs:
        print(f"\n-- malicious_fraction = {frac} --")
        table[frac] = {}
        for m in ["fedavg", "hsfl", "hsfl_no_detect"]:
            subset = [r for r in expB if r["job"]["frac"] == frac and r["job"]["method"] == m]
            accs = np.array([r["acc"] for r in subset])
            f1s = np.array([r["f1"] for r in subset])
            table[frac][m] = dict(acc_mean=float(accs.mean()), acc_std=float(accs.std()))
            print(f"  {m:18s} Acc={accs.mean()*100:6.2f}+/-{accs.std()*100:5.2f}%  "
                  f"MacroF1={f1s.mean()*100:6.2f}+/-{f1s.std()*100:5.2f}%")
    return table


def analyze_experiment_c(results):
    expC = [r for r in results if r["job"]["exp"] == "C"]
    print()
    print("=" * 78)
    print("TABLE VII: Robustness to client participation / dropout")
    print("=" * 78)
    parts = sorted(set(r["job"]["participation"] for r in expC))
    table = {}
    for part in parts:
        print(f"\n-- participation = {part} --")
        table[part] = {}
        for m in ["fedavg", "hsfl"]:
            subset = [r for r in expC if r["job"]["participation"] == part and r["job"]["method"] == m]
            accs = np.array([r["acc"] for r in subset])
            table[part][m] = dict(acc_mean=float(accs.mean()), acc_std=float(accs.std()))
            print(f"  {m:10s} Acc={accs.mean()*100:6.2f}+/-{accs.std()*100:5.2f}%")
    return table


if __name__ == "__main__":
    results = load_results()
    print(f"Loaded {len(results)} total job results from {CKPT_PATH}\n")

    summaryA, sigA = analyze_experiment_a(results)
    tableB = analyze_experiment_b(results)
    tableC = analyze_experiment_c(results)

    out = {"expA_summary": summaryA, "expA_significance": sigA,
           "expB_table": tableB, "expC_table": tableC}
    with open("/home/claude/pipeline/final_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved consolidated analysis to final_analysis.json")
