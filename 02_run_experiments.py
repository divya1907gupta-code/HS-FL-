import json, time, numpy as np
from fl_runner import run_federated

RESULTS = {}
T_MAIN = 12
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["fedavg", "fedprox", "compression_only", "encryption_only", "hsfl"]

t_start = time.time()

# ------------------------------------------------------------------
# Experiment A: main accuracy / macro-F1 comparison across methods x seeds
# ------------------------------------------------------------------
print("=== Experiment A: main comparison (5 methods x 5 seeds) ===")
expA = {m: [] for m in METHODS}
for m in METHODS:
    for s in SEEDS:
        t0 = time.time()
        r = run_federated(method=m, T=T_MAIN, seed=s, fedprox_mu=0.01 if m == "fedprox" else 0.0)
        dt = time.time() - t0
        expA[m].append({"seed": s, "final_acc": r["final_acc"],
                         "final_macro_f1": r["final_macro_f1"],
                         "bytes": r["total_bytes_transmitted"],
                         "bytes_baseline": r["total_bytes_baseline_fedavg_equiv"],
                         "history_acc": r["history"]["acc"],
                         "history_f1": r["history"]["macro_f1"]})
        print(f"  {m} seed={s} acc={r['final_acc']:.4f} f1={r['final_macro_f1']:.4f} "
              f"({dt:.1f}s)")
RESULTS["expA_main_comparison"] = expA
print(f"Elapsed so far: {time.time()-t_start:.1f}s\n")

# ------------------------------------------------------------------
# Experiment B: robustness to malicious clients (gradient-scaling attack)
# compare fedavg vs hsfl (with threat detection) vs hsfl-no-detect proxy
# (we approximate "no detection" by compression_only+encryption combo w/o filter
#  -> use hsfl but we also run a variant with z_thresh huge to disable filtering)
# ------------------------------------------------------------------
print("=== Experiment B: robustness to malicious clients ===")
mal_fracs = [0.0, 0.1, 0.2, 0.3]
expB = {"fedavg": [], "hsfl": [], "hsfl_no_detect": []}
for frac in mal_fracs:
    for s in [0, 1, 2]:
        r_fedavg = run_federated(method="fedavg", T=T_MAIN, seed=s,
                                  malicious_fraction=frac, malicious_kind="scale")
        r_hsfl = run_federated(method="hsfl", T=T_MAIN, seed=s,
                                malicious_fraction=frac, malicious_kind="scale")
        r_hsfl_nd = run_federated(method="hsfl", T=T_MAIN, seed=s,
                                   malicious_fraction=frac, malicious_kind="scale",
                                   z_thresh=1e9)  # effectively disables filtering
        expB["fedavg"].append({"frac": frac, "seed": s, "acc": r_fedavg["final_acc"],
                                "f1": r_fedavg["final_macro_f1"]})
        expB["hsfl"].append({"frac": frac, "seed": s, "acc": r_hsfl["final_acc"],
                              "f1": r_hsfl["final_macro_f1"]})
        expB["hsfl_no_detect"].append({"frac": frac, "seed": s, "acc": r_hsfl_nd["final_acc"],
                                        "f1": r_hsfl_nd["final_macro_f1"]})
    print(f"  malicious_fraction={frac} done")
RESULTS["expB_adversarial_robustness"] = expB
print(f"Elapsed so far: {time.time()-t_start:.1f}s\n")

# ------------------------------------------------------------------
# Experiment C: robustness to client participation / dropout
# ------------------------------------------------------------------
print("=== Experiment C: robustness to client dropout ===")
participations = [0.3, 0.5, 0.7, 1.0]
expC = {"fedavg": [], "hsfl": []}
for part in participations:
    for s in [0, 1, 2]:
        r_fedavg = run_federated(method="fedavg", T=T_MAIN, seed=s, participation=part)
        r_hsfl = run_federated(method="hsfl", T=T_MAIN, seed=s, participation=part)
        expC["fedavg"].append({"participation": part, "seed": s, "acc": r_fedavg["final_acc"]})
        expC["hsfl"].append({"participation": part, "seed": s, "acc": r_hsfl["final_acc"]})
    print(f"  participation={part} done")
RESULTS["expC_dropout_robustness"] = expC
print(f"Elapsed so far: {time.time()-t_start:.1f}s\n")

with open("/home/claude/pipeline/results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print(f"\nALL DONE. Total elapsed: {time.time()-t_start:.1f}s")
print("Saved to results.json")
