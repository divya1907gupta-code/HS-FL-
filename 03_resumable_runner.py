import json, time, os, sys
from fl_runner import run_federated

CKPT = "/home/claude/pipeline/checkpoint.json"
T_MAIN = 10
KEY_BITS_FAST = 160  # reduced Paillier key size purely for runtime feasibility in this
                      # sandbox; flagged explicitly in the write-up as a demo-scale
                      # parameter, not a production security recommendation.
SEEDS_A = [0, 1, 2, 3, 4]
METHODS_A = ["fedavg", "fedprox", "compression_only", "encryption_only", "hsfl"]

SEEDS_BC = [0, 1, 2]
MAL_FRACS = [0.0, 0.2, 0.4]
PARTICIPATIONS = [0.3, 0.6, 1.0]

TIME_BUDGET_SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0


def build_job_queue():
    jobs = []
    # Experiment A
    for m in METHODS_A:
        for s in SEEDS_A:
            jobs.append({"exp": "A", "method": m, "seed": s})
    # Experiment B
    for frac in MAL_FRACS:
        for s in SEEDS_BC:
            for m in ["fedavg", "hsfl", "hsfl_no_detect"]:
                jobs.append({"exp": "B", "method": m, "seed": s, "frac": frac})
    # Experiment C
    for part in PARTICIPATIONS:
        for s in SEEDS_BC:
            for m in ["fedavg", "hsfl"]:
                jobs.append({"exp": "C", "method": m, "seed": s, "participation": part})
    return jobs


def job_key(job):
    return json.dumps(job, sort_keys=True)


def load_checkpoint():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {"done": {}, "results": []}


def save_checkpoint(ckpt):
    tmp = CKPT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, CKPT)


def run_job(job):
    key_bits = KEY_BITS_FAST
    if job["exp"] == "A":
        r = run_federated(method=job["method"], T=T_MAIN, seed=job["seed"],
                           key_bits=key_bits,
                           fedprox_mu=0.01 if job["method"] == "fedprox" else 0.0)
        return {"job": job, "final_acc": r["final_acc"], "final_macro_f1": r["final_macro_f1"],
                "bytes": r["total_bytes_transmitted"],
                "bytes_baseline": r["total_bytes_baseline_fedavg_equiv"],
                "history_acc": r["history"]["acc"], "history_f1": r["history"]["macro_f1"]}
    elif job["exp"] == "B":
        m = job["method"]
        if m == "hsfl_no_detect":
            r = run_federated(method="hsfl", T=T_MAIN, seed=job["seed"], key_bits=key_bits,
                               malicious_fraction=job["frac"], malicious_kind="scale",
                               z_thresh=1e9)
        else:
            r = run_federated(method=m, T=T_MAIN, seed=job["seed"], key_bits=key_bits,
                               malicious_fraction=job["frac"], malicious_kind="scale")
        return {"job": job, "acc": r["final_acc"], "f1": r["final_macro_f1"]}
    elif job["exp"] == "C":
        r = run_federated(method=job["method"], T=T_MAIN, seed=job["seed"], key_bits=key_bits,
                           participation=job["participation"])
        return {"job": job, "acc": r["final_acc"]}


def main():
    jobs = build_job_queue()
    ckpt = load_checkpoint()
    done = ckpt["done"]
    remaining = [j for j in jobs if job_key(j) not in done]
    print(f"Total jobs: {len(jobs)} | already done: {len(done)} | remaining: {len(remaining)}")

    t_start = time.time()
    n_run_this_call = 0
    for job in remaining:
        if time.time() - t_start > TIME_BUDGET_SEC:
            print(f"Time budget ({TIME_BUDGET_SEC}s) reached, stopping cleanly.")
            break
        t0 = time.time()
        res = run_job(job)
        dt = time.time() - t0
        ckpt["results"].append(res)
        done[job_key(job)] = True
        n_run_this_call += 1
        print(f"  ran {job} in {dt:.1f}s")
        # checkpoint after EVERY job so nothing is lost on timeout
        save_checkpoint(ckpt)

    remaining_after = len([j for j in jobs if job_key(j) not in done])
    print(f"\nThis call ran {n_run_this_call} jobs in {time.time()-t_start:.1f}s. "
          f"{remaining_after} jobs remaining out of {len(jobs)}.")
    if remaining_after == 0:
        print("ALL JOBS COMPLETE.")


if __name__ == "__main__":
    main()
