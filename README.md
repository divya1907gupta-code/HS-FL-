# HS-FL Reproduction Pipeline — Complete Code

This is the complete, from-scratch implementation used to produce every
number, table, and figure in the revised manuscript and the reproduction
report. Nothing in the paper is hand-typed or assumed — every result traces
back to one of these scripts.

## Why from-scratch?

The execution sandbox had **no internet access**, so `torch` and any external
cryptography library (`liboqs`, `phe`, etc.) could not be installed. The
entire pipeline — the neural network, the federated training loop, gradient
compression, the Paillier homomorphic cryptosystem, and the threat detector —
is implemented in plain NumPy + SymPy in `fl_core.py`.

## Directory contents

| File | Purpose |
|---|---|
| `00_sample_dataset.sh` | Streams the raw NF-BoT-IoT-V2 zip (37.76M rows) and writes a class-stratified ~967K-row sample, without ever fully extracting the 6GB CSV. |
| `01_preprocess.py` | Loads the sample, restricts to the 10 device-IP clients, drops leakage-prone columns (IPs/ports), scales features, and writes one `.npz` file per client under `clients/`. |
| `fl_core.py` | Core building blocks: the numpy MLP (forward/backward), Adaptive Gradient Compression, the real Paillier cryptosystem (keygen/encrypt/add/decrypt), and the threat detectors — including the two rejected designs (`gradient_norm_filter`, `direction_filter`), kept in the code and documented as negative results, and the adopted `behavioral_filter`. |
| `fl_runner.py` | `run_federated(...)`: runs one full federated experiment (FedAvg / FedProx / Compression-only / Encryption-only / HS-FL) for T rounds, given a method name and hyperparameters. This is the function every experiment in the paper calls. |
| `02_run_experiments.py` | The original (non-checkpointed) experiment driver. Kept for reference; superseded by `03_resumable_runner.py` because a single sandbox command has an execution time limit shorter than the full experiment matrix. |
| `03_resumable_runner.py` | **The script actually used to produce the paper's results.** A checkpointed job queue: it runs jobs from a fixed list (Experiments A/B/C), saves progress to `checkpoint.json` after every single job, and can be safely re-invoked repeatedly (each call picks up where the last one stopped) until the full matrix (70 jobs) is done. |
| `04_analyze_results.py` | Loads `checkpoint.json` and reproduces every statistic in the paper: Table IV (main comparison), Table V (paired t-test / Wilcoxon significance), Table VI (adversarial robustness), Table VII (dropout robustness). |
| `05_generate_figures.py` | Loads `checkpoint.json` and regenerates all 6 figures used in the paper (architecture diagram + 5 result plots). |

## Reproducing everything, end to end

```bash
# 1. Sample the dataset (only needs to be run once; requires the NF-BoT-IoT-V2 zip)
./00_sample_dataset.sh /path/to/NF-BoT-IoT-V2.zip /home/claude/pipeline/data

# 2. Build per-client train/test arrays (10 real device clients)
python3 01_preprocess.py

# 3. Run the full experiment matrix (70 jobs: Experiment A/B/C).
#    Each call runs jobs until a time budget is hit, then stops cleanly.
#    Re-run the same command repeatedly until it prints "ALL JOBS COMPLETE."
python3 03_resumable_runner.py 220   # 220 = seconds per call; tune to your environment

# 4. Reproduce every statistic and table in the paper
python3 04_analyze_results.py

# 5. Reproduce every figure in the paper
python3 05_generate_figures.py
```

Total experiment matrix: **70 jobs** =
- Experiment A (main comparison): 5 methods × 5 seeds = 25 runs, T=10 rounds
- Experiment B (adversarial robustness): 3 malicious fractions × 3 seeds × 3 methods = 27 runs
- Experiment C (dropout robustness): 3 participation rates × 3 seeds × 2 methods = 18 runs

## Key implementation notes (things that are NOT obvious from reading the paper alone)

1. **Client partitioning is by real device IP**, not synthetic Dirichlet splitting — see `01_preprocess.py`'s `CLIENT_IPS` list. This is what makes the non-IID structure genuine rather than artificial.

2. **The threat detector went through three iterations.** `fl_core.py` keeps the code for all three:
   - `gradient_norm_filter()` — the first (rejected) design, documented in its own docstring as a negative result.
   - `direction_filter()` — the second (rejected) design, also documented as a negative result in its docstring.
   - `behavioral_filter()` — the adopted design, used by `fl_runner.py`. It is the only one wired into `run_federated()`.

3. **Paillier key size is 160–256 bits**, not production-secure — this was a deliberate, documented trade-off for CPU feasibility in pure Python (see comments in `03_resumable_runner.py`, `KEY_BITS_FAST`). Do not reuse this key size outside a research/demo context.

4. **`checkpoint.json`** (included alongside this code) contains every raw per-seed result produced during this study. `04_analyze_results.py` and `05_generate_figures.py` both read directly from it, so you can regenerate every table and figure without re-running the (slower) experiment matrix.

## Requirements

```
numpy
pandas
scikit-learn
scipy
sympy
matplotlib
```

All are pip-installable (`pip install <package> --break-system-packages` if
running in an externally-managed environment like the sandbox this was
built in).
