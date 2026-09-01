# Meta-Learning for Zero-Day Intrusion Detection in IoT Networks

A controlled comparison of **Prototypical Networks**, **first-order MAML**, and **Reptile** against a *matched* transfer-learning baseline, for detecting previously unseen ("zero-day") attack classes in IoT network traffic.

MSc Research Project (COM748) — Ulster University, Birmingham
Author: Adewale Abdul-Lateef Odukoya · Supervisor: Dr. Marwan M Radwan

---

## What this project does

Supervised intrusion detection systems collapse when they meet an attack class that was not in their training data. This project asks a narrower, more testable question than most of the literature does:

> Does meta-learning offer a genuine advantage over conventional supervised learning for detecting unseen IoT attack classes — one attributable to the **meta-learning mechanism itself** rather than to a more favourable evaluation setup?

Three design decisions separate this study from the prior work it builds on:

1. **A strictly disjoint zero-day protocol.** Meta-training classes and zero-day evaluation classes never overlap, and the partition is fixed *before* training begins. Several published "zero-day" results do not enforce this, which inflates their reported accuracy.
2. **A matched transfer-learning control.** The `FINETUNE` baseline sees exactly the same class partition, the same episodes, and the same adaptation budget as the meta-learners — so any accuracy difference is attributable to the adaptation mechanism, not to differing experimental conditions.
3. **Joint reporting of accuracy and adaptation cost.** Per-episode adaptation time is reported alongside detection metrics, which matters for resource-constrained IoT deployment and is almost never reported in this literature.

---

## Headline results

**CICIoT2023 — zero-day accuracy (mean over 5 seeds)**

| Model | K=1 | K=5 | K=10 | Adaptation time (K=5) |
|---|---|---|---|---|
| **Prototypical Networks** | **0.712** | **0.763** | **0.767** | **0.0006 s** |
| MAML (first-order) | 0.604 | 0.633 | 0.650 | 0.0148 s |
| Reptile | 0.483 | 0.608 | 0.625 | 0.0078 s |
| FINETUNE (transfer control) | 0.533 | 0.599 | 0.619 | 0.0309 s |

ProtoNet beats the matched transfer control by **17.9 / 16.4 / 14.8** points at K = 1 / 5 / 10, significant at every K (p ≤ 4×10⁻⁵) and surviving Bonferroni correction. It is simultaneously the **most accurate and the cheapest to adapt** — roughly 50× faster per episode than the fine-tuning control.

**TON-IoT — zero-day accuracy (mean over 3 seeds)**

| Model | K=1 | K=5 | K=10 |
|---|---|---|---|
| Prototypical Networks | 0.560 | 0.572 | 0.582 |
| MAML (first-order) | 0.476 | 0.581 | 0.572 |
| Reptile | 0.394 | 0.461 | 0.486 |

The ranking only **partially replicates**. ProtoNet's lead over MAML is significant at K=1 (p = 0.027) but disappears at K=5 and K=10. The most plausible explanation is **meta-training task diversity**: CICIoT2023 offers a 26-class meta-train pool, TON-IoT only 5.

**The central finding:** metric-based meta-learning is an accurate and computationally cheap route to zero-day IoT intrusion detection — but its advantage over optimisation-based methods is *conditional on meta-training task diversity*, not universal.

---

## Repository structure

```
.
├── src/
│   ├── config/            # Experiment configuration
│   ├── data/              # Dataset loading, preprocessing, episodic sampling
│   ├── evaluation/        # Metrics and evaluation harness
│   ├── losses/            # Loss functions
│   ├── meta/              # Meta-learning algorithm implementations
│   ├── models/            # Backbones and baseline architectures
│   ├── training/          # Shared training loops (train_meta.py, etc.)
│   ├── utils/             # Seeding, helpers
│   ├── visualization/     # Figure generation
│   └── classes.py         # Zero-day class partition definitions
├── notebooks/             # Exploratory analysis
├── scripts/               # Helper scripts
├── tests/                 # Test suite
├── logs/                  # Raw run logs (experimental provenance)
│
├── run_experiments.py         # Main sweep runner: model x k-shot x seed
├── evaluate_finetune.py       # Transfer-learning control evaluation
├── analysis_figures.py        # Generates dissertation figures
├── analysis_ttests.py         # Paired t-test / Wilcoxon significance testing
├── analysis_convergence.py    # Convergence curve analysis
├── backfill_adaptation_time.py
├── recover_sweep.py
├── check_sweep.py
└── requirements.txt
```

The framework separates configuration, data, models, training, and evaluation behind a shared interface. All three meta-learners and both baseline families reuse an identical training loop and evaluation harness — only algorithm-specific behaviour differs. That is what makes the comparison genuinely like-for-like.

---

## Getting started

```bash
git clone https://github.com/AdewaleOdukoya/MetaLearning-IoT-IDS.git
cd MetaLearning-IoT-IDS

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Datasets

Neither dataset is committed to this repository. Download them separately and place them under `data/raw/`:

| Dataset | Source |
|---|---|
| CICIoT2023 | https://www.unb.ca/cic/datasets/iotdataset-2023.html |
| TON-IoT (`Train_Test_Network.csv`) | https://research.unsw.edu.au/projects/toniot-datasets |

### Reproducing the experiments

```bash
# Full sweep: {ProtoNet, MAML, Reptile} x {1,5,10}-shot x 5 seeds
python run_experiments.py

# Or target a subset
python run_experiments.py --models MAML REPTILE --k-shots 1 5 --seeds 42 43

# Transfer-learning control
python evaluate_finetune.py

# Analysis and figures
python analysis_figures.py
python analysis_ttests.py
python analysis_convergence.py
```

Results are written to `reports/tables/sweep_raw_results.csv` and `reports/tables/sweep_summary.csv`.

---

## Experimental configuration

| Parameter | CICIoT2023 | TON-IoT |
|---|---|---|
| N-way | 5 | 5 |
| K-shot | 1, 5, 10 | 1, 5, 10 |
| Query set size | 15 | 15 |
| Episodes per epoch | 1,000 | 1,000 |
| Epochs | 20 | 20 |
| Meta-optimiser | Adam, lr 1×10⁻³ | Adam, lr 1×10⁻³ |
| MAML / Reptile inner steps | 5 @ lr 0.01 | 5 @ lr 0.01 |
| FINETUNE adaptation steps | 20 | — (not extended) |
| Seeds | 5 (42–46) | 3 (42–44) |
| Input features | 46 | 100 (post one-hot) |

**Backbone:** all models share a lightweight MLP (two hidden layers, 128 → 64 units), so that algorithmic — not architectural — differences drive the results.

**Statistical testing:** seed-matched paired t-tests plus Wilcoxon signed-rank, α = 0.05, cross-checked against a Bonferroni-corrected threshold (≈ 0.0028) for the 18 pairwise comparisons per dataset.

### Zero-day class partitions

| Dataset | Held-out classes | Scenario |
|---|---|---|
| CICIoT2023 | `Mirai-greeth_flood`, `Mirai-greip_flood`, `Mirai-udpplain` | Unseen attack family |
| CICIoT2023 | `DDoS-SlowLoris`, `DoS-HTTP_Flood`, `Recon-OSScan` | Unseen variant of a seen family |
| CICIoT2023 | `DNS_Spoofing`, `DictionaryBruteForce` | Unseen spoofing / brute-force |
| TON-IoT | `ddos`, `injection`, `password`, `ransomware`, `xss` | Mixed novel categories (5-class minimum) |

---

## Limitations

Stated plainly, because they bound what these results can support:

- **TON-IoT used 3 seeds, not 5** — fewer comparisons survive Bonferroni correction there, so TON-IoT claims rest on a weaker evidential base than CICIoT2023 claims.
- **A single fixed zero-day partition per dataset** — splits were chosen with a documented rationale, but not randomised and averaged over.
- **Timings are CPU, not edge hardware** — the cost *ranking* is likely to hold, but these are not proof of Raspberry Pi-class deployability.
- **No genuine cross-dataset transfer** — zero-day generalisation was tested within each dataset, not by meta-training on one and evaluating on the other.
- **The transfer-learning control was not extended to TON-IoT.**

## Future work

- Vary meta-training class count directly to test the task-diversity explanation, rather than inferring it from two datasets that happened to differ in size.
- Genuine cross-dataset transfer (requires reconciling CICIoT2023's and TON-IoT's feature sets).
- Benchmark adaptation and inference on real edge hardware.
- Raise TON-IoT to 5 seeds and randomise the zero-day partition across several splits.

---

## Key references

- Finn, Abbeel & Levine (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML.
- Snell, Swersky & Zemel (2017). *Prototypical Networks for Few-Shot Learning.* NeurIPS.
- Nichol, Achiam & Schulman (2018). *On First-Order Meta-Learning Algorithms.* arXiv:1803.02999.
- Neto et al. (2023). *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment.* Sensors 23(13):5941.
- Moustafa (2021). *A New Distributed Architecture for Evaluating AI-Based Security Systems at the Edge: Network TON_IoT Datasets.* Sustainable Cities and Society 72:102994.
- Raghu et al. (2020). *Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML.* ICLR.

The full reference list is in the dissertation.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

Datasets are **not** covered by this licence and remain subject to their original terms (Canadian Institute for Cybersecurity, and UNSW Canberra, respectively).

## Citation

```bibtex
@mastersthesis{odukoya2026metalearning,
  title  = {Meta-Learning for Adaptive Zero-Day Cyber Threat Detection in IoT Networks:
            A Comparative Evaluation of Prototypical Networks, First-Order MAML, and Reptile},
  author = {Odukoya, Adewale Abdul-Lateef},
  school = {Ulster University},
  year   = {2026},
  type   = {MSc dissertation}
}
```
