# DP-ADAGE: Differentially Private Active Defenses Against GNN Extraction

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b.svg)](https://arxiv.org/)

**Jamal Boussouf · Ibrahim Ouahbi · Khalid El Makkaoui**  
*University Mohammed First, Oujda, Morocco*

</div>

---

> **DP-ADAGE** is the **first GNN inference-time defense** that combines community-aware query-diversity tracking with a provable **Rényi Differential Privacy (RDP)** guarantee against model extraction attacks. It drives attacker surrogate accuracy to near-random-guess levels (avg. **32.7%** across 5 benchmarks) while preserving benign user accuracy within **0.1 points** of the undefended baseline.

---

## 📖 Paper

**"DP-ADAGE: Differentially Private Active Defenses Against GNN Extraction"**  
Submitted to *Elsevier* — preprint available.

> **TL;DR**: Graph Neural Networks deployed as cloud APIs are vulnerable to model extraction attacks. DP-ADAGE wraps any GNN API with four sequential components — a CommunityTracker, a BudgetAllocator, an RDPAccountant, and a Perturbation Mechanism — to deliver an auditable (ε\*, δ)-DP certificate for the entire query session without sacrificing benign utility.

---

## 🏗️ System Overview

```
Incoming Query (user u, node v)
         │
         ▼
┌─────────────────────┐
│  CommunityTracker   │  ← Louvain offline partition; per-user τ = |O_u| / K
└────────┬────────────┘
         │  τ (query diversity score)
         ▼
┌─────────────────────┐
│   BudgetAllocator   │  ← Dead-zone sigmoid: τ → ε(τ) per-query RDP budget
└────────┬────────────┘
         │  ε(τ)
         ▼
┌─────────────────────┐
│    RDPAccountant    │  ← Additive composition; hard cap ε_total; noise floor
└────────┬────────────┘
         │  ε_q (clamped)
         ▼
┌─────────────────────┐
│   Victim GNN +      │  ← Output clipping (ℓ₂-norm ≤ B or C)
│ Perturbation Mech.  │    + Randomized Response (labels) or Gaussian (embeddings)
└────────┬────────────┘
         │
         ▼
   Differentially Private Response
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| `CommunityTracker` | [`src/community_tracker.py`](src/community_tracker.py) | Louvain community detection; maintains per-user set O_u; computes τ = \|O_u\| / K |
| `BudgetAllocator` | [`src/budget_allocator.py`](src/budget_allocator.py) | Dead-zone sigmoid: ε(τ) = ε_max (if τ ≤ τ_dead) or ε_max + (ε_0 − ε_max)·g̃(τ) |
| `RDPAccountant` | [`src/rdp_accountant.py`](src/rdp_accountant.py) | Additive RDP composition; enforces ε_total cap; triggers noise floor |
| `Perturbation` | [`src/perturbation.py`](src/perturbation.py) | Randomized Response for logits; Gaussian mechanism (σ = √(α·Δ₂²/2ε_q)) for embeddings |
| `DP-ADAGE Pipeline` | [`src/dp_adage.py`](src/dp_adage.py) | Full query-response protocol (Algorithm 1 in the paper) |

---

## 📂 Project Structure

```
DP-ADAGE-V2/
├── README.md
├── requirements.txt
│
├── src/                              # Core system implementation
│   ├── __init__.py
│   ├── community_tracker.py          # CommunityTracker: Louvain + τ score
│   ├── budget_allocator.py           # BudgetAllocator: dead-zone sigmoid
│   ├── rdp_accountant.py             # RDPAccountant: composition + hard cap
│   ├── perturbation.py               # Gaussian / Randomized Response mechanism
│   ├── dp_adage.py                   # Full DP-ADAGE pipeline (Algorithm 1)
│   ├── adage_baseline.py             # Original ADAGE heuristic defense
│   ├── dawn_baseline.py              # DAWN watermarking baseline
│   ├── prada_baseline.py             # PRADA rate-limiting baseline
│   ├── gnn_models.py                 # GCN, GAT, GraphSAGE, GIN, GraphTransformer
│   ├── attacker.py                   # Surrogate model + extraction attack logic
│   └── utils.py                      # Helpers, metrics, dataset loading
│
├── experiments/                      # Reproducible experiment scripts
│   ├── exp_main_results.py           # Main defense comparison → Table 1
│   ├── exp_ablation_budget.py        # Ablation on ε_total → Table 2 / Fig 4
│   ├── exp_adaptive_attacker.py      # Adaptive attacker evaluation → Sec 6.2
│   ├── exp_large_scale.py            # Scalability on Flickr / OGB → Sec 6.1
│   ├── exp_kl_divergence.py          # KL divergence bound validation
│   ├── exp_cross_dataset.py          # Cross-dataset generalization
│   ├── exp_profiling.py              # Latency profiling
│   ├── run_all_datasets.py           # Driver: all 5 datasets × all defenses
│   └── _validate_benign.py           # Benign user utility validation
│
├── visuals/                          # Figure generation (→ PDF output)
│   ├── plot_main_results.py          # Fig 2: surrogate acc + budget trajectory
│   ├── plot_pareto.py                # Fig 3: Pareto trilemma scatter
│   ├── plot_all_results.py           # All-dataset comparison
│   └── outputs/                      # Generated PDFs
│       ├── fig2_main_results.pdf
│       ├── fig3_pareto.pdf
│       └── fig7_radar.pdf
│
└── data/                             # Experimental results & configuration
    ├── results_all_datasets.csv      # Main results: 5 datasets × 6 defenses × 5 seeds
    ├── results_main_cora.csv         # Detailed Cora trajectory (per-query)
    ├── results_ablation_budget.csv   # Budget ablation (ε_total ∈ {1, 10, 200})
    ├── results_adaptive_attacker.csv # Adaptive attacker (uncertainty + budget-aware)
    ├── results_large_scale.csv       # Flickr scalability results
    ├── p_values_table1.csv           # Statistical significance (Table 1)
    └── config_default.json           # Default hyperparameter configuration
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyG extensions**: `torch-scatter` and `torch-sparse` are optional. Install precompiled wheels matching your PyTorch/CUDA version from the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 2. Run the main experiment (Table 1 — all 5 datasets)

```bash
python experiments/run_all_datasets.py
# Results saved to data/results_all_datasets.csv
```

### 3. Run a single dataset experiment

```bash
python experiments/exp_main_results.py --dataset Cora --seeds 5
```

### 4. Ablation study (ε_total sensitivity)

```bash
python experiments/exp_ablation_budget.py
```

### 5. Adaptive attacker evaluation

```bash
python experiments/exp_adaptive_attacker.py
```

### 6. Generate all figures (PDF)

```bash
python visuals/plot_main_results.py
python visuals/plot_pareto.py
python visuals/plot_all_results.py
```

### 7. Interactive notebook

```bash
jupyter notebook RUN.ipynb
```

---

## ⚙️ Configuration

Default hyperparameters ([`data/config_default.json`](data/config_default.json)):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epsilon_0` | 0.01 | Strict privacy floor — max noise (attackers at budget cap) |
| `epsilon_max` | 10.0 | Relaxed budget — min noise (benign users in dead zone) |
| `epsilon_total` | 200.0 | Hard session budget cap (ε_total) |
| `k` (k_sharp) | 20 | Sigmoid steepness of the budget allocator |
| `tau_0` | 0.30 | Sigmoid inflection point (diversity fraction) |
| `tau_dead` | 0.05 | Dead-zone threshold — below this, ε(τ) = ε_max |
| `alpha` | 8 | RDP order α |
| `delta` | 1e-6 | DP failure probability δ |
| `clip_C` | 1.0 | Logit clipping norm C |
| `clip_B` | 1.0 | Embedding clipping norm B |
| `num_communities` | 10 | Target number of Louvain communities K |

> **Choosing ε_total**: For a *meaningful* formal certificate (Proposition 3.2 in the paper), use a single-digit ε_total (e.g., ε_total = 8.0 with α = 8, δ = 10⁻⁶ gives ε* ≈ 9.97, a tight guarantee). The ε_total = 200.0 default is a **high-availability configuration** that stress-tests the mechanism; it remains mathematically valid but the converted (ε*, δ)-DP bound is loose.

---

## 🔐 Formal Privacy Guarantee

For any user issuing T queries, the full interaction transcript satisfies **(ε\*, δ)-DP** where:

$$\varepsilon^* = \varepsilon_{\text{total}} + \frac{\log(1/\delta)}{\alpha - 1}$$

**Example** — with `α = 8`, `ε_total = 200.0`, `δ = 1e-6`:  
→ `ε* = 200.0 + log(10⁶)/7 ≈ 201.97`

This is backed by four theorems in Section 4 of the paper:

| Theorem | Statement |
|---------|-----------|
| **Theorem 4.1** | Per-query (α, ε_q)-RDP via calibrated Gaussian mechanism |
| **Theorem 4.2** | Cumulative (α, Σε_q)-RDP via additive RDP composition |
| **Corollary 4.3** | Hard budget cap: ε_cum,u ≤ ε_total for all users |
| **Theorem 4.4** | Final (ε\*, δ)-DP certificate via RDP-to-DP conversion |

The noise floor mechanism ensures that once ε_cum,u = ε_total, **every subsequent response carries maximum perturbation** — preventing further meaningful extraction regardless of output type.

---

## 📊 Key Results (Table 1)

Results averaged over **5 random seeds**, evaluated at **200 attacker queries** with **ε_total = 200.0**.

### Surrogate Accuracy ↓ (lower = stronger defense)

| Dataset | No Defense | StaticNoise | PRADA | DAWN | OriginalADAGE | **DP-ADAGE** |
|---------|-----------|-------------|-------|------|--------------|-------------|
| Cora | 0.814 | 0.806 | 0.677 | 0.794 | 0.663 | **0.276** |
| Citeseer | 0.712 | 0.695 | 0.588 | 0.710 | 0.720 | **0.263** |
| Pubmed | 0.790 | 0.797 | 0.672 | 0.774 | 0.535 | **0.439** |
| ACM | 0.866 | 0.847 | 0.700 | 0.823 | 0.783 | **0.305** |
| DBLP | 0.824 | 0.821 | 0.649 | 0.799 | 0.822 | **0.349** |
| **Average** | 0.801 | 0.793 | 0.657 | 0.780 | 0.705 | **0.327** |

### Surrogate Fidelity ↓ (lower = stronger defense)

| Dataset | No Defense | StaticNoise | PRADA | DAWN | OriginalADAGE | **DP-ADAGE** |
|---------|-----------|-------------|-------|------|--------------|-------------|
| Cora | 0.923 | 0.914 | 0.728 | 0.908 | 0.748 | **0.277** |
| Citeseer | 0.873 | 0.849 | 0.664 | 0.853 | 0.874 | **0.278** |
| Pubmed | 0.931 | 0.944 | 0.744 | 0.929 | 0.534 | **0.470** |
| ACM | 0.987 | 0.977 | 0.773 | 0.957 | 0.860 | **0.315** |
| DBLP | 0.923 | 0.931 | 0.701 | 0.900 | 0.926 | **0.363** |

### Benign Downstream Accuracy ↑ (higher = better utility)

| Dataset | No Defense | **DP-ADAGE** | Δ (loss) |
|---------|-----------|-------------|---------|
| Cora | 0.806 | **0.806** | 0.000 |
| Citeseer | 0.713 | **0.712** | -0.001 |
| Pubmed | 0.794 | **0.794** | 0.000 |
| ACM | 0.871 | **0.871** | 0.000 |
| DBLP | 0.845 | **0.845** | 0.000 |
| **Average** | 0.806 | **0.806** | **< 0.001** |

> DP-ADAGE is the **only defense** that simultaneously achieves near-random-guess attacker accuracy AND preserves benign utility within 0.001 of the undefended baseline. All improvements over baselines are statistically significant (p < 10⁻¹² by paired t-test; see [`data/p_values_table1.csv`](data/p_values_table1.csv)).

---

## 📉 Ablation: Budget Cap ε_total

DP-ADAGE was evaluated with three budget settings across all datasets:

| ε_total | Behavior | Surrogate Acc (Cora) | Benign Acc (Cora) |
|---------|----------|---------------------|-------------------|
| 1.0 | Noise floor reached after ~20 queries; maximum protection | ~0.12 | ~0.80 |
| 10.0 | Budget depletes after ~140 queries; balanced | ~0.24 | ~0.80 |
| 200.0 | Extended benign utility; generous budget | ~0.28 | ~0.81 |

**Finding**: Tightening ε_total slows extraction faster at essentially **zero cost to benign utility**, confirming that the dead-zone sigmoid cleanly separates user populations.

---

## 🛡️ Adaptive Attacker Evaluation

Two adaptive strategies tested against DP-ADAGE (500 queries on Cora):

| Strategy | No Defense | OriginalADAGE | **DP-ADAGE** |
|----------|-----------|--------------|-------------|
| Uncertainty sampling | 0.757 | 0.357–0.413 | **0.164–0.222** |
| Budget-aware | 0.670–0.780 | 0.327–0.476 | **0.177–0.240** |

**Finding**: Even a fully budget-aware attacker cannot break through the hard cap — once ε_total is exhausted, every response is noise-floored regardless of query strategy.

---

## 🌐 Large-Scale Scalability (Flickr)

| Dataset | Nodes | Edges | Defense | Surrogate Acc | Avg Latency |
|---------|-------|-------|---------|--------------|-------------|
| Flickr | ~89K | ~449K | NoDefense | 0.4234 | 269.5 ms |
| Flickr | ~89K | ~449K | **DP-ADAGE** | **0.4218** | 269.6 ms |

**Finding**: DP-ADAGE's latency overhead on large graphs is negligible (< 0.1 ms), confirming the O(U·K) memory footprint is manageable in production.

---

## 📐 Datasets

| Dataset | Nodes | Edges | Classes | Task | Modularity |
|---------|-------|-------|---------|------|------------|
| Cora | 2,708 | 5,429 | 7 | Node classification | High |
| Citeseer | 3,327 | 4,732 | 6 | Node classification | High |
| Pubmed | 19,717 | 88,648 | 3 | Node classification | 0.7705 |
| ACM | 3,025 | 26,256 | 3 | Node classification | 0.1531 |
| DBLP | 17,716 | 105,734 | 4 | Node classification | 0.4366 |
| Flickr | ~89K | ~449K | 7 | Node classification | — |

---

## 🔗 Related Work

| Work | Venue | Relation to DP-ADAGE |
|------|-------|----------------------|
| **ADAGE** (Xu et al.) | ASIACCS'26 | Predecessor; DP-ADAGE adds formal RDP guarantees |
| **PRADA** (Juuti et al.) | EuroS&P'19 | Rate-limiting baseline; no formal DP |
| **DAWN** (Szyller et al.) | ACM MM'21 | Watermarking baseline; detects theft post-hoc only |
| **DP-SGD** (Abadi et al.) | CCS'16 | Training-time DP; orthogonal to inference-time DP |
| **Rényi DP** (Mironov) | CSF'17 | Theoretical foundation for budget accounting |
| **Node-level DP** (Daigavane et al.) | arXiv'21 | Training-time node privacy; complementary |

---

## 📝 Citation

If you use DP-ADAGE in your research, please cite:

```bibtex
@article{boussouf2026dpadage,
  title     = {DP-ADAGE: Differentially Private Active Defenses Against GNN Extraction},
  author    = {Boussouf, Jamal and Ouahbi, Ibrahim and El Makkaoui, Khalid},
  journal   = {Elsevier},
  year      = {2026},
  institution = {University Mohammed First, Oujda, Morocco}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

This work builds on the [ADAGE framework](https://doi.org/10.1145/3708821.3710437) (Xu et al., ASIACCS'26). Louvain community detection is implemented via the [`python-louvain`](https://github.com/taynaud/python-louvain) library. GNN models use [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).
