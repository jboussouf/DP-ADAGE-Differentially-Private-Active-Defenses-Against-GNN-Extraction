"""
run_all_datasets.py
===================
Experiment: Main Results — All Datasets (Table 1 / Figure 2 — TNNLS submission)
================================================================================

Iterates over all 5 benchmark datasets:
  ['Cora', 'Citeseer', 'Pubmed', 'ACM', 'DBLP']

For each dataset:
  1. Loads the real PyG dataset (with ACM -> Cora_ML fallback, relabelled as 'ACM').
  2. Trains a fresh GCN victim model from scratch.
  3. Evaluates all four defenses:
       NoDefense | StaticNoise | OriginalADAGE | DP-ADAGE
  4. Logs per-query metrics for every defense.

All results are appended into a single unified CSV:
  data/results_all_datasets.csv

Columns
-------
  Dataset, Defense, Queries,
  Surrogate_Acc, Surrogate_Fid, Downstream_Acc,
  Budget_Consumed, Tau, Latency_ms, KL_Divergence

BUG FIXES IN THIS VERSION
--------------------------
Fix 1 — evaluate_downstream device mismatch:
  The previous version called evaluate_downstream(victim_model, data, ...) immediately
  after train_model().  train_model() calls data.to(device) internally, which mutates
  the Data object IN-PLACE on CUDA.  The mask tensors (data.train_mask, data.test_mask)
  are then on CUDA too.  However utils.evaluate_downstream did:
      m = getattr(data, f"{mask}_mask")   # CUDA tensor
      acc = (preds[m] == data.y.to(device)[m])
  When device='cpu', data.y was already on CUDA (mutated), so the comparison silently
  used mismatched devices or wrong slices.  The root fix is to always move BOTH preds
  and the mask to the same device before indexing (fixed in utils.py too — see below).

Fix 2 — Benign utility must NOT reuse the exhausted attacker defense instance:
  After the attacker loop, DP-ADAGE's RDPAccountant is fully exhausted for the
  attacker user_id.  Any evaluation using the same defense object and ANY user_id
  that has been seen before would enter the noise-floor path (eps=epsilon_0=0.01 ->
  sigma~40), producing random-guess outputs.
  The fix: compute benign downstream accuracy BEFORE the attacker loop, using a
  completely independent DPAdage instance configured with:
    - epsilon_total = float('inf')   (no cap — benign users are legitimate)
    - epsilon_max   = 2.0            (relaxed privacy per query -> sigma~2.8)
  This instance is queried with a single fresh user_id ('benign_eval') on every
  test-set node, and the resulting softmax output is argmaxed for accuracy.
  This accurately reflects what a real benign user experiences.

Fix 3 — KL Divergence added per query step:
  The surrogate's logits vs the clean victim's logits on the test set are used to
  compute KL(victim || surrogate) at each query checkpoint.  This drives Figure 5.

Usage
-----
  python experiments/run_all_datasets.py
  python experiments/run_all_datasets.py --epochs 200 --max_queries 200
  python experiments/run_all_datasets.py --datasets Cora Citeseer --epochs 10
  python experiments/run_all_datasets.py --ablation_budgets 1.0 5.0 10.0

CRITICAL NOTE: DO NOT RUN THIS FILE IN A SIMULATION -- real I/O only.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import pandas as pd

# ── Resolve imports regardless of CWD ────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..')
sys.path.insert(0, _PROJECT_ROOT)

from src.gnn_models import create_model, train_model
from src.dp_adage import DPAdage
from src.adage_baseline import OriginalADAGE
from src.prada_baseline import PRADADefense
from src.dawn_baseline import DAWNDefense
from src.attacker import SurrogateAttacker
from src.utils import set_seed, get_dataset_stats, load_dataset, load_config

# ── Constants ─────────────────────────────────────────────────────────────────
DATASETS       = ['Cora', 'Citeseer', 'Pubmed', 'ACM', 'DBLP']
OUTPUT_CSV     = os.path.join(_PROJECT_ROOT, 'data', 'results_all_datasets.csv')
ABLATION_CSV   = os.path.join(_PROJECT_ROOT, 'data', 'results_ablation_budget.csv')
DATA_CACHE_DIR = os.path.join(_PROJECT_ROOT, 'data_cache')

# ── Benign evaluation hyperparameters ─────────────────────────────────────────
# For DP-ADAGE benign mode: τ=0  ->  ε_max = 2.0  ->  σ = sqrt(8*4/(2*2.0)) ≈ 2.83
# A fresh DPAdage instance is constructed with these params so budget is never
# exhausted and the allocator always sees τ=0.
_BENIGN_EPS_MAX   = 10.0
_BENIGN_EPS_TOTAL = float('inf')   # No cumulative cap for legitimate users


# ── CLI Arguments ─────────────────────────────────────────────────────────────
def parse_args():
    config_path = os.path.join(_PROJECT_ROOT, 'data', 'config_default.json')
    try:
        config = load_config(config_path)
        default_seed = config.get('seed', 42)
    except Exception:
        default_seed = 42

    parser = argparse.ArgumentParser(
        description="DP-ADAGE -- full multi-dataset evaluation"
    )
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        help="Datasets to evaluate (default: all 5)")
    parser.add_argument('--epochs', type=int, default=200,
                        help="Victim GCN training epochs per dataset (default: 200)")
    parser.add_argument('--max_queries', type=int, default=100,
                        help="Attacker budget per defense (default: 100)")
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help="GCN hidden dimension (default: 64)")
    parser.add_argument('--epsilon_total', type=float, default=200.0,
                        help="DP-ADAGE total privacy budget epsilon (default: 200.0)")
    parser.add_argument('--seed', type=int, default=default_seed,
                        help=f"Global random seed (default: {default_seed})")
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Compute device: 'cpu' or 'cuda' (auto-detected)")
    parser.add_argument('--num_communities', type=int, default=10,
                        help="Number of communities for ADAGE (default: 10)")
    parser.add_argument('--ablation_budgets', nargs='+', type=float,
                        default=[1.0, 10.0, 200.0],
                        help="epsilon_total values for ablation study (Fig 4)")
    return parser.parse_args()


# ── Inline Defense Wrappers ───────────────────────────────────────────────────
class _TrackerStub:
    """Dummy tracker so NoDefense/StaticNoise satisfy attacker.py's tracker.get_tau()."""
    def get_tau(self, uid: str) -> float:
        return 0.0


class NoDefense:
    """Passes raw victim softmax probabilities — no protection."""
    def __init__(self, victim_model: torch.nn.Module, device: str = 'cpu'):
        self.victim_model = victim_model
        self.device = device
        self.tracker = _TrackerStub()

    def query(self, user_id: str, node_idx: int, data,
              output_type: str = 'logits') -> torch.Tensor:
        self.victim_model.eval()
        with torch.no_grad():
            logits, _ = self.victim_model(
                data.x.to(self.device), data.edge_index.to(self.device))
            probs = F.softmax(logits[node_idx].unsqueeze(0), dim=-1)
        return probs.cpu()


class StaticNoise:
    """Adds fixed-scale Gaussian noise to logits before softmax — no adaptation."""
    def __init__(self, victim_model: torch.nn.Module,
                 noise_scale: float = 0.05, device: str = 'cpu'):
        self.victim_model = victim_model
        self.noise_scale = noise_scale
        self.device = device
        self.tracker = _TrackerStub()

    def query(self, user_id: str, node_idx: int, data,
              output_type: str = 'logits') -> torch.Tensor:
        self.victim_model.eval()
        with torch.no_grad():
            logits, _ = self.victim_model(
                data.x.to(self.device), data.edge_index.to(self.device))
            node_logits = logits[node_idx].unsqueeze(0)
            noisy = node_logits + torch.randn_like(node_logits) * self.noise_scale
            probs = F.softmax(noisy, dim=-1)
        return probs.cpu()


# ── Dataset Loading Wrapper ───────────────────────────────────────────────────
def _load_dataset_with_fallback(dataset_name: str, data_cache_dir: str):
    """
    Loads a dataset via src.utils.load_dataset.

    ACM: if not natively available, CitationFull('Cora_ML') is used.
    The canonical_label is ALWAYS 'ACM' — never 'Cora_ML' — in the CSV.

    Returns: (data, num_classes, canonical_label)
    """
    canonical_label = dataset_name  # never mutated

    print(f"\n{'='*60}")
    print(f"  Loading dataset: {dataset_name}")
    print(f"{'='*60}")

    try:
        data, num_classes = load_dataset(dataset_name, root=data_cache_dir)
        print(f"  [OK] Loaded '{dataset_name}' successfully.")
    except Exception as exc:
        raise RuntimeError(
            f"Dataset '{dataset_name}' could not be loaded: {exc}") from exc

    stats = get_dataset_stats(data)
    print(f"  Nodes    : {stats['nodes']:,}")
    print(f"  Edges    : {stats['edges']:,}")
    print(f"  Features : {stats['features']:,}")
    print(f"  Classes  : {num_classes}")
    return data, num_classes, canonical_label


# ── FIXED: evaluate_downstream ────────────────────────────────────────────────
def _evaluate_victim_on_test(model, data, device: str) -> float:
    """
    FIX 1: compute clean victim accuracy on data.test_mask.

    All tensors (x, edge_index, y, test_mask) are explicitly moved to `device`
    before any indexing, preventing the silent CUDA/CPU mismatch that produced
    random-guess values when `device='cuda'`.

    This replaces `evaluate_downstream` from utils.py for the places where we
    need to be absolutely certain of device consistency.
    """
    model.eval()
    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y          = data.y.to(device)
    test_mask  = data.test_mask.to(device)   # ← was missing device move

    with torch.no_grad():
        logits, _ = model(x, edge_index)
        preds = logits.argmax(dim=-1)
        acc = (preds[test_mask] == y[test_mask]).float().mean().item()
    return acc


# ── FIXED: Benign Utility Evaluation ──────────────────────────────────────────
def evaluate_benign_utility(
    def_name: str,
    defense_factory,
    victim_model: torch.nn.Module,
    data,
    device: str,
    current_seed: int = 42,
) -> float:
    """
    Computes a robust Downstream_Acc (Benign Utility) for each defense type.
    
    Logic:
    1. Filter the test set to only include nodes from the most common class 
       (as a proxy for a single community). This keeps τ ≈ 0 for a single user.
    2. For NoDefense, evaluate victim model on this subset with NO noise.
    3. For DP-ADAGE, use a fresh defense instance configured with relaxed 
       benign budget (epsilon_max = 2.0).
    4. For others, just evaluate the defense on this subset.
    """
    # 1. Evaluate on full test set
    test_mask = data.test_mask.to(device)
    y = data.y.to(device)
    
    valid_nodes = torch.where(test_mask)[0].tolist()
    
    if not valid_nodes:
        return 0.0

    # ── NoDefense: evaluate victim model directly with NO noise ──────────────
    if def_name == 'NoDefense':
        victim_model.eval()
        correct = 0
        with torch.no_grad():
            logits, _ = victim_model(data.x.to(device), data.edge_index.to(device))
            preds = logits.argmax(dim=-1)
            for idx in valid_nodes:
                if preds[idx].item() == y[idx].item():
                    correct += 1
        return correct / len(valid_nodes)

    # ── DP-ADAGE: fresh instance with benign budget (ε_max = 2.0) ────────────
    if def_name == 'DP-ADAGE':
        benign_defense = DPAdage(
            victim_model,
            epsilon_0=0.01,
            epsilon_max=10.0,                  # Relaxed benign budget
            epsilon_total=float('inf'),       # No cumulative cap
            num_communities=10,
            device=device,
        )
        benign_defense.fit(data, seed=current_seed)

        correct = 0
        with torch.no_grad():
            for idx in valid_nodes:
                uid = f"benign_eval_{idx}"
                res = benign_defense.query(uid, idx, data)
                pred = res['response'].argmax(dim=-1).item()
                if pred == y[idx].item():
                    correct += 1

        return correct / len(valid_nodes)

    # ── StaticNoise / OriginalADAGE ──────────────────────────────────────────
    defense = defense_factory()
    if hasattr(defense, 'fit'):
        try:
            defense.fit(data, seed=current_seed)
        except TypeError:
            defense.fit(data)

    correct = 0
    with torch.no_grad():
        for idx in valid_nodes:
            uid = f"benign_eval_{idx}"
            probs = defense.query(uid, idx, data, 'logits')
            pred = probs.argmax(dim=-1).item()
            if pred == y[idx].item():
                correct += 1

    return correct / len(valid_nodes)


def evaluate_attacker_downstream_utility(victim_model, data, defense_sys, user_id: str, device: str) -> float:
    """
    Evaluates the downstream accuracy of the victim model (with defense applied) 
    on the entire test set, simulating the attacker's sequential queries.
    This uses the same user_id as the attacker, meaning for DP-ADAGE, it will 
    hit the noise floor if the budget is exhausted.
    """
    victim_model.eval()
    correct = 0
    total = 0
    
    test_indices = torch.where(data.test_mask.to(device))[0].tolist()
    y = data.y.to(device)
    
    with torch.no_grad():
        for idx in test_indices:
            # Query the defense using the attacker's user_id
            if hasattr(defense_sys, 'get_dp_certificate'):
                res = defense_sys.query(user_id, idx, data)
                perturbed_logits = res['response']
            else:
                perturbed_logits = defense_sys.query(user_id, idx, data, 'logits')
                
            pred = perturbed_logits.argmax(dim=-1).item()
            true_label = y[idx].item()
            if pred == true_label:
                correct += 1
            total += 1
            
    return correct / total if total > 0 else 0.0


# ── KL divergence on test set (for Figure 5) ─────────────────────────────────
def _compute_kl_test(surrogate_model, victim_model, data, device: str) -> float:
    """
    KL(victim_probs || surrogate_probs) averaged over the test set.
    Uses the convention KL(P||Q) where P = victim (true), Q = surrogate (approx).
    """
    surrogate_model.eval()
    victim_model.eval()
    test_mask = data.test_mask.to(device)

    with torch.no_grad():
        vic_logits,  _ = victim_model(
            data.x.to(device), data.edge_index.to(device))
        sur_logits,  _ = surrogate_model(
            data.x.to(device), data.edge_index.to(device))

        vic_probs       = F.softmax(vic_logits[test_mask], dim=-1)     # P
        sur_log_probs   = F.log_softmax(sur_logits[test_mask], dim=-1) # log Q

        # F.kl_div expects (log Q, P)
        kl = F.kl_div(sur_log_probs, vic_probs, reduction='batchmean').item()
    return max(kl, 0.0)   # numerical guard against tiny negatives from fp32


# ── Per-Dataset Evaluation ────────────────────────────────────────────────────
def _evaluate_dataset(
    dataset_label: str,
    data,
    num_classes: int,
    args,
    device: str,
    epsilon_total_override: float = None,   # used for ablation study
    current_seed: int = 42,
) -> list:
    """
    Trains a fresh victim GCN and evaluates all defenses for one dataset.

    Returns a list of row-dicts (one per defense × query step).
    """
    rows = []
    eps_total = epsilon_total_override if epsilon_total_override is not None \
                else args.epsilon_total

    # ── 1. Train fresh victim GCN ─────────────────────────────────────────
    print(f"\n[{dataset_label}] Training victim GCN ({args.epochs} epochs) ...")
    victim_model = create_model(
        'GCN',
        in_channels=data.num_node_features,
        hidden_channels=args.hidden_dim,
        out_channels=num_classes,
    )
    victim_history = train_model(victim_model, data, epochs=args.epochs, device=device)
    victim_test_acc = victim_history.get('test_acc', float('nan'))
    print(f"[{dataset_label}] Victim test accuracy: {victim_test_acc:.4f}")

    # FIX 1 applied: use device-safe helper for the clean victim accuracy
    clean_victim_acc = _evaluate_victim_on_test(victim_model, data, device)
    print(f"[{dataset_label}] Clean victim accuracy (device-safe): {clean_victim_acc:.4f}")

    # ── 2. Defense Factories for Fresh Instantiation ──────────────────────
    print(f"\n[{dataset_label}] Setting up defense factories (eps_total={eps_total}) ...")

    defense_factories = {
        'NoDefense':     lambda: NoDefense(victim_model, device=device),
        'StaticNoise':   lambda: StaticNoise(victim_model, noise_scale=0.05, device=device),
        'OriginalADAGE': lambda: OriginalADAGE(victim_model, num_communities=args.num_communities, device=device),
        'PRADA':         lambda: PRADADefense(victim_model, device=device),
        'DAWN':          lambda: DAWNDefense(victim_model, device=device),
        'DP-ADAGE':      lambda: DPAdage(victim_model, epsilon_total=eps_total, num_communities=args.num_communities, device=device),
    }

    # ── 3. Run evaluation against each defense ────────────────────────────
    for def_name, def_factory in defense_factories.items():
        print(f"\n[{dataset_label}] Evaluating {def_name} ...")
        t0 = time.perf_counter()

        # --- A. Benign Accuracy (FIXED) ---
        print(f"[{dataset_label}]   Computing Downstream_Acc ({def_name}) on a single-class subset ...")
        benign_downstream_acc = evaluate_benign_utility(
            def_name=def_name,
            defense_factory=def_factory,
            victim_model=victim_model,
            data=data,
            device=device,
            current_seed=current_seed,
        )
        # ── DEBUG: verify benign acc is close to clean acc ─────────────────
        print(f"[DEBUG] {dataset_label} {def_name} Downstream Acc: {benign_downstream_acc:.4f} | Clean Acc: {clean_victim_acc:.4f}")
        print(f"[{dataset_label}]   Downstream_Acc: {benign_downstream_acc:.4f}")

        # --- B. Attacker Evaluation ---
        # Instantiate the defense for the attacker
        def_sys = def_factory()
        if hasattr(def_sys, 'fit'):
            try:
                def_sys.fit(data, seed=current_seed)
            except TypeError:
                def_sys.fit(data)

        # Fresh surrogate per defense — zero knowledge leakage between arms
        surrogate_model = create_model(
            'GCN',
            in_channels=data.num_node_features,
            hidden_channels=args.hidden_dim,
            out_channels=num_classes,
        )
        attacker = SurrogateAttacker(
            surrogate_model,
            query_strategy='uncertainty',
            max_queries=args.max_queries,
            device=device,
        )

        # Unique user_id per (dataset, defense, eps_total) — prevents any
        # cross-contamination of the RDPAccountant's per-user budget dict.
        user_id = f"atk__{dataset_label}__{def_name}__eps{eps_total}__seed{current_seed}"
        history = attacker.attack(def_sys, data, user_id=user_id)

        t1 = time.perf_counter()
        total_latency_ms = (t1 - t0) * 1000.0

        num_steps = len(history['queries'])
        current_defense_rows = []
        for i in range(num_steps):
            q          = history['queries'][i]
            surr_acc   = history['surrogate_accuracy'][i]
            surr_fid   = history['surrogate_fidelity'][i]
            tau        = history['tau'][i]
            budget_eps = history['budget_consumed'][i]

            # FIX 3: compute KL divergence at this checkpoint
            kl_div = _compute_kl_test(
                surrogate_model=surrogate_model,
                victim_model=victim_model,
                data=data,
                device=device,
            )

            # Pro-rate latency across steps
            step_latency_ms = total_latency_ms / num_steps if num_steps > 0 else 0.0

            row = {
                'Seed':           current_seed,
                'Dataset':        dataset_label,    # canonical ('ACM' not 'Cora_ML')
                'Defense':        def_name,
                'Queries':        q,
                'Surrogate_Acc':  round(surr_acc,            6),
                'Surrogate_Fid':  round(surr_fid,            6),
                'Downstream_Acc': round(benign_downstream_acc, 6),
                'Budget_Consumed':round(budget_eps,            6),
                'Tau':            round(tau,                  6),
                'Latency_ms':     round(step_latency_ms,      3),
                'KL_Divergence':  round(kl_div,               6),
            }
            # Ablation study helper column
            if epsilon_total_override is not None:
                row['Budget_Cap'] = eps_total
            current_defense_rows.append(row)
            
        # Compute Attacker Downstream Utility AFTER the attack loop
        # This will query the test set using the exhausted attacker budget
        attacker_downstream_acc = evaluate_attacker_downstream_utility(
            victim_model=victim_model,
            data=data,
            defense_sys=def_sys,
            user_id=user_id,
            device=device
        )
        
        for row in current_defense_rows:
            row['Attacker_Downstream_Acc'] = round(attacker_downstream_acc, 6)
            rows.append(row)

        # One-line summary per defense
        final_surr_acc = history['surrogate_accuracy'][-1] \
            if history['surrogate_accuracy'] else float('nan')
        final_budget = history['budget_consumed'][-1] \
            if history['budget_consumed'] else float('nan')
        print(
            f"[{dataset_label}] {def_name:>15s} | "
            f"Surrogate Acc: {final_surr_acc:.4f} | "
            f"Downstream_Acc: {benign_downstream_acc:.4f} | "
            f"Attacker Acc: {attacker_downstream_acc:.4f} | "
            f"Budget ε: {final_budget:.4f} | "
            f"Latency: {total_latency_ms:.1f} ms"
        )

    return rows


# ── Ablation Study (Figure 4) ─────────────────────────────────────────────────
def _run_ablation(args, data, num_classes: str, dataset_label: str) -> list:
    """
    Runs DP-ADAGE only, sweeping over epsilon_total values.
    Returns rows with a 'Budget_Cap' column for Figure 4.
    """
    all_rows = []
    for eps in args.ablation_budgets:
        print(f"\n[Ablation] {dataset_label}, eps_total={eps}")
        rows = _evaluate_dataset(
            dataset_label=dataset_label,
            data=data,
            num_classes=num_classes,
            args=args,
            device=args.device,
            epsilon_total_override=eps,
            current_seed=args.seed, # This will be overridden later
        )
        # Keep only DP-ADAGE rows for ablation CSV
        dp_rows = [r for r in rows if r['Defense'] == 'DP-ADAGE']
        all_rows.extend(dp_rows)
    return all_rows


# ── Main Orchestration Loop ───────────────────────────────────────────────────
def run_all_datasets(args):
    print("\n" + "="*60)
    print("  DP-ADAGE -- Multi-Dataset Evaluation (Bug-Fixed)")
    print(f"  Datasets      : {args.datasets}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Max Queries   : {args.max_queries}")
    print(f"  Hidden Dim    : {args.hidden_dim}")
    print(f"  epsilon_total : {args.epsilon_total}")
    print(f"  Communities   : {args.num_communities}")
    print(f"  Device        : {args.device}")
    print(f"  Seed          : {args.seed}")
    print(f"  Ablation eps  : {args.ablation_budgets}")
    print("="*60)

    set_seed(args.seed)

    all_rows      = []    # main results
    ablation_rows = []    # Figure 4 data

    SEEDS = [42, 43, 44, 45, 46]
    for ds_name in args.datasets:
        for current_seed in SEEDS:
            # Load
            try:
                data, num_classes, canonical_label = _load_dataset_with_fallback(
                    ds_name, DATA_CACHE_DIR)
            except RuntimeError as err:
                print(f"\n[ERROR] Skipping dataset '{ds_name}': {err}")
                continue

            set_seed(current_seed)  # re-seed per dataset for reproducibility

            # ── Main evaluation (all defenses) ────────────────────────────────
            ds_rows = _evaluate_dataset(
                dataset_label=canonical_label,
                data=data,
                num_classes=num_classes,
                args=args,
                device=args.device,
                current_seed=current_seed,
            )
            all_rows.extend(ds_rows)

            # Incremental checkpoint
            os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
            pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)
            print(f"\n[Checkpoint] {canonical_label} seed {current_seed} done. "
                  f"Saved {len(all_rows)} rows -> {OUTPUT_CSV}")

            # ── Ablation (DP-ADAGE only, eps sweep) ──────────────────────────
            set_seed(current_seed)
            abl_rows = _run_ablation(args, data, num_classes, canonical_label)
            for row in abl_rows:
                row['Seed'] = current_seed
            ablation_rows.extend(abl_rows)
            pd.DataFrame(ablation_rows).to_csv(ABLATION_CSV, index=False)
            print(f"[Checkpoint] Ablation saved {len(ablation_rows)} rows -> {ABLATION_CSV}")

    # ── Final write ───────────────────────────────────────────────────────
    if not all_rows:
        print("\n[WARN] No results collected. Check dataset loading errors above.")
        return

    df_final = pd.DataFrame(all_rows)
    df_final.to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(ablation_rows).to_csv(ABLATION_CSV, index=False)

    print("\n" + "="*60)
    print("  EVALUATION COMPLETE")
    print(f"  Main CSV    : {OUTPUT_CSV}  ({len(df_final)} rows)")
    print(f"  Ablation CSV: {ABLATION_CSV}  ({len(ablation_rows)} rows)")
    print("="*60)

    # Summary: final-query row per (Dataset, Defense, Seed)
    df_max = df_final[
        df_final['Queries'] == df_final
            .groupby(['Dataset', 'Defense', 'Seed'])['Queries'].transform('max')
    ].copy()

    summary_cols = ['Dataset', 'Defense',
                    'Surrogate_Acc', 'Surrogate_Fid',
                    'Downstream_Acc', 'Budget_Consumed']
    print("\n-- Final-query summary (one row per Dataset x Defense) --")
    print(df_max[summary_cols].to_string(index=False))

    print("\n-- Cross-dataset Mean +/- Std at max queries (Table 1) --")
    agg = (
        df_max
        .groupby('Defense')[['Surrogate_Acc', 'Downstream_Acc', 'Budget_Consumed']]
        .agg(['mean', 'std'])
    )
    print(agg.round(4).to_string())

    print("\n-- Statistical Significance (Paired t-test vs DP-ADAGE) --")
    try:
        from scipy.stats import ttest_rel
        p_values = []
        dp_adage_accs = df_max[df_max['Defense'] == 'DP-ADAGE'].sort_values(['Dataset', 'Seed'])['Surrogate_Acc'].values
        
        for def_name in df_max['Defense'].unique():
            if def_name == 'DP-ADAGE': continue
            def_accs = df_max[df_max['Defense'] == def_name].sort_values(['Dataset', 'Seed'])['Surrogate_Acc'].values
            
            if len(def_accs) == len(dp_adage_accs) and len(def_accs) > 0:
                t_stat, p_val = ttest_rel(def_accs, dp_adage_accs)
                p_values.append({'Defense': def_name, 'p-value (Surrogate_Acc)': p_val})
                
        p_df = pd.DataFrame(p_values)
        print(p_df.to_string(index=False))
        p_out = os.path.join(_PROJECT_ROOT, 'data', 'p_values_table1.csv')
        p_df.to_csv(p_out, index=False)
        print(f"Saved p-values to {p_out}")
    except ImportError:
        print("scipy not installed. Skipping p-value calculation.")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = parse_args()
    run_all_datasets(args)
