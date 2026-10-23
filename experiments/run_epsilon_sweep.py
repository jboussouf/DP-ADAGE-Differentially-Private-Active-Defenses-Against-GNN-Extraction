#!/usr/bin/env python3
"""
run_epsilon_sweep.py
====================
Main experiment driver for the DP-ADAGE Total-Budget (epsilon_total) Sweep.

Sweeps epsilon_total in {1.0, 5.0, 10.0, 200.0} across seeds, datasets,
interaction protocols, query policies, and output types.

Usage Examples:
  # Fast sanity pilot (mock backend):
  python run_epsilon_sweep.py --pilot

  # Targeted sweep:
  python run_epsilon_sweep.py --caps 1.0 5.0 10.0 200.0 \
      --datasets cora pubmed --seeds 42 43 44 45 46 \
      --T 200 --T_attacker 500 --resume

  # Equal-budget ablation (static vs adaptive):
  python run_epsilon_sweep.py --ablation --resume
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sweep_engine import (
    SweepSession,
    SweepAccountant,
    _ALPHA,
    _DELTA,
    _EPS_MAX,
    _EPS_0,
    _K_SHARP,
    _TAU_0,
    _TAU_DEAD,
    _CLIP_C,
    _CLIP_B,
    _DELTA_2,
)

DATASET_CLASSES: Dict[str, int] = {
    "cora": 7,
    "pubmed": 3,
    "citeseer": 6,
    "acm": 7,       # maps to Cora_ML (7 classes)
    "dblp": 4,
}

ALL_DATASETS = ["cora", "pubmed", "citeseer", "acm", "dblp"]


def get_git_commit() -> str:
    """Retrieve short commit hash or fallback to unknown."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return commit
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DP-ADAGE Total-Budget (epsilon_total) Sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--caps",
        nargs="+",
        type=float,
        default=[1.0, 5.0, 10.0, 200.0],
        help="List of epsilon_total caps to sweep",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=["cora", "pubmed"],
        help="Datasets to evaluate (or 'all' for 5 benchmark graphs)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        type=str,
        default=["benign", "uncertainty", "budget_aware"],
        choices=["benign", "uncertainty", "budget_aware"],
        help="Query policies to evaluate",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        type=str,
        default=["fresh", "session"],
        choices=["fresh", "session"],
        help="Accounting protocols: fresh (non-binding) vs session (cumulative hard floor)",
    )
    parser.add_argument(
        "--output_types",
        nargs="+",
        type=str,
        default=["label", "embedding"],
        choices=["label", "embedding"],
        help="Model API output representations",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=200,
        help="Number of queries for benign policy / default session horizon",
    )
    parser.add_argument(
        "--T_attacker",
        type=int,
        default=500,
        help="Number of queries for attacker policies (uncertainty, budget_aware)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run quick sanity pilot (caps: [1.0, 200.0], dataset: cora, seeds: [42, 43], T: 50)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/sweep_epsilon",
        help="Directory to save per-query traces and manifests",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="reports/figures",
        help="Directory for generated figures and plots",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run equal-budget ablation (static per-query epsilon vs adaptive allocator)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Checkpoint/resume: skip runs whose trace.jsonl and manifest.json already exist and are complete",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Override class count for datasets (default derived per dataset)",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=64,
        help="Embedding dimension for embedding output mode",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mock", "real"],
        default="mock",
        help="Execution backend: 'mock' (stateless, labeled SIMULATED) or 'real' (GNN victim)",
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="results/epsilon_200_baseline",
        help="Directory containing or targeting the epsilon=200 baseline regression artifact",
    )

    args = parser.parse_args()

    # Apply pilot overrides if specified
    if args.pilot:
        args.caps = [1.0, 200.0]
        args.datasets = ["cora"]
        args.seeds = [42, 43]
        args.T = 50
        args.T_attacker = 50

    # Expand 'all' datasets
    if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
        args.datasets = ALL_DATASETS
    else:
        args.datasets = [d.lower() for d in args.datasets]

    return args


def is_run_complete(run_dir: Path, expected_lines: int) -> bool:
    """Check whether a previous run has valid, complete trace.jsonl and manifest.json."""
    trace_path = run_dir / "trace.jsonl"
    manifest_path = run_dir / "manifest.json"

    if not trace_path.is_file() or not manifest_path.is_file():
        return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            if not manifest.get("completed", False):
                return False

        with open(trace_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for line in f if line.strip())
        return line_count == expected_lines
    except Exception:
        return False


def execute_single_run(
    cap: float,
    dataset: str,
    seed: int,
    policy: str,
    protocol: str,
    output_type: str,
    horizon_T: int,
    args: argparse.Namespace,
    results_base: Path,
    git_commit: str,
    is_ablation: bool = False,
) -> Dict[str, Any]:
    """Execute one simulation session and write its trace + manifest."""
    num_classes = args.num_classes or DATASET_CLASSES.get(dataset, 7)
    run_suffix = "_ablation" if is_ablation else ""
    run_id = f"{dataset}_eps{cap}_seed{seed}_{policy}_{protocol}_{output_type}{run_suffix}"
    run_dir = results_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_file = run_dir / "trace.jsonl"
    manifest_file = run_dir / "manifest.json"

    if args.resume and is_run_complete(run_dir, horizon_T):
        return {
            "run_id": run_id,
            "status": "skipped",
            "reason": "checkpoint_found",
            "trace_path": str(trace_file),
        }

    tag = "[SIMULATED]" if args.backend == "mock" else "[REAL]"
    print(f" {tag} Executing {run_id} (T={horizon_T})...")

    # If ablation is active, we test static allocator: eps_alloc is constant = cap / horizon_T
    static_eps_alloc = (cap / float(horizon_T)) if is_ablation else None

    session = SweepSession(
        epsilon_total=cap,
        policy=policy,
        protocol=protocol,
        output_type=output_type,
        dataset=dataset,
        num_classes=num_classes,
        seed=seed,
        T=horizon_T,
        embed_dim=args.embed_dim,
    )

    # Run session queries
    records = session.run()

    # If running ablation, adjust eps_alloc and accounting to static budget
    if is_ablation:
        acc = SweepAccountant(epsilon_total=cap)
        cum = 0.0
        for idx, rec in enumerate(records):
            rec["eps_alloc"] = static_eps_alloc
            if cum >= cap - 1e-9:
                rec["eps_q"] = 0.0
                rec["floor_active"] = True
            else:
                charge = min(static_eps_alloc, cap - cum)
                cum += charge
                rec["eps_q"] = charge
                rec["floor_active"] = (cum >= cap - 1e-9)
            rec["eps_cum"] = cum

    # Write trace.jsonl
    with open(trace_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Write manifest.json
    manifest_data = {
        "run_id": run_id,
        "cap": cap,
        "dataset": dataset,
        "seed": seed,
        "policy": policy,
        "protocol": protocol,
        "output_type": output_type,
        "T": horizon_T,
        "is_ablation": is_ablation,
        "backend": args.backend,
        "simulated": (args.backend == "mock"),
        "git_commit": git_commit,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "completed": True,
        "hyperparameters": {
            "alpha": _ALPHA,
            "delta": _DELTA,
            "epsilon_max": _EPS_MAX,
            "epsilon_0": _EPS_0,
            "k_sharp": _K_SHARP,
            "tau_0": _TAU_0,
            "tau_dead": _TAU_DEAD,
            "clip_C": _CLIP_C,
            "clip_B": _CLIP_B,
            "delta_2": _DELTA_2,
        },
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return {
        "run_id": run_id,
        "status": "completed",
        "trace_path": str(trace_file),
    }


def main():
    args = parse_args()
    results_base = Path(args.results_dir)
    results_base.mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit()

    tag = "[SIMULATED]" if args.backend == "mock" else "[REAL]"
    print("=" * 70)
    print(f" {tag} DP-ADAGE Total-Budget (epsilon_total) Sweep Runner")
    print("=" * 70)
    print(f" Caps:        {args.caps}")
    print(f" Datasets:    {args.datasets}")
    print(f" Seeds:       {args.seeds}")
    print(f" Policies:    {args.policies}")
    print(f" Protocols:   {args.protocols}")
    print(f" Output Types:{args.output_types}")
    print(f" Horizon:     Benign T={args.T}, Attacker T={args.T_attacker}")
    print(f" Backend:     {args.backend.upper()} (Mock artifacts labeled SIMULATED)")
    print(f" Results Dir: {args.results_dir}")
    print(f" Git Commit:  {git_commit}")
    print("=" * 70)

    all_runs: List[Dict[str, Any]] = []
    completed_count = 0
    skipped_count = 0
    failed_count = 0

    grid = []
    for cap in args.caps:
        for ds in args.datasets:
            for seed in args.seeds:
                for pol in args.policies:
                    for proto in args.protocols:
                        for out_type in args.output_types:
                            grid.append((cap, ds, seed, pol, proto, out_type, False))
                            if args.ablation:
                                grid.append((cap, ds, seed, pol, proto, out_type, True))

    total_runs = len(grid)
    print(f"[*] Total planned runs in execution grid: {total_runs}\n")

    for idx, (cap, ds, seed, pol, proto, out_type, is_ab) in enumerate(grid, 1):
        horizon = args.T if pol == "benign" else args.T_attacker
        try:
            res = execute_single_run(
                cap=cap,
                dataset=ds,
                seed=seed,
                policy=pol,
                protocol=proto,
                output_type=out_type,
                horizon_T=horizon,
                args=args,
                results_base=results_base,
                git_commit=git_commit,
                is_ablation=is_ab,
            )
            all_runs.append(res)
            if res["status"] == "completed":
                completed_count += 1
            elif res["status"] == "skipped":
                skipped_count += 1
        except Exception as e:
            failed_count += 1
            print(f"[!] Error on run (cap={cap}, ds={ds}, seed={seed}, pol={pol}): {e}")
            all_runs.append({
                "run_id": f"{ds}_eps{cap}_seed{seed}_{pol}_{proto}_{out_type}",
                "status": "failed",
                "error": str(e),
            })

    # Save index of all runs
    index_file = results_base / "grid_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "git_commit": git_commit,
                "backend": args.backend,
                "total_runs": total_runs,
                "completed": completed_count,
                "skipped": skipped_count,
                "failed": failed_count,
                "runs": all_runs,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print(f" {tag} Sweep Execution Completed")
    print(f" Total Grid Runs: {total_runs}")
    print(f" Executed:        {completed_count}")
    print(f" Resumed/Skipped: {skipped_count}")
    print(f" Failed:          {failed_count}")
    print(f" Grid Index:      {index_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
