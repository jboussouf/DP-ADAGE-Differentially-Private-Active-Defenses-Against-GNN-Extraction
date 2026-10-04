"""
DP-ADAGE Source Package
=======================
Differentially Private Active Defenses Against GNN Extraction

Package structure:
    community_tracker    - Louvain community detection + per-user τ scoring
    budget_allocator     - Sigmoid τ → ε(τ) mapping
    rdp_accountant       - RDP composition + hard cap enforcement
    perturbation         - Gaussian mechanism (logit & embedding)
    dp_adage             - Full pipeline orchestration
    adage_baseline       - Original heuristic ADAGE defense
    gnn_models           - GCN, GAT, GraphSAGE, GIN, GraphTransformer
    attacker             - Surrogate model extraction attack
    utils                - Dataset loading, metrics, helpers
"""

from .community_tracker import CommunityTracker
from .budget_allocator import BudgetAllocator
from .rdp_accountant import RDPAccountant
from .perturbation import GaussianPerturbationMechanism
from .dp_adage import DPAdage
from .adage_baseline import OriginalADAGE
from .gnn_models import GCN, GAT, GraphSAGE, GIN
from .attacker import SurrogateAttacker
from .utils import (
    load_dataset,
    evaluate_downstream,
    compute_kl_divergence,
    set_seed,
)

__version__ = "1.0.0"
__author__ = "Jamal Boussouf"
__institution__ = "University Mohammed First, Oujda, Morocco"

__all__ = [
    "CommunityTracker",
    "BudgetAllocator",
    "RDPAccountant",
    "GaussianPerturbationMechanism",
    "DPAdage",
    "OriginalADAGE",
    "GCN",
    "GAT",
    "GraphSAGE",
    "GIN",
    "SurrogateAttacker",
    "load_dataset",
    "evaluate_downstream",
    "compute_kl_divergence",
    "set_seed",
]
