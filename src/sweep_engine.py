"""
sweep_engine.py
===============
Core accounting, mock backend, and session orchestrator for the DP-ADAGE
total-budget (epsilon_total) sweep experiment.

This module is ADDITIVE — it does not modify any existing source file.
The default epsilon_total = 200.0 experiment is unaffected.

Scientific constants (must match the paper exactly):
  alpha    = 8,     delta  = 1e-6
  eps_max  = 10.0,  eps_0  = 0.01
  k_sharp  = 20,    tau_0  = 0.30,  tau_dead = 0.05
  clip C=B = 1.0,   Delta_2 = 2*C  = 2.0

Accounting (exact, Eqs. 9-11):
  eps_alloc = eps_max                              if tau <= tau_dead
              eps_max + (eps_0-eps_max)*g(tau)     otherwise
  eps_q     = min(eps_alloc, max(0, eps_total - eps_cum))
  eps_cum  += eps_q
  floor_active = True when budget was exhausted BEFORE this query

Sigma (only pre-floor):
  sigma = sqrt(alpha * Delta_2^2 / (2 * eps_q))

Hard Floor:
  label output    -> uniform random over K classes   (rho = 1)
  embedding output -> fixed reference vector e_ref   (constant)

Certificate (Theorem 4.4):
  eps_star = eps_total + log(1/delta) / (alpha - 1)
  meaningful_certificate = (eps_total <= log(e^10) - log(1/delta)/(alpha-1))
                         = (eps_total <= ~8.034)
"""

from __future__ import annotations

import math
import time
import numpy as np
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────
_ALPHA     = 8.0
_DELTA     = 1e-6
_EPS_MAX   = 10.0
_EPS_0     = 0.01
_K_SHARP   = 20.0
_TAU_0     = 0.30
_TAU_DEAD  = 0.05
_CLIP_C    = 1.0
_CLIP_B    = 1.0
_DELTA_2   = 2.0 * _CLIP_C   # L2 sensitivity = 2.0

# Meaningful-budget threshold: eps_total <= log(e^10) - log(1/delta)/(alpha-1)
_LOG_M     = 10.0              # log(e^10) = 10
_CERT_TERM = math.log(1.0 / _DELTA) / (_ALPHA - 1.0)   # ≈ 1.9736
_MEANINGFUL_THRESHOLD = _LOG_M - _CERT_TERM             # ≈ 8.0264


# ──────────────────────────────────────────────────────────────────────────────
# SweepAccountant
# ──────────────────────────────────────────────────────────────────────────────

class SweepAccountant:
    """
    Per-session privacy budget accountant for the epsilon-total sweep.

    Implements the exact accounting from DP-ADAGE Algorithm 1 (lines 9-11)
    with explicit Hard Floor handling.

    Parameters
    ----------
    epsilon_total : float
        Total privacy budget for the session.
    epsilon_max : float
        Maximum per-query budget (charged when tau <= tau_dead).
    epsilon_0 : float
        Minimum per-query budget (strict regime, tau -> 1).
    k_sharp : float
        Sharpness parameter for the sigmoid allocator.
    tau_0 : float
        Inflection point of the sigmoid.
    tau_dead : float
        Dead-zone threshold; queries with tau <= tau_dead pay epsilon_max.
    alpha : float
        RDP order.
    delta : float
        Failure probability for DP conversion.
    """

    def __init__(
        self,
        epsilon_total: float = 200.0,
        epsilon_max:   float = _EPS_MAX,
        epsilon_0:     float = _EPS_0,
        k_sharp:       float = _K_SHARP,
        tau_0:         float = _TAU_0,
        tau_dead:      float = _TAU_DEAD,
        alpha:         float = _ALPHA,
        delta:         float = _DELTA,
    ) -> None:
        self.epsilon_total = epsilon_total
        self.epsilon_max   = epsilon_max
        self.epsilon_0     = epsilon_0
        self.k_sharp       = k_sharp
        self.tau_0         = tau_0
        self.tau_dead      = tau_dead
        self.alpha         = alpha
        self.delta         = delta
        self._cumulative: Dict[str, float] = {}

    # ── internal helpers ──────────────────────────────────────────────────────

    def _sigmoid(self, tau: float) -> float:
        """g̃(τ) = 1 / (1 + exp(-k*(τ - τ_0)))  (Eq. 7)."""
        return 1.0 / (1.0 + math.exp(-self.k_sharp * (tau - self.tau_0)))

    def _get_cum(self, user_id: str) -> float:
        return self._cumulative.get(user_id, 0.0)

    # ── public API ────────────────────────────────────────────────────────────

    def allocate(self, tau: float) -> float:
        """
        Returns the per-query privacy budget allocation eps_alloc(tau).

        For tau <= tau_dead: eps_alloc = epsilon_max (dead-zone, Eq. 8a).
        Otherwise:           eps_alloc = eps_max + (eps_0 - eps_max)*g̃(tau).
        """
        if tau <= self.tau_dead:
            return self.epsilon_max
        g = self._sigmoid(tau)
        return self.epsilon_max + (self.epsilon_0 - self.epsilon_max) * g

    def step(self, user_id: str, tau: float) -> Dict[str, Any]:
        """
        Process one query for ``user_id`` with diversity score ``tau``.

        Returns
        -------
        dict with keys:
            eps_alloc   : requested allocation (before clamping)
            eps_q       : actually approved budget (0.0 on floor)
            eps_cum     : cumulative budget AFTER this query
            floor_active: True if budget was exhausted BEFORE this query
            remaining   : budget remaining after this query
        """
        eps_alloc = self.allocate(tau)
        cum_before = self._get_cum(user_id)

        # Hard floor check: exhausted BEFORE this query?
        floor_active = cum_before >= self.epsilon_total - 1e-9

        if floor_active:
            # Floor: charge nothing, cumulative stays at cap
            eps_q   = 0.0
            eps_cum = min(cum_before, self.epsilon_total)
        else:
            # Clamp to remaining budget (Eq. 9)
            remaining_before = max(0.0, self.epsilon_total - cum_before)
            eps_q   = min(eps_alloc, remaining_before)
            eps_cum = cum_before + eps_q
            self._cumulative[user_id] = eps_cum

        remaining = max(0.0, self.epsilon_total - eps_cum)
        return {
            "eps_alloc":    eps_alloc,
            "eps_q":        eps_q,
            "eps_cum":      eps_cum,
            "floor_active": floor_active,
            "remaining":    remaining,
        }

    def is_exhausted(self, user_id: str) -> bool:
        """True if the user's cumulative budget equals epsilon_total."""
        return self._get_cum(user_id) >= self.epsilon_total - 1e-9

    def get_certificate(self) -> Dict[str, Any]:
        """
        Returns the (epsilon*, delta)-DP certificate (Theorem 4.4).

        eps_star = epsilon_total + log(1/delta) / (alpha - 1)

        The meaningful-budget threshold is derived by rearranging
        Proposition 1 of the paper:
          eps_star <= log(M)   (where M = e^10)
          => epsilon_total + cert_term <= 10
          => epsilon_total <= 10 - cert_term  (~8.034)
        Note: we SUBTRACT cert_term from log(M), not divide — the rearrangement
        is additive, not multiplicative.
        """
        cert_term = math.log(1.0 / self.delta) / (self.alpha - 1.0)
        eps_star  = self.epsilon_total + cert_term
        meaningful = self.epsilon_total <= (_LOG_M - cert_term)
        return {
            "eps_star":              eps_star,
            "meaningful_certificate": meaningful,
            "epsilon_total":         self.epsilon_total,
            "delta":                 self.delta,
            "alpha":                 self.alpha,
            "cert_term":             cert_term,
            "meaningful_threshold":  _LOG_M - cert_term,
        }

    def reset_user(self, user_id: str) -> None:
        """Reset tracking for a specific user."""
        self._cumulative.pop(user_id, None)

    def reset_all(self) -> None:
        """Reset all users."""
        self._cumulative.clear()


# ──────────────────────────────────────────────────────────────────────────────
# MockSweepBackend
# ──────────────────────────────────────────────────────────────────────────────

class MockSweepBackend:
    """
    Stateless mock backend for the sweep (no trained GNN required).

    All outputs are labeled [SIMULATED] in the session log.
    A mock run MUST NOT be described as a reproduction of the paper.

    Parameters
    ----------
    num_classes : int
        Number of output classes (7 for Cora).
    embed_dim : int
        Embedding dimension for embedding output type.
    seed : int, optional
        Seed for the internal RNG (query-level randomness uses caller-provided rng).
    """

    def __init__(
        self,
        num_classes: int = 7,
        embed_dim:   int = 64,
        seed:        Optional[int] = None,
    ) -> None:
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        # Reference embedding: constant vector 1/num_classes in each dim
        self._e_ref = np.full(embed_dim, 1.0 / num_classes, dtype=np.float64)

    def get_true_label(self, node_id: int) -> int:
        """Deterministic true label from node_id (node_id % num_classes)."""
        return int(node_id % self.num_classes)

    def get_reference_embedding(self, embed_dim: Optional[int] = None) -> np.ndarray:
        """
        Returns the fixed reference embedding for the Hard Floor.
        Same vector regardless of node_id or query content.
        """
        dim = embed_dim if embed_dim is not None else self.embed_dim
        return np.full(dim, 1.0 / self.num_classes, dtype=np.float64)

    def _compute_sigma(self, eps_q: float) -> float:
        """σ = sqrt(alpha * Delta_2^2 / (2 * eps_q)). Must NOT be called when eps_q=0."""
        if eps_q <= 0.0:
            raise ValueError(
                "sigma must not be computed when eps_q=0 (Hard Floor active). "
                "Branch to floor BEFORE calling this."
            )
        return math.sqrt(_ALPHA * (_DELTA_2 ** 2) / (2.0 * eps_q))

    def query_label(
        self,
        node_id:               int,
        accountant_step_result: Dict[str, Any],
        num_classes:           Optional[int] = None,
        rng:                   Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        Return a label prediction for ``node_id``.

        On Hard Floor (eps_q == 0): uniform random label (rho = 1).
        Pre-floor: Randomized Response with rho = K / (exp(eps_q) + K - 1).

        Parameters
        ----------
        node_id : int
        accountant_step_result : dict
            Output of SweepAccountant.step().
        num_classes : int, optional
            Overrides self.num_classes.
        rng : np.random.Generator, optional
            Caller-provided RNG for reproducibility.
        """
        if rng is None:
            rng = np.random.default_rng()
        K          = num_classes if num_classes is not None else self.num_classes
        true_label = self.get_true_label(node_id)
        eps_q      = accountant_step_result["eps_q"]
        floor      = accountant_step_result["floor_active"]

        if floor or eps_q <= 0.0:
            # Hard Floor: rho(0) = K/(exp(0)+K-1) = K/K = 1 -> always random
            predicted = int(rng.integers(0, K))
            sigma_or_rho = 1.0
        else:
            # Randomized Response (Eq. 18-19 from paper / perturbation.py)
            # rho = K / (exp(eps_q) + K - 1)
            eps_clamped  = min(eps_q, 50.0)   # avoid overflow
            rho          = K / (math.exp(eps_clamped) + K - 1.0)
            sigma_or_rho = rho
            if rng.random() < rho:
                predicted = int(rng.integers(0, K))  # random
            else:
                predicted = true_label               # correct

        return {
            "predicted_label": predicted,
            "true_label":      true_label,
            "floor_active":    floor or (eps_q <= 0.0),
            "sigma_or_rho":    sigma_or_rho,
            "raw_output":      f"label:{predicted}",
            "raw_output_type": "label",
        }

    def query_embedding(
        self,
        node_id:               int,
        accountant_step_result: Dict[str, Any],
        embed_dim:             Optional[int] = None,
        rng:                   Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        Return an embedding for ``node_id``.

        On Hard Floor: fixed reference vector e_ref (query-independent).
        Pre-floor: Gaussian-perturbed embedding with sigma calibrated to eps_q.

        Note: the floor accuracy for embeddings equals the frequency of the
        mode class in the pool (argmax(e_ref) is fixed), NOT necessarily 1/K.
        """
        if rng is None:
            rng = np.random.default_rng()
        dim        = embed_dim if embed_dim is not None else self.embed_dim
        true_label = self.get_true_label(node_id)
        eps_q      = accountant_step_result["eps_q"]
        floor      = accountant_step_result["floor_active"]

        if floor or eps_q <= 0.0:
            # Hard Floor: return the fixed constant reference vector
            emb   = self.get_reference_embedding(dim)
            sigma = 0.0
        else:
            # sigma = sqrt(alpha * Delta_2^2 / (2 * eps_q))
            sigma = self._compute_sigma(eps_q)
            # Mock "clean" embedding: one-hot-like vector for the true class
            clean_emb = np.zeros(dim, dtype=np.float64)
            # Place signal at class-indexed positions
            indices = [i for i in range(dim) if i % self.num_classes == true_label]
            if indices:
                clean_emb[indices] = 1.0 / len(indices)
            # Clip to unit norm (L2 clipping, Eq. 15)
            norm = np.linalg.norm(clean_emb)
            if norm > _CLIP_B:
                clean_emb = clean_emb * (_CLIP_B / norm)
            # Add Gaussian noise
            noise = rng.standard_normal(dim) * sigma
            emb   = clean_emb + noise

        # Predicted label = argmax of embedding (first K dims as class scores)
        class_scores   = emb[:self.num_classes] if dim >= self.num_classes else emb
        predicted_label = int(np.argmax(class_scores))

        return {
            "embedding":       emb,
            "predicted_label": predicted_label,
            "true_label":      true_label,
            "floor_active":    floor or (eps_q <= 0.0),
            "sigma":           sigma,
            "sigma_or_rho":    sigma,
            "raw_output":      f"emb:dim{dim}",
            "raw_output_type": "embedding",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Tau simulation per policy (mock — no real GNN)
# ──────────────────────────────────────────────────────────────────────────────

def _simulate_tau(t: int, policy: str, T: int, K: int = 10) -> float:
    """
    Simulate the query diversity score tau for each policy.

    benign:       tau = 0.0 always (localized, stays in dead-zone)
    uncertainty:  tau increases gradually: min(1.0, t / (2*T))
    budget_aware: tau cycles per community; tau = min(1.0, (t//5) / K)
    """
    if policy == "benign":
        return 0.0
    elif policy == "uncertainty":
        # Simulates an attacker querying progressively diverse nodes
        return min(1.0, float(t) / (2.0 * T))
    elif policy == "budget_aware":
        # Simulates cycling communities (every 5 queries, enter a new one)
        return min(1.0, float(t // 5) / K)
    else:
        raise ValueError(f"Unknown policy: {policy!r}")


# ──────────────────────────────────────────────────────────────────────────────
# SweepSession
# ──────────────────────────────────────────────────────────────────────────────

class SweepSession:
    """
    Orchestrates a single (cap, policy, protocol, output_type, dataset, seed) run.

    Parameters
    ----------
    epsilon_total : float
        Budget cap for this session.
    policy : str
        One of 'benign', 'uncertainty', 'budget_aware'.
    protocol : str
        'session' (enforced Hard Floor) or 'fresh' (non-binding ledger).
    output_type : str
        'label' or 'embedding'.
    dataset : str
        Dataset name (metadata only; real data not loaded in mock mode).
    num_classes : int
        Number of output classes.
    seed : int
        Random seed for reproducibility.
    T : int
        Total number of queries to issue.
    embed_dim : int
        Embedding dimension (for embedding output type).
    num_communities : int
        Number of communities for tau simulation.
    epsilon_max, epsilon_0, k_sharp, tau_0, tau_dead, alpha, delta : float
        Hyperparameters (defaults match the paper).
    """

    def __init__(
        self,
        epsilon_total:   float = 200.0,
        policy:          str   = "benign",
        protocol:        str   = "session",
        output_type:     str   = "label",
        dataset:         str   = "cora",
        num_classes:     int   = 7,
        seed:            int   = 42,
        T:               int   = 200,
        embed_dim:       int   = 64,
        num_communities: int   = 10,
        epsilon_max:     float = _EPS_MAX,
        epsilon_0:       float = _EPS_0,
        k_sharp:         float = _K_SHARP,
        tau_0:           float = _TAU_0,
        tau_dead:        float = _TAU_DEAD,
        alpha:           float = _ALPHA,
        delta:           float = _DELTA,
    ) -> None:
        self.epsilon_total   = epsilon_total
        self.policy          = policy
        self.protocol        = protocol
        self.output_type     = output_type
        self.dataset         = dataset
        self.num_classes     = num_classes
        self.seed            = seed
        self.T               = T
        self.embed_dim       = embed_dim
        self.num_communities = num_communities

        self._accountant = SweepAccountant(
            epsilon_total = epsilon_total,
            epsilon_max   = epsilon_max,
            epsilon_0     = epsilon_0,
            k_sharp       = k_sharp,
            tau_0         = tau_0,
            tau_dead      = tau_dead,
            alpha         = alpha,
            delta         = delta,
        )
        self._backend = MockSweepBackend(
            num_classes = num_classes,
            embed_dim   = embed_dim,
        )
        self._rng = np.random.default_rng(seed)

    def _user_id(self) -> str:
        return f"{self.dataset}_eps{self.epsilon_total}_{self.policy}_{self.protocol}_s{self.seed}"

    def run(self) -> List[Dict[str, Any]]:
        """
        Execute T sequential queries and return per-query log records.

        Each record contains:
          t, node_id, community_id, tau, eps_alloc, eps_q, eps_cum,
          sigma_or_rho, raw_output, predicted_label, true_label,
          floor_active, status_code, latency_ms

        All records from the mock backend have status_code=200 (no blocking).
        The session NEVER raises an exception — the Hard Floor degrades service
        gracefully without terminating the session.
        """
        user_id = self._user_id()
        logs: List[Dict[str, Any]] = []

        for t in range(1, self.T + 1):
            t_start = time.perf_counter()

            # Node and community assignment (mock)
            node_id      = t - 1          # 0-indexed
            community_id = node_id % self.num_communities

            # Tau simulation
            tau = _simulate_tau(t, self.policy, self.T, self.num_communities)

            # Accounting step
            # 'fresh' protocol uses a throwaway accountant (non-binding ledger)
            if self.protocol == "fresh":
                # Non-binding: always allocate, never enforce hard floor
                eps_alloc = self._accountant.allocate(tau)
                eps_q     = eps_alloc   # no clamping
                # Track cumulative only for logging (but don't enforce floor)
                eps_cum_logged = logs[-1]["eps_cum"] + eps_q if logs else eps_q
                step_result = {
                    "eps_alloc":    eps_alloc,
                    "eps_q":        eps_q,
                    "eps_cum":      eps_cum_logged,
                    "floor_active": False,   # no floor in fresh protocol
                    "remaining":    max(0.0, self.epsilon_total - eps_cum_logged),
                }
            else:
                # session protocol: hard floor enforced
                step_result = self._accountant.step(user_id, tau)

            # Query the (mock) backend
            floor = step_result["floor_active"]
            eps_q = step_result["eps_q"]

            if self.output_type == "label":
                response = self._backend.query_label(
                    node_id                = node_id,
                    accountant_step_result = step_result,
                    num_classes            = self.num_classes,
                    rng                    = self._rng,
                )
                sigma_or_rho = response["sigma_or_rho"]
                raw_output   = response["raw_output"]
                predicted    = response["predicted_label"]
                true_label   = response["true_label"]
            else:  # embedding
                response = self._backend.query_embedding(
                    node_id                = node_id,
                    accountant_step_result = step_result,
                    embed_dim              = self.embed_dim,
                    rng                    = self._rng,
                )
                sigma_or_rho = response["sigma_or_rho"]
                raw_output   = response["raw_output"]
                predicted    = response["predicted_label"]
                true_label   = response["true_label"]

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            record = {
                "t":              t,
                "node_id":        node_id,
                "community_id":   community_id,
                "tau":            tau,
                "eps_alloc":      step_result["eps_alloc"],
                "eps_q":          eps_q,
                "eps_cum":        step_result["eps_cum"],
                "sigma_or_rho":   sigma_or_rho,
                "raw_output":     raw_output,
                "predicted_label": predicted,
                "true_label":     true_label,
                "floor_active":   floor,
                "status_code":    200,   # DP-ADAGE never blocks
                "latency_ms":     round(latency_ms, 4),
                "simulated":      True,
            }
            logs.append(record)

        return logs
