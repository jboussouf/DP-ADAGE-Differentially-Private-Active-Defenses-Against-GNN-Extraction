"""
evaluate_legitimate_user.py
===========================
Simulates and evaluates a *legitimate user* interacting with the DP-ADAGE
protected GNN API over 200 sequential queries.

Paper context
-------------
- Total Privacy Budget   (ε_total) = 200.0
- Max per-query budget   (ε_max)   = 10.0  [charged when τ ≤ τ_dead]
- Dead-zone threshold    (τ_dead)  = 0.05
- Budget exhaustion      after exactly 20 queries (200.0 / 10.0 = 20)
- Post-exhaustion behaviour ("Hard Floor"): the API continues to respond
  (no block / no exception) but returns a **uniform random label** instead of
  the model's true prediction. The user experiences a silent degradation of
  service quality, NOT a hard rejection.

Metrics collected per query
---------------------------
1. Rolling prediction accuracy  – 10-query sliding window.
2. Cumulative privacy budget     – monotonically non-decreasing up to ε_total.
3. API availability              – always True (no blocking in DP-ADAGE).

Output
------
- Console summary report.
- PDF figure: ``legitimate_user_evaluation.pdf``

Usage
-----
    python evaluate_legitimate_user.py

Dependencies: numpy, matplotlib (both in requirements.txt)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import deque

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Simulation hyper-parameters (must match the paper / DPAdage defaults)
# ---------------------------------------------------------------------------
NUM_QUERIES      = 200    # Total queries issued by the legitimate user
NUM_CLASSES      = 7      # Cora-style 7-class dataset
EPSILON_TOTAL    = 200.0  # Global privacy budget
EPSILON_MAX      = 10.0   # Per-query budget in the dead-zone (τ ≤ τ_dead)
TAU_DEAD         = 0.05   # Dead-zone threshold – legitimate user stays below this
BUDGET_EXHAUSTION_QUERY = int(EPSILON_TOTAL / EPSILON_MAX)  # = 20
ROLLING_WINDOW   = 10     # Window size for the rolling accuracy computation


# ===========================================================================
# Mock DP-ADAGE API
# ===========================================================================

class MockDPADAGEAPI:
    """
    Self-contained mock of the DP-ADAGE API that faithfully reproduces the
    hard-floor behaviour described in the paper (Algorithm 1, lines 10-16).

    For a legitimate user (τ ≤ τ_dead):
      - Each query costs ε_max from the global budget.
      - While budget remains: the API returns the **correct** class label.
      - After budget exhaustion (hard floor active):
          * The API still responds with HTTP-200 (no exception, no block).
          * The response is a **uniform random label** over the C classes,
            making it statistically useless for model extraction.

    Parameters
    ----------
    num_classes : int
        Number of output classes (7 for Cora).
    epsilon_total : float
        Global privacy budget shared across all queries from this user.
    epsilon_max : float
        Per-query budget deducted in the dead-zone.
    rng : np.random.Generator
        Seeded random number generator for reproducibility.
    """

    def __init__(
        self,
        num_classes: int     = NUM_CLASSES,
        epsilon_total: float = EPSILON_TOTAL,
        epsilon_max: float   = EPSILON_MAX,
        rng: np.random.Generator = rng,
    ) -> None:
        self.num_classes    = num_classes
        self.epsilon_total  = epsilon_total
        self.epsilon_max    = epsilon_max
        self.rng            = rng

        # Cumulative budget consumed so far
        self._epsilon_cum: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, ground_truth_label: int) -> dict:
        """
        Process a single query and return a response dict.

        The method mirrors the structure of ``DPAdage.query()`` in
        ``src/dp_adage.py``:
          - ``predicted_label``  : int  – the label returned to the caller.
          - ``epsilon_q``        : float – budget charged for this query.
          - ``epsilon_cum``      : float – cumulative budget after this query.
          - ``is_noise_floor``   : bool – True when the hard floor is active.
          - ``available``        : bool – always True (no hard blocking).

        Parameters
        ----------
        ground_truth_label : int
            The true class label for this node query.

        Returns
        -------
        dict
        """
        is_noise_floor = self._epsilon_cum >= self.epsilon_total - 1e-9

        if not is_noise_floor:
            # Budget still available – charge ε_max and return correct label.
            # Clamp so we never overshoot the total (mirrors RDPAccountant logic).
            approved = min(self.epsilon_max, self.epsilon_total - self._epsilon_cum)
            self._epsilon_cum += approved
            epsilon_q = approved

            # Check again: did this query push us to the floor?
            is_noise_floor = self._epsilon_cum >= self.epsilon_total - 1e-9

            predicted_label = ground_truth_label   # Accurate response
        else:
            # Hard floor is active: no additional budget charged (already at cap).
            epsilon_q = 0.0
            # Return a uniform random label – statistically useless to the user.
            predicted_label = int(self.rng.integers(0, self.num_classes))

        return {
            "predicted_label": predicted_label,
            "epsilon_q":       epsilon_q,
            "epsilon_cum":     min(self._epsilon_cum, self.epsilon_total),
            "is_noise_floor":  is_noise_floor,
            "available":       True,    # DP-ADAGE never blocks; it degrades gracefully
        }

    @property
    def epsilon_cum(self) -> float:
        """Current cumulative privacy budget consumed."""
        return self._epsilon_cum


# ===========================================================================
# Legitimate user simulation
# ===========================================================================

def generate_ground_truth_labels(n: int, num_classes: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generates synthetic ground-truth labels for N node queries.

    A realistic graph dataset has an unequal class distribution; we use
    a Dirichlet-drawn distribution to simulate this while keeping it
    reproducible.

    Parameters
    ----------
    n : int
        Number of queries.
    num_classes : int
        Number of classes.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray of shape (n,), dtype int
    """
    # Draw a class-frequency vector from Dirichlet(α=1) – mildly imbalanced.
    class_probs = rng.dirichlet(np.ones(num_classes))
    labels = rng.choice(num_classes, size=n, p=class_probs)
    return labels.astype(int)


def rolling_accuracy(correct_flags: list, window: int = ROLLING_WINDOW) -> np.ndarray:
    """
    Computes a rolling mean accuracy over a sliding window.

    For the first (window - 1) entries, the window is expanded rather than
    truncated so the curve starts immediately at query 1.

    Parameters
    ----------
    correct_flags : list of bool / int
        Sequence indicating whether each query was predicted correctly.
    window : int
        Width of the rolling window.

    Returns
    -------
    np.ndarray of shape (len(correct_flags),)
    """
    arr      = np.array(correct_flags, dtype=float)
    n        = len(arr)
    roll     = np.empty(n)
    buf      = deque(maxlen=window)
    for i, val in enumerate(arr):
        buf.append(val)
        roll[i] = np.mean(buf)
    return roll


def simulate_legitimate_user(num_queries: int = NUM_QUERIES) -> dict:
    """
    Runs the full 200-query simulation of a legitimate user interacting
    with the MockDPADAGEAPI.

    Returns
    -------
    dict with keys:
        ``ground_truth``   – array of true labels
        ``predictions``    – array of API-returned labels
        ``correct``        – bool array (True = correct prediction)
        ``epsilon_cum``    – array of cumulative budget after each query
        ``is_noise_floor`` – bool array (True = hard floor active)
        ``available``      – bool array (API availability per query)
        ``rolling_acc``    – float array (rolling 10-query accuracy)
    """
    api = MockDPADAGEAPI()

    # Synthetic ground-truth labels for the Cora-like 7-class dataset
    ground_truth = generate_ground_truth_labels(num_queries, NUM_CLASSES, rng)

    predictions    = np.empty(num_queries, dtype=int)
    epsilon_cums   = np.empty(num_queries, dtype=float)
    is_noise_floor = np.empty(num_queries, dtype=bool)
    available      = np.empty(num_queries, dtype=bool)
    correct_flags  = []

    for q_idx in range(num_queries):
        gt_label = int(ground_truth[q_idx])
        response = api.query(gt_label)

        predictions[q_idx]    = response["predicted_label"]
        epsilon_cums[q_idx]   = response["epsilon_cum"]
        is_noise_floor[q_idx] = response["is_noise_floor"]
        available[q_idx]      = response["available"]
        correct_flags.append(int(predictions[q_idx] == gt_label))

    roll_acc = rolling_accuracy(correct_flags, window=ROLLING_WINDOW)

    return {
        "ground_truth":   ground_truth,
        "predictions":    predictions,
        "correct":        np.array(correct_flags, dtype=bool),
        "epsilon_cum":    epsilon_cums,
        "is_noise_floor": is_noise_floor,
        "available":      available,
        "rolling_acc":    roll_acc,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def plot_results(results: dict, save_path: str = "legitimate_user_evaluation.pdf") -> None:
    """
    Generates a dual-axis figure showing:
      - Left  Y-axis : Rolling prediction accuracy (10-query window)
      - Right Y-axis : Cumulative privacy budget spent
      - Vertical dashed line at the budget-exhaustion query

    Parameters
    ----------
    results : dict
        Output of ``simulate_legitimate_user()``.
    save_path : str
        File path for the saved PDF.
    """
    # Use a clean, publication-quality style
    matplotlib.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.linewidth":   1.2,
        "axes.spines.top":  False,
        "figure.dpi":       150,
    })

    queries       = np.arange(1, NUM_QUERIES + 1)
    rolling_acc   = results["rolling_acc"]
    epsilon_cum   = results["epsilon_cum"]

    # Random-guess baseline for a C-class problem
    random_baseline = 1.0 / NUM_CLASSES   # ≈ 0.143 for Cora (7 classes)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    # ── Left axis: Rolling Accuracy ──────────────────────────────────────────
    color_acc      = "#2196F3"   # Material Blue
    color_baseline = "#90CAF9"   # Light Blue

    line_acc, = ax1.plot(
        queries, rolling_acc,
        color=color_acc, linewidth=2.0,
        label=f"Rolling Accuracy ({ROLLING_WINDOW}-query window)",
        zorder=3,
    )
    ax1.axhline(
        y=random_baseline,
        color=color_baseline, linewidth=1.4, linestyle=":",
        label=f"Random Guess Baseline ({random_baseline:.1%})",
        zorder=2,
    )

    ax1.set_xlabel("Number of Queries", fontsize=12, labelpad=8)
    ax1.set_ylabel("Rolling Prediction Accuracy", color=color_acc, fontsize=12, labelpad=8)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(-0.05, 1.15)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax1.set_xlim(0, NUM_QUERIES + 1)

    # Shade the "accurate" zone (queries 1–20) and "noise floor" zone (21–200)
    ax1.axvspan(0, BUDGET_EXHAUSTION_QUERY,
                alpha=0.08, color="#4CAF50", zorder=0, label="Accurate Phase")
    ax1.axvspan(BUDGET_EXHAUSTION_QUERY, NUM_QUERIES + 1,
                alpha=0.06, color="#F44336", zorder=0, label="Hard Floor Phase")

    # ── Right axis: Cumulative Privacy Budget ────────────────────────────────
    color_eps = "#E53935"   # Material Red

    ax2 = ax1.twinx()
    line_eps, = ax2.plot(
        queries, epsilon_cum,
        color=color_eps, linewidth=2.0, linestyle="--",
        label="Cumulative Privacy Budget (ε)",
        zorder=3,
    )
    ax2.set_ylabel("Cumulative Privacy Budget (ε)", color=color_eps,
                   fontsize=12, labelpad=8)
    ax2.tick_params(axis="y", labelcolor=color_eps)
    ax2.set_ylim(-5, EPSILON_TOTAL * 1.15)
    ax2.axhline(
        y=EPSILON_TOTAL,
        color=color_eps, linewidth=1.0, linestyle=":",
        alpha=0.6, zorder=2,
    )
    ax2.text(
        NUM_QUERIES * 0.55, EPSILON_TOTAL + 3,
        f"ε_total = {EPSILON_TOTAL:.0f}",
        color=color_eps, fontsize=9, alpha=0.8,
    )

    # ── Budget Exhaustion vertical marker ────────────────────────────────────
    vline = ax1.axvline(
        x=BUDGET_EXHAUSTION_QUERY,
        color="#FF6F00", linewidth=2.0, linestyle="--",
        label=f"Budget Exhaustion (Hard Floor)", zorder=4,
    )
    ax1.text(
        BUDGET_EXHAUSTION_QUERY + 1.5, 1.05,
        f"Budget Exhaustion\n(Hard Floor)\nQuery {BUDGET_EXHAUSTION_QUERY}",
        fontsize=8.5, color="#FF6F00",
        verticalalignment="top", zorder=5,
    )

    # ── Legend ───────────────────────────────────────────────────────────────
    # Merge handles from both axes into one legend placed at top-right
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1  + labels2,
        loc="upper right",
        fontsize=9,
        framealpha=0.85,
        edgecolor="#BDBDBD",
    )

    # ── Title ────────────────────────────────────────────────────────────────
    # Exact title as specified – no "Fig X:" prefix
    ax1.set_title(
        "Rolling Prediction Accuracy and Cumulative Privacy Budget",
        fontsize=13, fontweight="bold", pad=14,
    )

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"\n[✓] Figure saved → {save_path}\n")
    plt.close(fig)


# ===========================================================================
# Console summary report
# ===========================================================================

def print_summary_report(results: dict) -> None:
    """
    Prints a structured console report summarising key metrics.

    Parameters
    ----------
    results : dict
        Output of ``simulate_legitimate_user()``.
    """
    correct        = results["correct"]
    available      = results["available"]
    num_queries    = len(correct)

    # ── Phase boundaries ─────────────────────────────────────────────────────
    # Queries 1-20  → accurate phase  (0-indexed: 0-19)
    # Queries 21-200 → noise-floor phase (0-indexed: 20-199)
    q_split = BUDGET_EXHAUSTION_QUERY  # 20

    acc_phase1 = correct[:q_split].mean()         if q_split > 0            else float("nan")
    acc_phase2 = correct[q_split:].mean()         if q_split < num_queries  else float("nan")
    availability  = available.mean() * 100.0

    random_baseline = 1.0 / NUM_CLASSES

    separator = "─" * 60

    print()
    print(separator)
    print("  DP-ADAGE  –  Legitimate User Evaluation Summary")
    print(separator)
    print(f"  Dataset (simulated) : Cora-like ({NUM_CLASSES} classes)")
    print(f"  Total queries       : {num_queries}")
    print(f"  ε_total             : {EPSILON_TOTAL:.1f}")
    print(f"  ε_max (dead-zone)   : {EPSILON_MAX:.1f}")
    print(f"  τ_dead              : {TAU_DEAD}")
    print(f"  Budget exhaustion   : after query {BUDGET_EXHAUSTION_QUERY}")
    print(separator)
    print(f"  [Phase 1]  Queries  1 – {q_split:>3d}  (accurate phase)")
    print(f"             Avg accuracy  : {acc_phase1:.1%}  (expected ~100.0%)")
    print()
    print(f"  [Phase 2]  Queries {q_split+1:>3d} – {num_queries}  (hard floor phase)")
    print(f"             Avg accuracy  : {acc_phase2:.1%}  "
          f"(random baseline ≈ {random_baseline:.1%})")
    print()
    print(f"  [Overall]  API Availability : {availability:.1f}%")
    print(f"             (DP-ADAGE never blocks; it degrades gracefully)")
    print(separator)
    print()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("[*] Running legitimate user simulation …")
    results = simulate_legitimate_user(num_queries=NUM_QUERIES)

    print_summary_report(results)
    plot_results(results, save_path="legitimate_user_evaluation.pdf")
    print("[✓] Evaluation complete.")
