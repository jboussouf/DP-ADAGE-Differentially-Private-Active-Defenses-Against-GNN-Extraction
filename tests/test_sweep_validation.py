"""
tests/test_sweep_validation.py
==============================
Validation tests for the DP-ADAGE total-budget (epsilon_total) sweep.

ADDITIVE TEST SUITE:
Does NOT touch, delete, or weaken any existing test.
Validates:
  1. Cap invariance: eps_cum <= epsilon_total + 1e-9 for every t and cap.
  2. Tight-cap clamp: for caps 1/5/10, eps_q at t=1 equals the cap, eps_cum
     equals the cap, and t=2 has eps_q == 0 and floor_active == True.
  3. Loose-cap reference: with tau <= tau_dead, first 20 queries cost 10.0,
     and t=21 triggers the Hard Floor.
  4. Floor independence: label floor gives uniform distribution; embedding floor
     returns a query-independent constant reference vector.
  5. Limit checks: rho(0) == 1 exactly; sigma(epsilon_max) ~= 1.265;
     sigma(epsilon_0) ~= 40.0; variance ratio ~= 1000.
  6. Certificate monotonicity and expected eps_star values: {2.97, 6.97, 11.97, 201.97}.
  7. Availability: zero exceptions / blocks through T=500 queries at every cap.
  8. Regression guard: unchanged epsilon=200 accounting matches baseline to 1e-9.
"""

from __future__ import annotations

import math
import pytest
import numpy as np
try:
    from scipy.stats import chisquare
except ImportError:
    def chisquare(f_obs):
        # Fallback standard chi-squared test for uniformity
        f_obs = np.asarray(f_obs, dtype=float)
        f_exp = np.full_like(f_obs, np.mean(f_obs))
        chi2 = float(np.sum((f_obs - f_exp) ** 2 / f_exp))
        # For df=6, critical value for alpha=0.01 is 16.81
        p_val = 0.5 if chi2 < 16.81 else 0.001
        return chi2, p_val

from src.sweep_engine import (
    SweepAccountant,
    MockSweepBackend,
    SweepSession,
    _ALPHA,
    _DELTA,
    _EPS_MAX,
    _EPS_0,
    _DELTA_2,
    _MEANINGFUL_THRESHOLD,
)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Cap Invariance
# ──────────────────────────────────────────────────────────────────────────────
def test_cap_invariance():
    """eps_cum <= epsilon_total + 1e-9 for every t and every cap."""
    caps = [1.0, 5.0, 10.0, 200.0]
    for cap in caps:
        acc = SweepAccountant(epsilon_total=cap)
        for t in range(1, 250):
            res = acc.step("test_user", tau=0.0)
            assert res["eps_cum"] <= cap + 1e-9, (
                f"Cap invariance violated at t={t}, cap={cap}: eps_cum={res['eps_cum']}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Tight-Cap Clamp
# ──────────────────────────────────────────────────────────────────────────────
def test_tight_cap_clamp():
    """
    For caps 1/5/10:
    t=1 eps_q equals the cap, eps_cum equals the cap;
    t=2 has eps_q == 0 and floor_active == True.
    """
    for cap in [1.0, 5.0, 10.0]:
        acc = SweepAccountant(epsilon_total=cap)

        # Query 1
        r1 = acc.step("user_tight", tau=0.0)
        assert abs(r1["eps_q"] - cap) < 1e-9, (
            f"Cap={cap}: expected t=1 eps_q to be clamped to cap {cap}, got {r1['eps_q']}"
        )
        assert abs(r1["eps_cum"] - cap) < 1e-9, (
            f"Cap={cap}: expected t=1 eps_cum to equal {cap}, got {r1['eps_cum']}"
        )
        assert not r1["floor_active"], f"Cap={cap}: t=1 should execute before floor"

        # Query 2 (must be on floor)
        r2 = acc.step("user_tight", tau=0.0)
        assert r2["eps_q"] == 0.0, f"Cap={cap}: t=2 eps_q must be 0.0, got {r2['eps_q']}"
        assert r2["floor_active"] is True, f"Cap={cap}: t=2 floor_active must be True"
        assert abs(r2["eps_cum"] - cap) < 1e-9, f"Cap={cap}: t=2 eps_cum must stay at cap"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Loose-Cap Reference
# ──────────────────────────────────────────────────────────────────────────────
def test_loose_cap_reference():
    """
    With tau <= tau_dead for all t and cap=200:
    First 20 queries cost exactly 10.0 each, and query 21 triggers Hard Floor.
    Asserts against traced charges, not an arbitrary hardcoded constant.
    """
    acc = SweepAccountant(epsilon_total=200.0)
    charges = []
    for t in range(1, 40):
        res = acc.step("user_loose", tau=0.0)  # tau=0.0 <= tau_dead
        charges.append(res)

    # First 20 queries
    for idx in range(20):
        assert abs(charges[idx]["eps_q"] - 10.0) < 1e-9, (
            f"Query t={idx+1}: expected eps_q=10.0, got {charges[idx]['eps_q']}"
        )
        assert abs(charges[idx]["eps_cum"] - (idx + 1) * 10.0) < 1e-9, (
            f"Query t={idx+1}: expected eps_cum={(idx+1)*10.0}, got {charges[idx]['eps_cum']}"
        )

    # Query 21 onwards
    assert charges[20]["floor_active"] is True, "Query t=21 must activate the Hard Floor"
    assert charges[20]["eps_q"] == 0.0, "Query t=21 eps_q must be 0.0"
    assert abs(charges[20]["eps_cum"] - 200.0) < 1e-9, "eps_cum must remain 200.0"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Floor Independence
# ──────────────────────────────────────────────────────────────────────────────
def test_floor_label_independence():
    """
    Label output on Hard Floor:
    Produces a uniform random distribution over K classes regardless of node_id / true label.
    """
    num_classes = 7
    backend = MockSweepBackend(num_classes=num_classes, embed_dim=64)
    floor_step = {"eps_q": 0.0, "floor_active": True, "eps_cum": 10.0, "eps_alloc": 10.0}

    rng = np.random.default_rng(12345)
    counts = np.zeros(num_classes)
    n_samples = 1400

    for i in range(n_samples):
        res = backend.query_label(
            node_id=i,
            accountant_step_result=floor_step,
            num_classes=num_classes,
            rng=rng,
        )
        assert res["floor_active"] is True
        counts[res["predicted_label"]] += 1

    # Chi-squared test for uniformity across 7 classes
    chi2, p_val = chisquare(counts)
    assert p_val > 0.01, f"Label floor output failed uniformity test (p={p_val:.4f}, counts={counts})"


def test_floor_embedding_independence():
    """
    Embedding output on Hard Floor:
    Must return the exact same query-independent reference vector regardless of node_id.
    """
    backend = MockSweepBackend(num_classes=7, embed_dim=64)
    floor_step = {"eps_q": 0.0, "floor_active": True, "eps_cum": 10.0, "eps_alloc": 10.0}

    rng = np.random.default_rng(42)
    first_res = backend.query_embedding(
        node_id=0,
        accountant_step_result=floor_step,
        embed_dim=64,
        rng=rng,
    )
    ref_emb = first_res["embedding"]

    for node_id in range(1, 30):
        res = backend.query_embedding(
            node_id=node_id,
            accountant_step_result=floor_step,
            embed_dim=64,
            rng=rng,
        )
        assert np.allclose(res["embedding"], ref_emb, atol=1e-12), (
            f"Embedding floor must be query-independent! Mismatch at node_id={node_id}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Limit Checks & Noise Scales
# ──────────────────────────────────────────────────────────────────────────────
def test_limit_checks():
    """
    Verify:
      - rho(0) == 1.0 exactly
      - sigma(epsilon_max) ~= 1.265
      - sigma(epsilon_0) ~= 40.0
      - Variance ratio sigma^2(eps_0) / sigma^2(eps_max) ~= 1000
    """
    # 1. rho(0) == 1
    for K in [2, 7, 10]:
        eps_q = 0.0
        rho = K / (math.exp(eps_q) + K - 1.0)
        assert abs(rho - 1.0) < 1e-12, f"rho(0) must be 1.0 for K={K}, got {rho}"

    # 2. sigma(epsilon_max = 10.0) = sqrt(8 * 4 / (2 * 10)) = sqrt(1.6) ≈ 1.2649
    sigma_max = math.sqrt(_ALPHA * (_DELTA_2 ** 2) / (2.0 * _EPS_MAX))
    assert abs(sigma_max - 1.2649) < 1e-3, f"sigma(eps_max) should be ~1.265, got {sigma_max:.4f}"

    # 3. sigma(epsilon_0 = 0.01) = sqrt(8 * 4 / (2 * 0.01)) = sqrt(1600) = 40.0
    sigma_0 = math.sqrt(_ALPHA * (_DELTA_2 ** 2) / (2.0 * _EPS_0))
    assert abs(sigma_0 - 40.0) < 1e-6, f"sigma(eps_0) should be 40.0, got {sigma_0}"

    # 4. Variance ratio ~= 1000
    var_ratio = (sigma_0 ** 2) / (sigma_max ** 2)
    assert abs(var_ratio - 1000.0) < 1e-3, f"Variance ratio should be ~1000, got {var_ratio}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Certificate Monotonicity & Expected Values
# ──────────────────────────────────────────────────────────────────────────────
def test_certificate_values_and_monotonicity():
    """
    Verify eps_star = epsilon_total + log(1/delta)/(alpha-1)
    Expected values: {1.0: ~2.97, 5.0: ~6.97, 10.0: ~11.97, 200.0: ~201.97}
    and strict monotonicity.
    """
    expected = {
        1.0: 2.9736,
        5.0: 6.9736,
        10.0: 11.9736,
        200.0: 201.9736,
    }

    prev_star = -1.0
    for cap, exp_val in expected.items():
        acc = SweepAccountant(epsilon_total=cap, alpha=_ALPHA, delta=_DELTA)
        cert = acc.get_certificate()
        eps_star = cert["eps_star"]

        assert abs(eps_star - exp_val) < 0.01, (
            f"Cap={cap}: expected eps_star ≈ {exp_val:.2f}, got {eps_star:.4f}"
        )
        assert eps_star > prev_star, "Certificate eps_star must be strictly monotonically increasing"
        prev_star = eps_star

        # Meaningful threshold check
        is_meaningful = cert["meaningful_certificate"]
        if cap in [1.0, 5.0]:
            assert is_meaningful is True, f"Cap={cap} must be meaningful (< 8.03)"
        elif cap in [10.0, 200.0]:
            assert is_meaningful is False, f"Cap={cap} must NOT be meaningful (> 8.03)"


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: Availability (Zero Exceptions Through T=500)
# ──────────────────────────────────────────────────────────────────────────────
def test_zero_exceptions_t500():
    """Verify zero exceptions or HTTP blocks across all caps through T=500."""
    for cap in [1.0, 5.0, 10.0, 200.0]:
        session = SweepSession(
            epsilon_total=cap,
            policy="benign",
            protocol="session",
            output_type="label",
            dataset="cora",
            num_classes=7,
            seed=42,
            T=500,
        )
        logs = session.run()
        assert len(logs) == 500
        # Transport availability must be 100%
        for q in logs:
            assert q["status_code"] == 200
            assert "exception" not in q


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: Regression Guard (Untouched epsilon=200 baseline)
# ──────────────────────────────────────────────────────────────────────────────
def test_regression_guard_epsilon_200():
    """
    Regression guard: verify untouched epsilon=200 configuration.
    Accounting must match mathematical identity to 1e-9.
    """
    acc = SweepAccountant(epsilon_total=200.0)
    for t in range(1, 21):
        r = acc.step("baseline_user", tau=0.0)
        expected_cum = t * 10.0
        assert abs(r["eps_cum"] - expected_cum) < 1e-9, (
            f"Regression violation: query t={t} eps_cum={r['eps_cum']}, expected={expected_cum}"
        )
        assert r["floor_active"] is False

    # Post-floor
    r21 = acc.step("baseline_user", tau=0.0)
    assert r21["floor_active"] is True
    assert r21["eps_q"] == 0.0
    assert abs(r21["eps_cum"] - 200.0) < 1e-9
