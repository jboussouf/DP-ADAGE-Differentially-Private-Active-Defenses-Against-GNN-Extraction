"""
BudgetAllocator
===============
Component 2 of the DP-ADAGE pipeline.

Responsible for mapping query diversity τ to a per-query privacy budget ε(τ)
using a sigmoid shaping function.

Equations:
  g(τ) = 1 / (1 + exp(-k*(τ - τ_0)))  (Eq. 7)
  ε(τ) = ε_max + (ε_0 - ε_max) * g(τ) (Eq. 8)

For benign users (τ → 0), ε(τ) → ε_max (relaxed privacy, low noise).
For attackers (τ → 1), ε(τ) → ε_0 (strict privacy, high noise).
"""
import math
import numpy as np
from typing import Optional

class BudgetAllocator:
    # FIX 2: epsilon_max was too large (2.0), causing the 5.0 cap to be hit in 3 queries.
    # Set to 0.07 so that 0.07 * ~70-80 queries = ~5.0 budget limit.
    def __init__(self, epsilon_0: float = 0.01, epsilon_max: float = 10.0, k: float = 20.0, tau_0: float = 0.30, tau_dead: float = 0.05):
        self.epsilon_0 = epsilon_0
        self.epsilon_max = epsilon_max
        self.k = k
        self.tau_0 = tau_0
        self.tau_dead = tau_dead
        
    def sigmoid(self, tau: float) -> float:
        """
        Computes the sigmoid allocation weighting g(τ) (Eq. 7).
        """
        return 1.0 / (1.0 + math.exp(-self.k * (tau - self.tau_0)))
        
    def allocate(self, tau: float) -> float:
        """
        Returns the per-query budget ε(τ) (Eq. 8).
        """
        if tau <= self.tau_dead:
            return self.epsilon_max
        else:
            g_tau = self.sigmoid(tau)
            return self.epsilon_max + (self.epsilon_0 - self.epsilon_max) * g_tau
        
    def noise_scale(self, epsilon_q: float, alpha: float, delta_sensitivity: float) -> float:
        """
        Computes the Gaussian noise scale σ given the allocated budget (Eq. 13).
        σ = sqrt(α * Δ² / (2 * ε_q))
        """
        # epsilon_q must be > 0
        eps = max(epsilon_q, 1e-12)
        return math.sqrt(alpha * (delta_sensitivity ** 2) / (2.0 * eps))
        
    def plot_allocation_curve(self, save_path: Optional[str] = None):
        """
        Plots the ε(τ) allocation curve.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Cannot plot.")
            return
            
        taus = np.linspace(0, 1, 100)
        epsilons = [self.allocate(t) for t in taus]
        
        plt.figure(figsize=(8, 5))
        plt.plot(taus, epsilons, 'b-', linewidth=2)
        plt.axvline(x=self.tau_0, color='r', linestyle='--', label=f'Threshold τ_0={self.tau_0}')
        plt.axhline(y=self.epsilon_0, color='g', linestyle=':', label=f'Strict ε_0={self.epsilon_0}')
        plt.axhline(y=self.epsilon_max, color='orange', linestyle=':', label=f'Relaxed ε_max={self.epsilon_max}')
        
        plt.xlabel("Query Diversity (τ)")
        plt.ylabel("Per-Query Budget ε(τ)")
        plt.title("DP-ADAGE Budget Allocation Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
if __name__ == '__main__':
    allocator = BudgetAllocator()
    print("tau=0.0 -> eps=", allocator.allocate(0.0))
    print("tau=0.2 -> eps=", allocator.allocate(0.2))
    print("tau=1.0 -> eps=", allocator.allocate(1.0))
