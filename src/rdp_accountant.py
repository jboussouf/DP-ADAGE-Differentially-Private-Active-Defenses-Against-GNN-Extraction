"""
RDPAccountant
=============
Component 3 of the DP-ADAGE pipeline.

Responsible for tracking cumulative privacy expenditure and enforcing the hard cap.

Equations & Theorems:
  - Budget Check and Clamping (Eq. 9, 10): ε_q = min(ε(τ), ε_total - ε_cum,u)
  - Noise Floor (Eq. 11): once cap is reached, queries forced onto strict budget ε_0
  - RDP Composition (Theorem 4.2): ε_cum = Σ ε_q
  - Hard Cap (Corollary 4.3): ε_cum ≤ ε_total
  - RDP-to-DP Conversion (Theorem 4.4): ε* = ε_total + log(1/δ)/(α-1)
"""
from typing import Dict, Tuple
import math

class RDPAccountant:
    def __init__(self, epsilon_total: float = 5.0, alpha: float = 8.0, delta: float = 1e-6):
        self.epsilon_total = epsilon_total
        self.alpha = alpha
        self.delta = delta
        self._cumulative: Dict[str, float] = {}
        
    def check_and_consume(self, user_id: str, requested_budget: float, epsilon_0: float) -> Tuple[float, bool]:
        """
        Enforces the budget cap and updates cumulative spending (Eq. 9, 10).
        Returns (approved_budget, is_noise_floor).
        """
        if user_id not in self._cumulative:
            self._cumulative[user_id] = 0.0
            
        current_cum = self._cumulative[user_id]
        
        # If already exhausted, apply noise floor
        if current_cum >= self.epsilon_total - 1e-9:
            return epsilon_0, True
            
        # Clamping
        approved_budget = min(requested_budget, self.epsilon_total - current_cum)
        
        # Update cumulative
        self._cumulative[user_id] += approved_budget
        
        # Is this the query that hits the cap?
        is_exhausted = self._cumulative[user_id] >= self.epsilon_total - 1e-9
        
        return approved_budget, is_exhausted
        
    def get_cumulative(self, user_id: str) -> float:
        return self._cumulative.get(user_id, 0.0)
        
    def is_exhausted(self, user_id: str) -> bool:
        return self.get_cumulative(user_id) >= self.epsilon_total - 1e-9
        
    def get_dp_certificate(self) -> Tuple[float, float]:
        """
        Converts (α, ε_total)-RDP to (ε*, δ)-DP (Theorem 4.4).
        ε* = ε_total + log(1/δ) / (α - 1)
        """
        eps_star = self.epsilon_total + math.log(1.0 / self.delta) / (self.alpha - 1.0)
        return eps_star, self.delta
        
    def reset_user(self, user_id: str) -> None:
        if user_id in self._cumulative:
            del self._cumulative[user_id]
            
    def get_all_users_state(self) -> dict:
        return self._cumulative.copy()
        
if __name__ == '__main__':
    accountant = RDPAccountant(epsilon_total=5.0)
    eps_star, delta = accountant.get_dp_certificate()
    print(f"Formal DP Certificate: ({eps_star:.2f}, {delta})-DP")
    
    eps1, floor1 = accountant.check_and_consume("userA", 2.0, 0.01)
    print(f"Query 1: approved={eps1}, floor={floor1}, cum={accountant.get_cumulative('userA')}")
    
    eps2, floor2 = accountant.check_and_consume("userA", 4.0, 0.01)
    print(f"Query 2: approved={eps2}, floor={floor2}, cum={accountant.get_cumulative('userA')}")
    
    eps3, floor3 = accountant.check_and_consume("userA", 2.0, 0.01)
    print(f"Query 3: approved={eps3}, floor={floor3}, cum={accountant.get_cumulative('userA')}")
