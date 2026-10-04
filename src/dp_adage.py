"""
DPAdage
=======
Full DP-ADAGE Pipeline

Orchestrates CommunityTracker, BudgetAllocator, RDPAccountant, and PerturbationMechanism.
Implements Algorithm 1 from the DP-ADAGE paper.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Any

from .community_tracker import CommunityTracker
from .budget_allocator import BudgetAllocator
from .rdp_accountant import RDPAccountant
from .perturbation import GaussianPerturbationMechanism

class DPAdage:
    def __init__(self, 
                 victim_model: torch.nn.Module, 
                 epsilon_0: float = 0.01, 
                 epsilon_max: float = 10.0, 
                 epsilon_total: float = 200.0, 
                 k: float = 20.0, 
                 tau_0: float = 0.30, 
                 tau_dead: float = 0.05,
                 alpha: float = 8.0, 
                 delta: float = 1e-6, 
                 clip_C: float = 1.0, 
                 clip_B: float = 1.0, 
                 output_type: str = 'logits',
                 num_communities: int = 10,
                 device: str = 'cpu'):
        
        self.victim_model = victim_model
        self.output_type = output_type
        self.device = device
        
        self.tracker = CommunityTracker(num_communities=num_communities, device=device)
        self.allocator = BudgetAllocator(epsilon_0=epsilon_0, epsilon_max=epsilon_max, k=k, tau_0=tau_0, tau_dead=tau_dead)
        self.accountant = RDPAccountant(epsilon_total=epsilon_total, alpha=alpha, delta=delta)
        self.mechanism = GaussianPerturbationMechanism(clip_norm_C=clip_C, clip_norm_B=clip_B, alpha=alpha)
        
        # Sensitivities
        self.delta_logits = 2.0 * clip_C
        self.delta_embeddings = 2.0 * clip_B
        self.sensitivity = self.delta_logits if output_type == 'logits' else self.delta_embeddings

    def fit(self, data, seed: int = 42) -> None:
        """
        Offline (once): Line 2-3 of Alg 1.
        Computes base embeddings on training graph and finds communities.
        """
        self.victim_model.eval()
        with torch.no_grad():
            _, embeddings = self.victim_model(data.x.to(self.device), data.edge_index.to(self.device))
            self.tracker.fit(embeddings, data.edge_index, seed=seed)
            
    def query(self, user_id: str, node_idx: int, data) -> Dict[str, Any]:
        """
        Per-query interaction (Lines 4-17 of Alg 1).
        """
        self.victim_model.eval()
        with torch.no_grad():
            logits, embeddings = self.victim_model(data.x.to(self.device), data.edge_index.to(self.device))
            
            # Line 5: e = f_θ^(L-1)(v)
            e_v = embeddings[node_idx].unsqueeze(0)
            
            # Lines 6-8: Community assignment and τ update
            tau = self.tracker.update_user(user_id, e_v)
            
            # Line 9: ε(τ)
            eps_tau = self.allocator.allocate(tau)
            
            # Lines 10-11: Budget enforcement
            eps_q, is_noise_floor = self.accountant.check_and_consume(user_id, eps_tau, self.allocator.epsilon_0)
            
            # Line 12: Noise scale
            sigma_tau = self.mechanism.compute_sigma(eps_q, self.sensitivity)
            
            # Lines 13-16: Perturbation
            if self.output_type == 'logits':
                z_v = logits[node_idx].unsqueeze(0)
                response = self.mechanism.randomized_response(z_v, eps_q)
            else: # embeddings
                noisy_emb = self.mechanism.perturb_embeddings(e_v, sigma_tau)
                response = noisy_emb
                
            return {
                'response': response.cpu(),
                'tau': tau,
                'epsilon_q': eps_q,
                'epsilon_cum': self.accountant.get_cumulative(user_id),
                'sigma': sigma_tau,
                'is_noise_floor': is_noise_floor
            }

    def get_dp_certificate(self) -> Dict[str, float]:
        eps, delta = self.accountant.get_dp_certificate()
        return {'epsilon': eps, 'delta': delta}
        
    def reset_user(self, user_id: str) -> None:
        self.tracker.reset_user(user_id)
        self.accountant.reset_user(user_id)

if __name__ == '__main__':
    # Simple mock model and data for demo
    class MockModel(torch.nn.Module):
        def forward(self, x, edge_index):
            return x, x
            
    class MockData:
        def __init__(self):
            self.x = torch.randn(10, 16)
            self.edge_index = torch.randint(0, 10, (2, 20))
            
    model = MockModel()
    data = MockData()
    
    defense = DPAdage(model, output_type='logits')
    defense.fit(data)
    print("DP Certificate:", defense.get_dp_certificate())
    
    res = defense.query("userX", 0, data)
    print(f"Query 1 -> tau: {res['tau']:.3f}, eps_q: {res['epsilon_q']:.3f}, sigma: {res['sigma']:.3f}")
