"""
OriginalADAGE
=============
Implements the heuristic active defense from "ADAGE: Active Defenses Against GNN Extraction"
(Xu et al., ASIACCS'26).

Uses community tracking to compute τ, but relies on heuristic label flipping
and embedding noise scaling without formal privacy guarantees.
"""
import math
import torch
import torch.nn.functional as F

from .community_tracker import CommunityTracker
from .perturbation import GaussianPerturbationMechanism

class OriginalADAGE:
    def __init__(self, 
                 victim_model: torch.nn.Module, 
                 eta: float = 10.0, 
                 lambda_: float = 1e-6, 
                 alpha: float = 1.0, 
                 beta: float = 0.9, 
                 num_communities: int = 10, 
                 device: str = 'cpu'):
                 
        self.victim_model = victim_model
        self.eta = eta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        self.tracker = CommunityTracker(num_communities=num_communities, device=device)
        self.mechanism = GaussianPerturbationMechanism() # Used for flip_labels helper
        
    def fit(self, data, seed: int = 42) -> None:
        self.victim_model.eval()
        with torch.no_grad():
            _, embeddings = self.victim_model(data.x.to(self.device), data.edge_index.to(self.device))
            self.tracker.fit(embeddings, data.edge_index, seed=seed)
            
    def _add_embedding_noise(self, embedding: torch.Tensor, tau: float) -> torch.Tensor:
        """
        σ = λ * (exp(ln(α)/λ * τ * β^(-1)) - 1)  (Eq. 8 of ADAGE)
        """
        if tau == 0:
            sigma = 0.0
        else:
            exponent = (math.log(self.alpha + 1e-9) / self.lambda_) * tau * (1.0 / self.beta)
            # prevent overflow
            exponent = min(exponent, 50.0) 
            sigma = self.lambda_ * (math.exp(exponent) - 1.0)
            
        noise = torch.randn_like(embedding) * sigma
        return embedding + noise
        
    def query(self, user_id: str, node_idx: int, data, output_type: str = 'logits') -> torch.Tensor:
        self.victim_model.eval()
        with torch.no_grad():
            logits, embeddings = self.victim_model(data.x.to(self.device), data.edge_index.to(self.device))
            e_v = embeddings[node_idx].unsqueeze(0)
            tau = self.tracker.update_user(user_id, e_v)
            
            if output_type == 'logits':
                probs = F.softmax(logits[node_idx].unsqueeze(0), dim=-1)
                return self.mechanism.flip_labels(probs, tau, self.eta).cpu()
            else:
                return self._add_embedding_noise(e_v, tau).cpu()
                
    def reset_user(self, user_id: str):
        self.tracker.reset_user(user_id)

if __name__ == '__main__':
    class MockModel(torch.nn.Module):
        def forward(self, x, edge_index):
            return x, x
            
    class MockData:
        def __init__(self):
            self.x = torch.randn(10, 16)
            self.edge_index = torch.randint(0, 10, (2, 20))
            
    model = MockModel()
    data = MockData()
    
    adage = OriginalADAGE(model)
    adage.fit(data)
    
    res = adage.query("user1", 0, data, 'logits')
    print("OriginalADAGE response:", res.shape)
