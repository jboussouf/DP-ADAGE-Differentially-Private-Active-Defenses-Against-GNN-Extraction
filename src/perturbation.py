"""
GaussianPerturbationMechanism
=============================
Component 4 of the DP-ADAGE pipeline.

Responsible for clipping outputs and injecting calibrated Gaussian noise.

Equations:
  - Logit Perturbation (Eq. 14): z̃_v = clip(z_v, C) + η, η ~ N(0, σ_τ² I)
    Sensitivity: Δ_2 = 2C
  - Embedding Perturbation (Eq. 15): ẽ_v = clip(e_v, B) + η, η ~ N(0, σ_τ² I)
    Sensitivity: Δ_2 = 2B
  - ADAGE Label Flipping (Eq. 7): ρ = h_η(τ) = 1 / (1 + exp(η*(1-2τ)))
"""
import math
import torch
import torch.nn.functional as F

class GaussianPerturbationMechanism:
    def __init__(self, clip_norm_C: float = 1.0, clip_norm_B: float = 1.0, alpha: float = 8.0):
        self.clip_C = clip_norm_C
        self.clip_B = clip_norm_B
        self.alpha = alpha
        
    def compute_sigma(self, epsilon_q: float, sensitivity: float) -> float:
        """
        σ = sqrt(α * Δ² / (2 * ε_q))
        """
        eps = max(epsilon_q, 1e-12)
        return math.sqrt(self.alpha * (sensitivity ** 2) / (2.0 * eps))
        
    def clip_tensor(self, tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
        """
        Clips tensor to have maximum L2 norm of `max_norm`.
        """
        # Calculate L2 norm along the feature dimension
        norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
        # Scale down if norm > max_norm, else keep same
        scale = torch.clamp(max_norm / (norm + 1e-12), max=1.0)
        return tensor * scale
        
    def perturb_logits(self, logits: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Applies Eq. 14 for logits.
        """
        clipped = self.clip_tensor(logits, self.clip_C)
        noise = torch.randn_like(clipped) * sigma
        return clipped + noise
        
    def perturb_embeddings(self, embeddings: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Applies Eq. 15 for embeddings.
        """
        clipped = self.clip_tensor(embeddings, self.clip_B)
        noise = torch.randn_like(clipped) * sigma
        return clipped + noise
        
    def randomized_response(self, logits: torch.Tensor, epsilon_q: float) -> torch.Tensor:
        """
        Applies Randomized Response (RR) for discrete class outputs (Setup A).
        """
        batch_size, num_classes = logits.size()
        
        # Prevent math.exp overflow if epsilon_q is extremely large
        eps = min(epsilon_q, 50.0)
        rho = num_classes / (math.exp(eps) + num_classes - 1.0)
        
        y_true = torch.argmax(logits, dim=-1)
        perturbed_probs = torch.zeros_like(logits)
        
        for i in range(batch_size):
            if torch.rand(1).item() < (1.0 - rho):
                y_hat = y_true[i].item()
            else:
                y_hat = torch.randint(0, num_classes, (1,)).item()
            perturbed_probs[i, y_hat] = 1.0
            
        return perturbed_probs
        
    def flip_labels(self, probs: torch.Tensor, tau: float, eta: float = 10.0) -> torch.Tensor:
        """
        Original ADAGE label flipping (Eq. 7).
        ρ = 1 / (1 + exp(η*(1 - 2*τ)))
        """
        rho = 1.0 / (1.0 + math.exp(eta * (1.0 - 2.0 * tau)))
        
        batch_size, num_classes = probs.size()
        device = probs.device
        
        perturbed_probs = probs.clone()
        
        for i in range(batch_size):
            if torch.rand(1).item() < rho:
                # Need to swap max class with a random class
                max_idx = torch.argmax(probs[i]).item()
                rand_idx = torch.randint(0, num_classes, (1,)).item()
                
                # Swap probabilities
                temp = perturbed_probs[i, max_idx].item()
                perturbed_probs[i, max_idx] = perturbed_probs[i, rand_idx]
                perturbed_probs[i, rand_idx] = temp
                
        return perturbed_probs

if __name__ == '__main__':
    mech = GaussianPerturbationMechanism(clip_norm_C=1.0)
    logits = torch.tensor([[5.0, 1.0, 0.1]])
    
    sigma = mech.compute_sigma(epsilon_q=0.01, sensitivity=2.0)
    print(f"Sigma for eps=0.01: {sigma}")
    
    noisy_logits = mech.perturb_logits(logits, sigma)
    print("Original logits:", logits)
    print("Noisy logits:", noisy_logits)
    print("Noisy probs:", F.softmax(noisy_logits, dim=-1))
