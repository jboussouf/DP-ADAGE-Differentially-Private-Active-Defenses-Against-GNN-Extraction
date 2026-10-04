import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import shapiro

class _TrackerStub:
    def get_tau(self, uid: str) -> float:
        return 0.0

class PRADADefense:
    """
    PRADA method to detect model extraction attacks based on MD_PRADA.md.
    It analyzes the distribution of distances between consecutive queries.
    If it detects an attack, it raises an alarm (here implemented by returning
    random probabilities to stop the attacker from learning anything useful).
    """
    def __init__(self, victim_model: torch.nn.Module, delta: float = 0.05, device: str = 'cpu'):
        self.victim_model = victim_model
        self.delta = delta
        self.device = device
        self.tracker = _TrackerStub()

        # PRADA states per user
        self.D = {} 
        self.G = {} # dict of class -> list of features
        self.T = {} # dict of class -> threshold
        self.D_Gc = {} # dict of class -> list of min distances
        self.attack_detected = {}

    def query(self, user_id: str, node_idx: int, data, output_type: str = 'logits') -> torch.Tensor:
        if user_id not in self.attack_detected:
            self.attack_detected[user_id] = False
            self.D[user_id] = []
            self.G[user_id] = {}
            self.T[user_id] = {}
            self.D_Gc[user_id] = {}

        self.victim_model.eval()
        with torch.no_grad():
            x = data.x[node_idx].to(self.device)
            logits, _ = self.victim_model(data.x.to(self.device), data.edge_index.to(self.device))
            node_logits = logits[node_idx].unsqueeze(0)
            probs = F.softmax(node_logits, dim=-1)
            c = probs.argmax().item()

        # If attack already detected, return random uniform probabilities
        if self.attack_detected[user_id]:
            return F.softmax(torch.randn_like(node_logits), dim=-1).cpu()

        G_c = self.G[user_id].setdefault(c, [])
        D_Gc_c = self.D_Gc[user_id].setdefault(c, [])
        T_c = self.T[user_id].get(c, 0.0)

        if len(G_c) == 0:
            G_c.append(x.cpu())
            D_Gc_c.append(0.0)
            self.T[user_id][c] = 0.0
        else:
            G_c_tensor = torch.stack(G_c).to(self.device)
            distances = torch.norm(G_c_tensor - x.unsqueeze(0), dim=1)
            d_min = distances.min().item()
            self.D[user_id].append(d_min)

            if d_min > T_c:
                G_c.append(x.cpu())
                D_Gc_c.append(d_min)

                if len(D_Gc_c) > 1:
                    mean_val = np.mean(D_Gc_c)
                    std_val = np.std(D_Gc_c)
                    self.T[user_id][c] = max(self.T[user_id][c], mean_val - std_val)

        # Check for attack if |D| > 100
        if len(self.D[user_id]) > 100:
            D_vals = np.array(self.D[user_id])
            mean_D = np.mean(D_vals)
            std_D = np.std(D_vals)
            # Filter D to within 3 std
            D_prime = D_vals[(D_vals >= mean_D - 3*std_D) & (D_vals <= mean_D + 3*std_D)]

            if len(D_prime) >= 3:
                stat, p_val = shapiro(D_prime)
                if p_val < self.delta:  # Using p_val for Shapiro-Wilk test
                    self.attack_detected[user_id] = True
                    print(f"[PRADA] Attack detected for user {user_id}! (p_val={p_val:.4f})")
                    return F.softmax(torch.randn_like(node_logits), dim=-1).cpu()

        return probs.cpu()

    def fit(self, data):
        pass
