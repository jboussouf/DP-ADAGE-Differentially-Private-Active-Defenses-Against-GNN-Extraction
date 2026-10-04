import torch
import torch.nn.functional as F
import hashlib

class _TrackerStub:
    def get_tau(self, uid: str) -> float:
        return 0.0

class DAWNDefense:
    """
    DAWN method (Dynamic Adversarial Watermarking of Neural Networks).
    Dynamically watermarks a small fraction (r_w) of queries from a client
    by changing the prediction responses for them.
    """
    def __init__(self, victim_model: torch.nn.Module, r_w: float = 0.05, device: str = 'cpu'):
        self.victim_model = victim_model
        self.r_w = r_w
        self.device = device
        self.tracker = _TrackerStub()
        self.secret_key = "DAWN_SECRET_KEY"
        self.medians = None

    def query(self, user_id: str, node_idx: int, data, output_type: str = 'logits') -> torch.Tensor:
        self.victim_model.eval()
        with torch.no_grad():
            logits, emb = self.victim_model(data.x.to(self.device), data.edge_index.to(self.device))
            node_logits = logits[node_idx].unsqueeze(0)
            
            # Smooth embedding by binarizing against the pre-calculated median
            node_emb = emb[node_idx]
            if self.medians is not None:
                binarized_emb = (node_emb >= self.medians).to(torch.int8)
                binarized_bytes = binarized_emb.cpu().numpy().tobytes()
            else:
                # Fallback if fit was not called (e.g. some tests)
                binarized_bytes = data.x[node_idx].cpu().numpy().tobytes()
            
            # Always return probabilities to match attacker.py expectations
            base_output = F.softmax(node_logits, dim=-1)

        # Hash secret + user_id + binarized input to deterministically decide if this query is watermarked
        hash_input = self.secret_key.encode('utf-8') + user_id.encode('utf-8') + binarized_bytes
        h = hashlib.sha256(hash_input).digest()
        
        # Use first 16 bytes for the watermark decision
        hash_val = int.from_bytes(h[:16], 'big')
        max_val = 2**128
        
        if hash_val < self.r_w * max_val:
            # Watermark this query: permute the outputs
            # We use the next 16 bytes as a seed for permutation
            seed = int.from_bytes(h[16:], 'big') % (2**32 - 1)
            rng = torch.Generator(device=base_output.device)
            rng.manual_seed(seed)
            
            # Permute outputs randomly to return an incorrect label
            perm_idx = torch.randperm(base_output.size(-1), generator=rng, device=base_output.device)
            watermarked_output = base_output[:, perm_idx]
            return watermarked_output.cpu()

        return base_output.cpu()

    def fit(self, data):
        self.victim_model.eval()
        with torch.no_grad():
            # Pass data through the victim model to get embeddings
            _, emb = self.victim_model(data.x.to(self.device), data.edge_index.to(self.device))
            # Only use training set for median calculation
            train_emb = emb[data.train_mask.to(self.device)]
            # Compute median along dimension 0 (the nodes)
            self.medians = train_emb.median(dim=0).values
