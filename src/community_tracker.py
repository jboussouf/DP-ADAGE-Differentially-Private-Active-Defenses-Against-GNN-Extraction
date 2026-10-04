"""
CommunityTracker
================
Component 1 of the DP-ADAGE pipeline.

Responsible for:
  1. Running Louvain community detection on the training graph G (offline, once)
  2. Computing per-community centroid embeddings ω_k = (1/|C_k|) Σ_{v∈C_k} e_v  (Eq. 4)
  3. For each incoming query embedding e, finding nearest community k* = argmin_k ||e - ω_k||_2 (Eq. 5)
  4. Maintaining per-user set O_u of occupied community indices
  5. Computing τ = |O_u| / K  (Eq. 6) — the query diversity score

Reference: Section 3.1 of DP-ADAGE paper; Section 3.3.1 of ADAGE paper.
"""
from typing import Dict, Set, Optional
import numpy as np
import torch
import networkx as nx

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

class CommunityTracker:
    def __init__(self, num_communities: int = 10, device: str = 'cpu'):
        self.num_communities = num_communities
        self.device = device
        self._user_communities: Dict[str, Set[int]] = {}
        self._centroids: Optional[np.ndarray] = None
        
    def fit(self, embeddings: torch.Tensor, edge_index: torch.Tensor, seed: int = 42) -> None:
        """
        Runs Louvain community detection on the graph and computes centroid embeddings.
        """
        num_nodes = embeddings.size(0)
        
        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
        
        # Run Louvain
        try:
            if community_louvain is not None and hasattr(community_louvain, 'best_partition'):
                partition = community_louvain.best_partition(G, random_state=seed)
            else:
                # Fallback to networkx's built-in Louvain algorithm
                from networkx.algorithms.community import louvain_communities
                communities = louvain_communities(G, seed=seed)
                partition = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        partition[node] = i
        except Exception as e:
            print(f"Warning: Louvain community detection failed ({e}). Using random assignment.")
            partition = {i: np.random.randint(0, self.num_communities) for i in range(num_nodes)}
            
        # Extract unique communities (may not equal num_communities exactly, but we use the actual count)
        unique_comms = set(partition.values())
        self.num_communities = len(unique_comms)
        
        # Compute centroids (Eq. 4)
        emb_np = embeddings.detach().cpu().numpy()
        self._centroids = np.zeros((self.num_communities, emb_np.shape[1]))
        
        for k in range(self.num_communities):
            nodes_in_k = [n for n, c in partition.items() if c == k]
            if len(nodes_in_k) > 0:
                self._centroids[k] = np.mean(emb_np[nodes_in_k], axis=0)
                
    def get_community(self, embedding: torch.Tensor) -> int:
        """
        Finds nearest community centroid (Eq. 5).
        """
        if self._centroids is None:
            raise ValueError("CommunityTracker must be fitted first.")
            
        emb_np = embedding.detach().cpu().numpy()
        distances = np.linalg.norm(self._centroids - emb_np, axis=1)
        return int(np.argmin(distances))
        
    def update_user(self, user_id: str, embedding: torch.Tensor) -> float:
        """
        Updates the set of occupied communities for a user and returns τ.
        """
        k_star = self.get_community(embedding)
        
        if user_id not in self._user_communities:
            self._user_communities[user_id] = set()
            
        self._user_communities[user_id].add(k_star)
        return self.get_tau(user_id)
        
    def get_tau(self, user_id: str) -> float:
        """
        Computes query diversity score τ = |O_u| / K (Eq. 6).
        """
        if user_id not in self._user_communities:
            return 0.0
        return len(self._user_communities[user_id]) / self.num_communities
        
    def reset_user(self, user_id: str) -> None:
        """
        Resets tracking state for a specific user.
        """
        if user_id in self._user_communities:
            del self._user_communities[user_id]
            
    def get_stats(self) -> dict:
        return {
            "num_communities": self.num_communities,
            "num_tracked_users": len(self._user_communities),
            "is_fitted": self._centroids is not None
        }

if __name__ == '__main__':
    # Simple demo
    tracker = CommunityTracker(num_communities=3)
    dummy_embeddings = torch.randn(100, 64)
    dummy_edges = torch.randint(0, 100, (2, 500))
    
    tracker.fit(dummy_embeddings, dummy_edges)
    print("Fitted tracker:", tracker.get_stats())
    
    query_emb = torch.randn(64)
    tau = tracker.update_user("user_1", query_emb)
    print(f"User 1 τ after 1 query: {tau}")
