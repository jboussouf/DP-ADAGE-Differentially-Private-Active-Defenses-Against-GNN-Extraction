"""
SurrogateAttacker
=================
Implements model extraction attacks on GNNs via API querying.
Supports Random and Margin-based (Uncertainty) active sampling strategies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Any, Callable

class SurrogateAttacker:
    def __init__(self, surrogate_model: nn.Module, query_strategy: str = 'uncertainty', max_queries: int = 100, device: str = 'cpu'):
        self.surrogate = surrogate_model.to(device)
        self.query_strategy = query_strategy
        self.max_queries = max_queries
        self.device = device
        self._attacker_communities = None
        self._community_cycle = None
        self._current_comm_idx = 0
        self._queries_in_current_comm = 0
        
    def _init_budget_aware(self, data):
        if self._attacker_communities is None:
            import networkx as nx
            from torch_geometric.utils import to_networkx
            G = to_networkx(data, to_undirected=True)
            try:
                import community.community_louvain as community_louvain
                partition = community_louvain.best_partition(G, random_state=42)
            except ImportError:
                from networkx.algorithms.community import louvain_communities
                communities = louvain_communities(G, seed=42)
                partition = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        partition[node] = i
            
            # Map comm_id to list of nodes
            self._attacker_communities = {}
            for node, comm_id in partition.items():
                if comm_id not in self._attacker_communities:
                    self._attacker_communities[comm_id] = []
                self._attacker_communities[comm_id].append(node)
            self._community_cycle = list(self._attacker_communities.keys())
            self._current_comm_idx = 0
            self._queries_in_current_comm = 0
        
    def _margin_score(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Uncertainty sampling score: smaller margin between top-2 probs means higher uncertainty.
        Returns a score where HIGHER is MORE uncertain.
        """
        top2, _ = torch.topk(probs, 2, dim=-1)
        margin = top2[:, 0] - top2[:, 1]
        return -margin # Negative margin so higher is more uncertain
        
    def _select_query_nodes(self, data, already_queried: set, num_nodes: int, victim_model: nn.Module = None) -> List[int]:
        """
        Selects nodes to query based on the strategy.
        """
        available_nodes = [i for i in range(data.num_nodes) if i not in already_queried and not data.train_mask[i]]
        num_to_select = min(num_nodes, len(available_nodes))
        
        if num_to_select == 0:
            return []
            
        if self.query_strategy == 'random':
            return random.sample(available_nodes, num_to_select)
            
        elif self.query_strategy == 'uncertainty':
            # Need predictions from current surrogate to find uncertain nodes
            self.surrogate.eval()
            with torch.no_grad():
                logits, _ = self.surrogate(data.x.to(self.device), data.edge_index.to(self.device))
                probs = F.softmax(logits, dim=-1)
                
                # Get scores only for available nodes
                scores = self._margin_score(probs[available_nodes])
                
                # Top-k uncertain nodes
                _, top_indices = torch.topk(scores, num_to_select)
                return [available_nodes[idx.item()] for idx in top_indices]
                
        elif self.query_strategy == 'budget_aware':
            if getattr(self, '_attacker_communities', None) is None:
                self._init_budget_aware(data)
                
            self.surrogate.eval()
            with torch.no_grad():
                logits, _ = self.surrogate(data.x.to(self.device), data.edge_index.to(self.device))
                probs = F.softmax(logits, dim=-1)
                scores = self._margin_score(probs)
                
            selected = []
            while len(selected) < num_to_select:
                # Get current community
                comm_id = self._community_cycle[self._current_comm_idx]
                comm_nodes = self._attacker_communities[comm_id]
                
                # Filter available nodes in this community
                avail_in_comm = [n for n in comm_nodes if n in available_nodes and n not in selected]
                
                if avail_in_comm and self._queries_in_current_comm < 5: # switch every 5 queries
                    # Pick most uncertain in this community
                    comm_scores = scores[avail_in_comm]
                    best_idx = torch.argmax(comm_scores).item()
                    selected.append(avail_in_comm[best_idx])
                    self._queries_in_current_comm += 1
                else:
                    # Switch community
                    self._current_comm_idx = (self._current_comm_idx + 1) % len(self._community_cycle)
                    self._queries_in_current_comm = 0
                    
                    # Safety break if no nodes left anywhere
                    if sum(len([n for n in self._attacker_communities[cid] if n in available_nodes and n not in selected]) for cid in self._community_cycle) == 0:
                        break
                        
            # If we still need nodes, fallback to regular uncertainty
            if len(selected) < num_to_select:
                still_needed = num_to_select - len(selected)
                avail_others = [n for n in available_nodes if n not in selected]
                if avail_others:
                    other_scores = scores[avail_others]
                    _, top_indices = torch.topk(other_scores, min(still_needed, len(avail_others)))
                    selected.extend([avail_others[idx.item()] for idx in top_indices])
                    
            return selected
                
        else:
            raise ValueError(f"Unknown query strategy: {self.query_strategy}")

    def _train_surrogate(self, query_nodes: List[int], query_responses: torch.Tensor, data, epochs=50):
        """
        Trains surrogate model on collected responses (distillation).
        Uses KL Divergence or CrossEntropy for soft labels.
        """
        self.surrogate.train()
        optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=0.01)
        
        node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
        node_mask[query_nodes] = True
        
        targets = query_responses.to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits, _ = self.surrogate(data.x.to(self.device), data.edge_index.to(self.device))
            
            # FIX 1: Index logits directly with query_nodes list so the order exactly
            # matches the sequentially concatenated targets (query_responses).
            log_probs = F.log_softmax(logits[query_nodes], dim=-1)
            loss = F.kl_div(log_probs, targets, reduction='batchmean')
            
            loss.backward()
            optimizer.step()
            
            # Add a print statement on the last epoch of large query batches to verify learning
            if epoch == epochs - 1 and len(query_nodes) >= 100:
                print(f"    [Debug] Surrogate Training - Queries: {len(query_nodes)}, Loss: {loss.item():.4f}")
            
    def evaluate(self, data, victim_model) -> Dict[str, float]:
        """
        Computes surrogate accuracy (on true labels) and surrogate fidelity (match with victim).
        """
        self.surrogate.eval()
        victim_model.eval()
        with torch.no_grad():
            surr_logits, _ = self.surrogate(data.x.to(self.device), data.edge_index.to(self.device))
            vic_logits, _ = victim_model(data.x.to(self.device), data.edge_index.to(self.device))
            
            surr_preds = surr_logits.argmax(dim=-1)
            vic_preds = vic_logits.argmax(dim=-1)
            
            test_mask = data.test_mask
            
            # Accuracy: surrogate vs ground truth
            acc = (surr_preds[test_mask] == data.y.to(self.device)[test_mask]).float().mean().item()
            
            # Fidelity: surrogate vs victim model
            fid = (surr_preds[test_mask] == vic_preds[test_mask]).float().mean().item()
            
        return {'surrogate_accuracy': acc, 'surrogate_fidelity': fid}

    def attack(self, defense_sys, data, user_id: str = "attacker_1") -> Dict[str, list]:
        """
        Main attack loop. Queries victim API through the defense system.
        """
        queried_nodes = []
        collected_responses = []
        
        history = {
            'queries': [],
            'surrogate_accuracy': [],
            'surrogate_fidelity': [],
            'tau': [],
            'budget_consumed': []
        }
        
        batch_size = 10
        num_batches = self.max_queries // batch_size
        
        for b in range(num_batches):
            new_nodes = self._select_query_nodes(data, set(queried_nodes), batch_size)
            if not new_nodes:
                break
                
            batch_responses = []
            tau, eps_cum = 0.0, 0.0
            
            for node in new_nodes:
                # Issue query through defense API
                # DPAdage returns a dict, OriginalADAGE returns tensor directly
                if hasattr(defense_sys, 'get_dp_certificate'):
                    res = defense_sys.query(user_id, node, data)
                    resp_tensor = res['response']
                    tau = res['tau']
                    eps_cum = res['epsilon_cum']
                else:
                    resp_tensor = defense_sys.query(user_id, node, data, 'logits')
                    # Just mock tau and eps for baseline logging
                    tau = defense_sys.tracker.get_tau(user_id)
                    eps_cum = 0.0
                    
                batch_responses.append(resp_tensor.detach())
                
            queried_nodes.extend(new_nodes)
            collected_responses.append(torch.cat(batch_responses, dim=0))
            
            # Retrain surrogate
            all_responses = torch.cat(collected_responses, dim=0)
            self._train_surrogate(queried_nodes, all_responses, data)
            
            # Evaluate
            metrics = self.evaluate(data, defense_sys.victim_model)
            
            # Log
            history['queries'].append(len(queried_nodes))
            history['surrogate_accuracy'].append(metrics['surrogate_accuracy'])
            history['surrogate_fidelity'].append(metrics['surrogate_fidelity'])
            history['tau'].append(tau)
            history['budget_consumed'].append(eps_cum)
            
        return history

if __name__ == '__main__':
    print("Surrogate attacker structure ready.")
