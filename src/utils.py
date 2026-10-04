"""
Utility Functions
=================
Dataset loading, metrics, and helper functions for DP-ADAGE experiments.
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
import json
import math
import pandas as pd
from typing import Tuple, Dict

try:
    import torch_geometric
    from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor, Flickr
    import torch_geometric.transforms as T
except ImportError:
    pass

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_data_splits(data, train_ratio=0.8, val_ratio=0.1):
    num_nodes = data.num_nodes
    indices = np.random.permutation(num_nodes)
    
    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    return data

def load_dataset(name: str, root='./data_cache'):
    name = name.lower()
    transform = T.NormalizeFeatures()

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=name.capitalize(), transform=transform)
    elif name == 'cora_ml':
        # Direct key used by the ACM fallback path in run_all_datasets.py.
        # CitationFull does not create train/val/test masks, so create_data_splits
        # is always called below.
        dataset = CitationFull(root=root, name='Cora_ML', transform=transform)
    elif name in ['acm', 'dblp']:
        try:
            dataset = CitationFull(root=root, name=name.upper(), transform=transform)
        except Exception:
            if name == 'acm':
                print(
                    "Warning: ACM dataset not found/supported natively. "
                    "Falling back to Cora_ML (CitationFull)."
                )
                dataset = CitationFull(root=root, name='Cora_ML', transform=transform)
            else:
                raise
    elif name == 'amazon':
        dataset = Amazon(root=root, name='Photo', transform=transform)
    elif name == 'coauthor':
        dataset = Coauthor(root=root, name='Physics', transform=transform)
    elif name == 'flickr':
        dataset = Flickr(root=root, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    data = dataset[0]

    # Guard: create proper 80/10/10 splits when:
    #   (a) train_mask is missing entirely, OR
    #   (b) train_mask is all-True (CitationFull behaviour -- every node flagged
    #       as training, which would make surrogate evaluation trivially perfect).
    needs_splits = (
        not hasattr(data, 'train_mask')
        or data.train_mask is None
        or (hasattr(data.train_mask, 'all') and data.train_mask.all())
    )
    if needs_splits:
        data = create_data_splits(data)

    return data, dataset.num_classes

def get_dataset_stats(data) -> dict:
    return {
        "nodes": data.num_nodes,
        "edges": data.num_edges,
        "features": data.num_node_features
    }

def evaluate_downstream(model, data, mask='test', device='cpu') -> float:
    """
    Evaluates model node-classification accuracy on a data split.

    BUG FIX: the mask tensor is now explicitly moved to `device` before
    indexing.  Previously, `m = getattr(data, f'{mask}_mask')` could be a
    CPU tensor even when `device='cuda'`, because train_model() mutates
    `data` in-place via `data.to(device)` — but only the *first* call does
    so reliably.  Subsequent calls (e.g. from evaluate_downstream called
    after train_model returned) could see a CPU mask with CUDA preds,
    producing silent wrong-index semantics or RuntimeErrors.
    """
    model.eval()
    with torch.no_grad():
        x          = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y          = data.y.to(device)
        m          = getattr(data, f"{mask}_mask").to(device)  # ← FIX

        logits, _ = model(x, edge_index)
        preds     = logits.argmax(dim=-1)
        acc       = (preds[m] == y[m]).float().mean().item()

    return acc

def compute_kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Computes KL(P || Q) where P and Q are probabilities derived from logits."""
    p_probs = F.softmax(p_logits, dim=-1)
    q_log_probs = F.log_softmax(q_logits, dim=-1)
    return F.kl_div(q_log_probs, p_probs, reduction='batchmean').item()

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)
        
def save_results_csv(results_dict: dict, path: str):
    df = pd.DataFrame(results_dict)
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")

if __name__ == '__main__':
    print("Utils module ready.")
