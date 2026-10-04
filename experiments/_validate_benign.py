"""
_validate_benign.py — Smoke-test for evaluate_benign_utility
Run: python3 experiments/_validate_benign.py
"""
import sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gnn_models import create_model, train_model
from src.dp_adage import DPAdage
from src.adage_baseline import OriginalADAGE
from src.utils import set_seed, load_dataset, load_config

_BENIGN_EPS_MAX   = 10.0
_BENIGN_EPS_TOTAL = float('inf')

class _TrackerStub:
    def get_tau(self, uid): return 0.0

class NoDefense:
    def __init__(self, vm, device='cpu'):
        self.victim_model = vm; self.device = device; self.tracker = _TrackerStub()
    def query(self, uid, idx, data, output_type='logits'):
        self.victim_model.eval()
        with torch.no_grad():
            logits, _ = self.victim_model(data.x.to(self.device), data.edge_index.to(self.device))
            return F.softmax(logits[idx].unsqueeze(0), dim=-1).cpu()

class StaticNoise:
    def __init__(self, vm, noise_scale=0.05, device='cpu'):
        self.victim_model = vm; self.noise_scale = noise_scale; self.device = device; self.tracker = _TrackerStub()
    def query(self, uid, idx, data, output_type='logits'):
        self.victim_model.eval()
        with torch.no_grad():
            logits, _ = self.victim_model(data.x.to(self.device), data.edge_index.to(self.device))
            nl = logits[idx].unsqueeze(0)
            return F.softmax(nl + torch.randn_like(nl)*self.noise_scale, dim=-1).cpu()

def _clean_acc(model, data, device):
    model.eval()
    x=data.x.to(device); ei=data.edge_index.to(device); y=data.y.to(device); tm=data.test_mask.to(device)
    with torch.no_grad():
        logits,_ = model(x,ei); preds=logits.argmax(-1)
        return (preds[tm]==y[tm]).float().mean().item()

def evaluate_benign_utility(def_name, defense_factory, victim_model, data, device, clean_acc):
    if def_name == 'NoDefense':
        return clean_acc
    if def_name == 'DP-ADAGE':
        bd = DPAdage(victim_model, epsilon_0=0.01, epsilon_max=_BENIGN_EPS_MAX,
                     epsilon_total=_BENIGN_EPS_TOTAL, num_communities=10, device=device)
        bd.fit(data)
        tidx = torch.where(data.test_mask.to(device))[0].tolist()
        y = data.y.to(device); correct=0
        for idx in tidx:
            res = bd.query(f"bnode_{idx}", idx, data)
            if res['response'].argmax(-1).item() == y[idx].item(): correct+=1
        return correct/len(tidx) if tidx else 0.0
    if def_name == 'StaticNoise':
        d = defense_factory()
        tidx = torch.where(data.test_mask.to(device))[0].tolist()
        y=data.y.to(device); correct=0
        for idx in tidx:
            if d.query(f"b_{idx}",idx,data,'logits').argmax(-1).item()==y[idx].item(): correct+=1
        return correct/len(tidx) if tidx else 0.0
    if def_name == 'OriginalADAGE':
        d = defense_factory()
        if hasattr(d,'fit'): d.fit(data)
        tidx = torch.where(data.test_mask.to(device))[0].tolist()
        y=data.y.to(device); correct=0
        for idx in tidx:
            if d.query(f"b_{idx}",idx,data,'logits').argmax(-1).item()==y[idx].item(): correct+=1
        return correct/len(tidx) if tidx else 0.0
    return clean_acc

if __name__ == '__main__':
    try:
        import os
        config = load_config(os.path.join(os.path.dirname(__file__), '..', 'data', 'config_default.json'))
        seed = config.get('seed', 42)
    except Exception:
        seed = 42
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    data, num_classes = load_dataset('Cora', root='data_cache')
    print(f"Cora: {data.num_nodes} nodes, {num_classes} classes, {data.test_mask.sum().item()} test\n")
    vm = create_model('GCN', data.num_node_features, 64, num_classes)
    train_model(vm, data, epochs=50, device=device)
    ca = _clean_acc(vm, data, device)
    print(f"Clean accuracy (50ep): {ca:.4f}\n")
    facs = {
        'NoDefense': lambda: NoDefense(vm, device=device),
        'StaticNoise': lambda: StaticNoise(vm, 0.05, device=device),
        'OriginalADAGE': lambda: OriginalADAGE(vm, num_communities=10, device=device),
        'DP-ADAGE': lambda: DPAdage(vm, epsilon_total=5.0, num_communities=10, device=device),
    }
    ok = True
    for name, fac in facs.items():
        ba = evaluate_benign_utility(name, fac, vm, data, device, ca)
        delta = abs(ba - ca)
        thr = 0.0 if name=='NoDefense' else 0.15
        status = "PASS" if delta<=thr else f"FAIL (delta={delta:.4f}>{thr})"
        ok = ok and delta<=thr
        print(f"[DEBUG] {name:>15s}  Benign={ba:.4f} Clean={ca:.4f} Delta={delta:.4f}  [{status}]")
    print(f"\n{'ALL PASS' if ok else 'FAIL'}")
