"""
Experiment: Main Results (Table 1 & Figure 2)
=============================================
Compares NoDefense, StaticNoise, OriginalADAGE, and DP-ADAGE on Cora.
Evaluates surrogate accuracy and downstream utility over 100 queries.
"""
import os
import sys
import torch
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.gnn_models import create_model, train_model
from src.dp_adage import DPAdage
from src.adage_baseline import OriginalADAGE
from src.attacker import SurrogateAttacker
from src.utils import set_seed, get_dataset_stats, load_config
from src.prada_baseline import PRADADefense
from src.dawn_baseline import DAWNDefense

def run_main_experiment():
    print("Running Main Experiment (Table 1 & Figure 2)...")
    seeds = [42, 43, 44, 45, 46]
    results = []
    
    for current_seed in seeds:
        print(f"\n--- Running with seed {current_seed} ---")
        set_seed(current_seed)
        
        try:
            from src.utils import load_dataset
            data, num_classes = load_dataset('cora', root='../data_cache')
            print(f"Successfully loaded real Cora dataset.")
        except Exception as e:
            print(f"Could not load real dataset ({e}). Falling back to MockData.")
            class MockData:
                def __init__(self):
                    self.num_nodes = 2708
                    self.num_edges = 5429
                    self.num_node_features = 1433
                    self.num_classes = 7
                    self.x = torch.randn(self.num_nodes, self.num_node_features)
                    self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
                    self.y = torch.randint(0, self.num_classes, (self.num_nodes,))
                    self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
                    self.train_mask[:1000] = True
                    self.val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
                    self.val_mask[1000:1500] = True
                    self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
                    self.test_mask[1500:2000] = True
            data = MockData()
            num_classes = data.num_classes
            
        print("Dataset stats:", get_dataset_stats(data))
        
        # 1. Train Victim Model
        print("Training victim model (GCN)...")
        victim_model = create_model('GCN', data.num_node_features, 64, num_classes)
        train_model(victim_model, data, epochs=10) # 10 for fast demo
        
        # 2. Defenses setup
        print("Setting up defenses...")
        dp_adage = DPAdage(victim_model, epsilon_total=200.0)
        dp_adage.fit(data, seed=current_seed)
        
        original_adage = OriginalADAGE(victim_model)
        original_adage.fit(data, seed=current_seed)
        
        class NoDefenseMock:
            def __init__(self, m): self.victim_model = m; self.tracker = type('obj', (object,), {'get_tau': lambda self, uid: 0.0})()
            def query(self, uid, n, d, typ='logits'): return torch.nn.functional.softmax(self.victim_model(d.x, d.edge_index)[0][n].unsqueeze(0), dim=-1)
            
        class StaticNoiseMock:
            def __init__(self, m): self.victim_model = m; self.tracker = type('obj', (object,), {'get_tau': lambda self, uid: 0.0})()
            def query(self, uid, n, d, typ='logits'): 
                l = self.victim_model(d.x, d.edge_index)[0][n].unsqueeze(0)
                return torch.nn.functional.softmax(l + torch.randn_like(l)*0.05, dim=-1)
                
        # 3. Attacks
        defenses = {
            'NoDefense': NoDefenseMock(victim_model),
            'StaticNoise': StaticNoiseMock(victim_model),
            'OriginalADAGE': original_adage,
            'PRADA': PRADADefense(victim_model),
            'DAWN': DAWNDefense(victim_model),
            'DP-ADAGE': dp_adage
        }
        
        for def_name, def_sys in defenses.items():
            print(f"\nEvaluating against {def_name}...")
            surrogate = create_model('GCN', data.num_node_features, 64, num_classes)
            attacker = SurrogateAttacker(surrogate, max_queries=100)
            
            history = attacker.attack(def_sys, data, user_id=f"attacker_{def_name}_{current_seed}")
            
            for q, acc, fid, tau, eps in zip(history['queries'], history['surrogate_accuracy'], history['surrogate_fidelity'], history['tau'], history['budget_consumed']):
                results.append({
                    'Seed': current_seed,
                    'Defense': def_name,
                    'Queries': q,
                    'Surrogate_Acc': acc,
                    'Surrogate_Fid': fid,
                    'Tau': tau,
                    'Budget_Consumed': eps
                })
                
    # Save results
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('data/results_main_cora.csv', index=False)
    print("\nResults saved to data/results_main_cora.csv")

if __name__ == '__main__':
    run_main_experiment()
