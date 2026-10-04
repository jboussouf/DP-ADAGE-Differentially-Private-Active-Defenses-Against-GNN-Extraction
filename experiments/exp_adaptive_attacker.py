"""
Experiment: Adaptive Attacker Evaluation (Varying Query Budget)
===============================================================
Evaluates all defenses against the 'uncertainty' and 'budget_aware' 
attacker strategies over 500 queries to demonstrate the long-term 
behavior and the effectiveness of the epsilon_total budget cap.
"""
import os
import sys
import torch
import pandas as pd
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.gnn_models import create_model, train_model
from src.dp_adage import DPAdage
from src.adage_baseline import OriginalADAGE
from src.attacker import SurrogateAttacker
from src.utils import set_seed, get_dataset_stats, load_config
from src.prada_baseline import PRADADefense
from src.dawn_baseline import DAWNDefense
from experiments.run_all_datasets import NoDefense, StaticNoise

def run_adaptive_experiment():
    print("Running Adaptive Attacker Experiment (500 queries)...")
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
            
        # 1. Train Victim Model
        print("Training victim model (GCN)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print("[INFO] Using GPU (CUDA) for evaluation.")
        else:
            print("[WARNING] CUDA is not available! Falling back to CPU.")
            
        if hasattr(data, 'x'):
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.y = data.y.to(device)
            if hasattr(data, 'train_mask'): data.train_mask = data.train_mask.to(device)
            if hasattr(data, 'val_mask'): data.val_mask = data.val_mask.to(device)
            if hasattr(data, 'test_mask'): data.test_mask = data.test_mask.to(device)
            
        victim_model = create_model('GCN', data.num_node_features, 64, num_classes)
        victim_model = victim_model.to(device)
        train_model(victim_model, data, epochs=10, device=device)
        
        # 2. Defenses setup
        print("Setting up defenses...")
        defense_factories = {
            'NoDefense':     lambda: NoDefense(victim_model, device=device),
            'StaticNoise':   lambda: StaticNoise(victim_model, noise_scale=0.05, device=device),
            'OriginalADAGE': lambda: OriginalADAGE(victim_model, num_communities=10, device=device),
            'PRADA':         lambda: PRADADefense(victim_model, device=device),
            'DAWN':          lambda: DAWNDefense(victim_model, device=device),
            'DP-ADAGE':      lambda: DPAdage(victim_model, epsilon_total=200.0, num_communities=10, device=device),
        }
        
        for def_name, def_factory in defense_factories.items():
            for strategy in ['uncertainty', 'budget_aware']:
                print(f"\nEvaluating against {def_name} with '{strategy}' strategy...")
                
                # Create a fresh defense instance
                def_sys = def_factory()
                if hasattr(def_sys, 'fit'):
                    try:
                        def_sys.fit(data, seed=current_seed)
                    except TypeError:
                        def_sys.fit(data)
                        
                # Create fresh surrogate
                surrogate = create_model('GCN', data.num_node_features, 64, num_classes)
                attacker = SurrogateAttacker(surrogate, query_strategy=strategy, max_queries=500, device=device)
                
                # Unique user ID for each run
                user_id = f"atk_{def_name}_{strategy}_{current_seed}"
                
                history = attacker.attack(def_sys, data, user_id=user_id)
                
                for q, acc, fid, tau, eps in zip(history['queries'], history['surrogate_accuracy'], history['surrogate_fidelity'], history['tau'], history['budget_consumed']):
                    results.append({
                        'Seed': current_seed,
                        'Defense': def_name,
                        'Strategy': strategy,
                        'Queries': q,
                        'Surrogate_Acc': acc,
                        'Surrogate_Fid': fid,
                        'Tau': tau,
                        'Budget_Consumed': eps
                    })
                    
    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'results_adaptive_attacker.csv'), index=False)
    print("\nResults saved to data/results_adaptive_attacker.csv")

if __name__ == '__main__':
    run_adaptive_experiment()
