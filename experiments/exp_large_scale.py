"""
Experiment: Large-Scale Graph Evaluation
========================================
Evaluates DP-ADAGE and NoDefense on the Flickr dataset (~89k nodes)
to measure Louvain preprocessing time, per-query latency, and 
defense effectiveness on massive graphs.
"""
import os
import sys
import time
import torch
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.gnn_models import create_model, train_model
from src.dp_adage import DPAdage
from src.attacker import SurrogateAttacker
from src.utils import set_seed, load_dataset
from experiments.run_all_datasets import NoDefense

def run_large_scale_experiment():
    print("Running Large-Scale Graph Evaluation (Flickr - 89k nodes)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("[INFO] Using GPU (CUDA) for evaluation.")
    else:
        print("[WARNING] CUDA is not available! Falling back to CPU.")
    
    try:
        print("Loading Flickr dataset... (This may take a moment to download)")
        data, num_classes = load_dataset('flickr', root='../data_cache')
        print(f"Loaded Flickr: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_node_features} features, {num_classes} classes.")
    except Exception as e:
        print(f"Could not load Flickr dataset ({e}). Make sure torch_geometric is fully installed with relevant dependencies.")
        return
        
    print("\nTraining Victim Model (GCN)...")
    data = data.to(device)
    victim_model = create_model('GCN', data.num_node_features, 64, num_classes)
    victim_model = victim_model.to(device)
    train_model(victim_model, data, epochs=50, device=device)
    
    results = []
    
    # Evaluate Defenses
    defenses = {
        'NoDefense': NoDefense(victim_model, device=device),
        'DP-ADAGE': DPAdage(victim_model, epsilon_total=200.0, num_communities=50, device=device) # more communities for larger graph
    }
    
    for def_name, defense_sys in defenses.items():
        print(f"\n--- Evaluating {def_name} ---")
        
        # Measure fit / preprocessing time
        t0 = time.perf_counter()
        if hasattr(defense_sys, 'fit'):
            print(f"[{def_name}] Running preprocessing (e.g., Louvain)...")
            defense_sys.fit(data, seed=42)
        fit_time = time.perf_counter() - t0
        print(f"[{def_name}] Preprocessing took: {fit_time:.2f} seconds")
        
        # Run Attacker to measure defense effectiveness and per-query latency
        print(f"[{def_name}] Running Attacker (100 queries)...")
        surrogate = create_model('GCN', data.num_node_features, 64, num_classes)
        attacker = SurrogateAttacker(surrogate, query_strategy='uncertainty', max_queries=100, device=device)
        
        t1 = time.perf_counter()
        history = attacker.attack(defense_sys, data, user_id=f"atk_{def_name}")
        attack_time = time.perf_counter() - t1
        
        avg_latency_ms = (attack_time / 100) * 1000
        print(f"[{def_name}] Total attack time: {attack_time:.2f} s | Avg Latency: {avg_latency_ms:.2f} ms/query")
        
        for q, acc, eps in zip(history['queries'], history['surrogate_accuracy'], history['budget_consumed']):
            results.append({
                'Dataset': 'Flickr',
                'Defense': def_name,
                'Queries': q,
                'Surrogate_Acc': acc,
                'Budget_Consumed': eps,
                'Fit_Time_s': fit_time,
                'Avg_Latency_ms': avg_latency_ms
            })
            
    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'results_large_scale.csv'), index=False)
    print("\nResults saved to data/results_large_scale.csv")

if __name__ == '__main__':
    run_large_scale_experiment()
