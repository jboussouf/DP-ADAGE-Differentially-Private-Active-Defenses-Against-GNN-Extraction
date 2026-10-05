"""
Experiment: Memory & Throughput Profiling
=========================================
Profiles DP-ADAGE under high concurrent load (1000 users) to measure:
1. Memory footprint of the CommunityTracker / RDPAccountant per user.
2. Throughput (Queries Per Second).
"""
import os
import sys
import time
import tracemalloc
import numpy as np
import torch
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.gnn_models import create_model
from src.dp_adage import DPAdage

def run_profiling_experiment():
    print("Running Memory & Throughput Profiling (1000 concurrent users)...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print("[INFO] Using GPU (CUDA) for evaluation.")
    else:
        print("[WARNING] CUDA is not available! Falling back to CPU.")
    
    # Create a mock large graph (100k nodes) for pure API benchmarking
    print("Generating 100k node mock graph...")
    class MockData:
        def __init__(self):
            self.num_nodes = 100000
            self.num_edges = 500000
            self.num_node_features = 128
            self.num_classes = 10
            self.x = torch.randn(self.num_nodes, self.num_node_features)
            self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
            self.y = torch.randint(0, self.num_classes, (self.num_nodes,))
            self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            
    data = MockData()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.train_mask = data.train_mask.to(device)
    data.test_mask = data.test_mask.to(device)
    
    victim_model = create_model('GCN', data.num_node_features, 64, data.num_classes)
    victim_model.to(device)
    
    print("Initializing DP-ADAGE...")
    defense_sys = DPAdage(victim_model, epsilon_total=200.0, num_communities=100, device=device)
    
    # To bypass Louvain for pure API profiling, mock the community centroids (100 communities, 64-dim embeddings)
    print("Bypassing actual Louvain to focus purely on query tracking overhead...")
    defense_sys.tracker._centroids = np.random.randn(defense_sys.tracker.num_communities, 64).astype(np.float32)
    
    # 1. Memory Profiling
    print("\n--- Memory Profiling ---")
    print("Simulating 1000 users making 1 query each to initialize tracking state...")
    
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    num_users = 1000
    for i in range(num_users):
        user_id = f"user_{i}"
        # Node index to query
        node_idx = i % data.num_nodes
        # Query
        _ = defense_sys.query(user_id, node_idx, data)
        
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    # Calculate approximate memory per user
    total_mem_diff = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
    mem_per_user_bytes = total_mem_diff / num_users
    
    print(f"Total memory increase for {num_users} users: {total_mem_diff / 1024:.2f} KB")
    print(f"Estimated memory overhead per user: {mem_per_user_bytes:.2f} bytes")
    
    # 2. Throughput Profiling (QPS)
    print("\n--- Throughput Profiling (QPS) ---")
    print("Running 10,000 continuous queries across all 1000 users...")
    
    num_queries = 10000
    t0 = time.perf_counter()
    
    for i in range(num_queries):
        user_id = f"user_{i % num_users}"
        node_idx = (i * 13) % data.num_nodes # pseudo-random node
        _ = defense_sys.query(user_id, node_idx, data)
        
    t1 = time.perf_counter()
    total_time = t1 - t0
    qps = num_queries / total_time
    
    print(f"Executed {num_queries} queries in {total_time:.2f} seconds.")
    print(f"Throughput: {qps:.2f} Queries Per Second (QPS)")
    
    # Save results
    results = [{
        'Metric': 'Memory Overhead Per User (Bytes)',
        'Value': mem_per_user_bytes
    }, {
        'Metric': 'Throughput (QPS)',
        'Value': qps
    }]
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'results_profiling.csv'), index=False)
    print("\nResults saved to data/results_profiling.csv")

if __name__ == '__main__':
    run_profiling_experiment()
