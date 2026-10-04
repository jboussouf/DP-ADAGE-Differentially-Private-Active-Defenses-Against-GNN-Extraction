"""
Plot Pareto Frontier (Figure 3)
===============================
Generates scatter plot showing the privacy-utility tradeoff (Pareto frontier).
Bubble size represents efficiency (latency).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_pareto():
    df = pd.read_csv('data/results_all_datasets.csv')
    df_200 = df[df['Queries'] == 200]
    agg = df_200.groupby('Defense').mean(numeric_only=True).reset_index()
    
    utility_vals = []
    security_vals = []
    sizes = []
    defense_names = []
    
    min_latency = agg['Latency_ms'].min()
    
    for _, row in agg.iterrows():
        defense = row['Defense']
        if defense == 'OriginalADAGE':
            defense = 'ADAGE'
            
        utility = row['Downstream_Acc']
        security = 1.0 - row['Surrogate_Acc']
        efficiency = min_latency / row['Latency_ms'] if row['Latency_ms'] > 0 else 1.0
        
        utility_vals.append(utility)
        security_vals.append(security)
        sizes.append(300 * efficiency)
        defense_names.append(defense)
        
    color_map = {
        'NoDefense': 'gray',
        'StaticNoise': 'blue',
        'ADAGE': 'orange',
        'MDP_Adapted': 'purple',
        'PRADA': 'purple',
        'DAWN': 'brown',
        'DP-ADAGE': 'red'
    }
    
    plt.figure(figsize=(9, 6))
    
    for i in range(len(defense_names)):
        d = defense_names[i]
        c = color_map.get(d, 'green')
        plt.scatter(utility_vals[i], security_vals[i], s=sizes[i], color=c, label=d, alpha=0.7, edgecolors='black')
        
    plt.xlabel('Utility (Benign Accuracy)')
    plt.ylabel('Security (1 - Surrogate Accuracy)')
    plt.title('The Inference Trilemma (Bubble Size = Efficiency)')
    
    # Plot Pareto frontier dynamically
    points = list(zip(utility_vals, security_vals))
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    pareto_front = []
    max_sec = -float('inf')
    
    for u, s in reversed(sorted_points):
        if s > max_sec:
            pareto_front.append((u, s))
            max_sec = s
            
    pareto_front.reverse()
    
    if pareto_front:
        px = [p[0] for p in pareto_front]
        py = [p[1] for p in pareto_front]
        plt.plot(px, py, 'k--', alpha=0.3, label='Pareto Frontier')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    
    os.makedirs('visuals/outputs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('visuals/outputs/fig3_pareto.pdf')
    print("Plot saved to visuals/outputs/fig3_pareto.pdf")

if __name__ == '__main__':
    plot_pareto()
