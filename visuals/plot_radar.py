"""
Plot Radar Chart (Figure 7)
===========================
Generates multidimensional comparison radar chart for defenses.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_radar():
    df = pd.read_csv('data/results_all_datasets.csv')
    df_200 = df[df['Queries'] == 200]
    agg = df_200.groupby('Defense').mean(numeric_only=True).reset_index()
    
    data = {}
    defenses = []
    
    max_downstream = agg['Downstream_Acc'].max()
    min_latency = agg['Latency_ms'].min()
    
    for _, row in agg.iterrows():
        defense = row['Defense']
        if defense == 'OriginalADAGE':
            defense = 'ADAGE'
            
        security = 1.0 - row['Attacker_Downstream_Acc']
        fidelity = 1.0 - row['Surrogate_Fid']
        utility = row['Downstream_Acc'] / max_downstream if max_downstream > 0 else 1.0
        efficiency = min_latency / row['Latency_ms'] if row['Latency_ms'] > 0 else 1.0
        
        data[defense] = [security, utility, fidelity, efficiency]
        defenses.append(defense)
    
    metrics = ['Security', 'Utility', 'Fidelity', 'Efficiency']
    N = len(metrics)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    color_map = {
        'NoDefense': 'gray',
        'StaticNoise': 'blue',
        'ADAGE': 'orange',
        'MDP_Adapted': 'purple',
        'PRADA': 'purple',
        'DAWN': 'brown',
        'DP-ADAGE': 'red'
    }
    
    for defense in defenses:
        values = data[defense]
        values += values[:1]
        
        c = color_map.get(defense, 'green')
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=defense, color=c)
        if defense == 'DP-ADAGE':
            ax.fill(angles, values, color=c, alpha=0.25)
            
    plt.xticks(angles[:-1], metrics, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.1)
    
    plt.title('Multidimensional Defense Comparison', size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    os.makedirs('visuals/outputs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('visuals/outputs/fig7_radar.pdf')
    print("Plot saved to visuals/outputs/fig7_radar.pdf")

if __name__ == '__main__':
    plot_radar()
