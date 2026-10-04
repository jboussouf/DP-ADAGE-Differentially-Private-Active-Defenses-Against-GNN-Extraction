"""
Plot Main Results (Figure 2)
============================
Generates line plot of surrogate accuracy over queries, 
and budget consumption trajectory for DP-ADAGE.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_main():
    if not os.path.exists('data/results_main_cora.csv'):
        print("Data not found. Run experiments/exp_main_results.py first.")
        return
        
    df = pd.read_csv('data/results_main_cora.csv')
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Accuracy lines
    colors = {'NoDefense': 'black', 'StaticNoise': 'blue', 'ADAGE': 'green', 'OriginalADAGE': 'green', 'PRADA': 'purple', 'DAWN': 'brown', 'DP-ADAGE': 'red'}
    for def_name in df['Defense'].unique():
        sub = df[df['Defense'] == def_name]
        ax1.plot(sub['Queries'], sub['Surrogate_Acc'], label=def_name, color=colors[def_name], marker='o')
        
    ax1.set_xlabel('Number of Queries')
    ax1.set_ylabel('Surrogate Accuracy', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim([0, 1.0])
    
    # Budget consumed for DP-ADAGE on secondary axis
    ax2 = ax1.twinx()
    dp_sub = df[df['Defense'] == 'DP-ADAGE']
    ax2.plot(dp_sub['Queries'], dp_sub['Budget_Consumed'], label='Budget Consumed (DP-ADAGE)', color='purple', linestyle='--')
    ax2.set_ylabel('Cumulative RDP Budget (ε_cum)', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim([0, 6.0])
    
    # Hard cap line
    ax2.axhline(y=5.0, color='purple', linestyle=':', label='Hard Cap (ε_total=5.0)')
    
    fig.suptitle('Figure 2: Surrogate Accuracy and Cumulative Budget (Cora)')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85))
    
    os.makedirs('visuals/outputs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('visuals/outputs/fig2_main_results.pdf')
    print("Plot saved to visuals/outputs/fig2_main_results.pdf")

if __name__ == '__main__':
    plot_main()
