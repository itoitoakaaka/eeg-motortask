import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.polynomial.polynomial import Polynomial
import os

# Settings from official script
excel_path = '/Users/itoakane/Research/SEP_processed/measurement2.xlsx'
output_dir = '/Users/itoakane/Research/SEP_processed/Figures_0217'
os.makedirs(output_dir, exist_ok=True)

L_COLOR_SCATTER = 'darkorange'

# Load data
df = pd.read_excel(excel_path)
df['sid'] = df['file_id'].astype(str).str[:5]
df_ana = df[(df['sid'].str.contains('id0')) & (~df['sid'].str.contains('id001')) & (~df['sid'].str.contains('id014'))].copy()

# Focus on Land Pre
df_land = df_ana[(df_ana['condition'] == 'Land') & (df_ana['phase'] == 'Pre')].copy()
df_land['is_std'] = df_land['file_id'].astype(str).str.endswith('0003')
df_land = df_land.sort_values(['sid', 'is_std'], ascending=[True, False]).drop_duplicates('sid')

# HYPOTHESIS MAPPING
mapping = {'id010': 'id015', 'id011': 'id010', 'id015': 'id011'}
auc_dict = df_land.set_index('sid')['AUC'].to_dict()

df_swapped = df_land.copy()
for n_sid, b_sid in mapping.items():
    if n_sid in df_swapped['sid'].values:
        idx = df_swapped[df_swapped['sid'] == n_sid].index[0]
        df_swapped.at[idx, 'AUC'] = auc_dict[b_sid]

def plot_official_style(ax, data, x_col, y_col, xlabel, ylabel, title):
    d = data.dropna(subset=[x_col, y_col])
    x = d[x_col].values
    y = d[y_col].values
    
    # Scatter
    ax.scatter(x, y, color=L_COLOR_SCATTER, s=120, label='Land (Swapped)', zorder=5, alpha=0.6, edgecolors='none')
    
    # Text labels for SIDs
    for i, sid in enumerate(d['sid']):
        ax.annotate(sid, (x[i], y[i]), xytext=(5,5), textcoords='offset points', fontsize=9, alpha=0.7)
    
    # Stats
    r, p = stats.pearsonr(x, y)
    def get_s(p_val): return '*' if p_val < 0.05 else ''
    st_text = f"Pearson: r={r:.3f}, p={p:.4f}{get_s(p)}"
    
    # Regression Line (Only if significant as per official rule, but here always to show hypothesis)
    if p < 0.05:
        poly = Polynomial.fit(x, y, 1)
        x_min_v, x_max_v = x.min(), x.max()
        xr = np.linspace(x_min_v - abs(x_max_v-x_min_v)*0.2, x_max_v + abs(x_max_v-x_min_v)*0.2, 100)
        ax.plot(xr, poly(xr), color=L_COLOR_SCATTER, ls='--', lw=2.0, alpha=0.8)

    # Decoration (Official Style)
    ax.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.98, st_text, transform=ax.transAxes, ha='right', va='top', fontsize=10.5, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))
    
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.text(0.5, -0.22, title, transform=ax.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')

    x_range = abs(x.max() - x.min())
    ax.set_xlim(x.min() - x_range*0.20, x.max() + x_range*0.20)
    y_range = abs(y.max() - y.min())
    ax.set_ylim(y.min() - y_range*0.20, y.max() + y_range*0.20)
    
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
    ax.set_box_aspect(1)

# Generate Plot
fig, ax = plt.subplots(figsize=(7, 7))
plot_official_style(ax, df_swapped, 'sp_pp_amp', 'AUC', 'spSEP Amp (Pre) [uV]', 'Behavioral AUC', 'Hypothesis: Land AUC vs spSEP (id10,11,15 Swapped)')
fig.tight_layout()
out_file = os.path.join(output_dir, 'Fig_Exp_Land_AUC_vs_spSEP_Swapped.png')
fig.savefig(out_file, dpi=300, bbox_inches='tight')
print(f"Generated hypothesis plot with official design: {out_file}")
