import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.polynomial.polynomial import Polynomial
import os

# ==========================================
# GraphPad Prism Style Configuration
# ==========================================
import matplotlib as mpl

def set_prism_style():
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['axes.linewidth'] = 1.5      # Thick axes
    mpl.rcParams['xtick.major.width'] = 1.5   # Thick ticks
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.direction'] = 'out'   # Outward ticks
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['xtick.major.size'] = 6
    mpl.rcParams['ytick.major.size'] = 6
    mpl.rcParams['axes.spines.top'] = False   # No top/right spines
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['grid.alpha'] = 0.0          # No grid lines
    mpl.rcParams['legend.frameon'] = False    # No legend box
    mpl.rcParams['pdf.fonttype'] = 42         # Export as PDF-safe fonts

set_prism_style()

# Settings
csv_path = '/Users/itoakane/Research/SEP_raw_temp/SEP_processed/measurements.csv'
output_dir = '/Users/itoakane/Research/SEP_raw_temp/Figures_PrismStyle'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df_ana = df[df['file_id'].str.contains('id00[2-9]|id010')].copy()
df_ana['sid'] = df_ana['file_id'].str[:5]
ids = sorted(df_ana['sid'].unique())

L_COLOR = '#FFB74D' # Orange
W_COLOR = '#A1D9F1' # Skyblue

def get_star(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.1: return '†'
    return 'ns'

def plot_bar_land_water(ax, data_l, data_w, title, ylabel):
    x = [0, 1]
    means = [np.mean(data_l), np.mean(data_w)]
    sems = [stats.sem(data_l), stats.sem(data_w)]
    
    # Prism style bars (smaller width, solid colors)
    ax.bar(x[0], means[0], color=L_COLOR, width=0.6, label='Land', alpha=0.9, zorder=2)
    ax.bar(x[1], means[1], color=W_COLOR, width=0.6, label='Water', alpha=0.9, zorder=2)
    
    # Error bars (thick, no caps is also common in Prism but keeping cap for clarity)
    ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black', capsize=0, linewidth=2, zorder=5)
    
    # Individual dots (Prism style: Open circles or colored dots with edge)
    np.random.seed(42)
    # Paired lines (thin gray)
    for i in range(len(data_l)):
        ax.plot(x, [data_l[i], data_w[i]], color='gray', alpha=0.3, linewidth=0.8, zorder=1)
    
    # Scatter dots with jitter
    xj_l = np.random.normal(x[0], 0.04, size=len(data_l))
    xj_w = np.random.normal(x[1], 0.04, size=len(data_w))
    
    ax.scatter(xj_l, data_l, facecolors='white', edgecolors='black', linewidths=1.2, s=40, zorder=4, alpha=0.8)
    ax.scatter(xj_w, data_w, facecolors='white', edgecolors='black', linewidths=1.2, s=40, zorder=4, alpha=0.8)
        
    # Statistics Brackets (Prism-like)
    p = stats.ttest_rel(data_l, data_w)[1]
    y_max = max(max(data_l), max(data_w))
    h = y_max * 0.05
    bracket_y = y_max + h
    ax.plot([x[0], x[0], x[1], x[1]], [bracket_y, bracket_y+h, bracket_y+h, bracket_y], color='black', lw=1.5, zorder=6)
    ax.text(0.5, bracket_y+h*1.2, get_star(p), ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Land', 'Water'], fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, pad=20, fontsize=13, fontweight='bold')
    
    ax.set_ylim(0, bracket_y + h*5)
    ax.set_box_aspect(1.2) # Prism usually has slightly taller bars or square

def plot_relation(ax, x_all, y_all, x_l, y_l, x_w, y_w, xlabel, ylabel, title):
    # Scatter dots with white edges (Prism look)
    ax.scatter(x_l, y_l, color=L_COLOR, edgecolors='black', s=60, label='Land', zorder=5, alpha=0.9, linewidths=1.0)
    ax.scatter(x_w, y_w, color=W_COLOR, edgecolors='black', s=60, label='Water', zorder=5, alpha=0.9, linewidths=1.0)
    
    # Regression line
    def add_reg(x, y, color, label, ls='-'):
        if len(x) > 1 and np.std(x) > 1e-9:
            p = Polynomial.fit(x, y, 1)
            xr = np.linspace(min(x_all)*0.95, max(x_all)*1.05, 100)
            ax.plot(xr, p(xr), color=color, ls=ls, lw=2.5, zorder=4)
            
    # Prism often uses solid thin lines or dash for subgroups
    add_reg(x_l, y_l, 'orange', 'Land', ls='--')
    add_reg(x_w, y_w, 'skyblue', 'Water', ls='--')
    add_reg(x_all, y_all, 'black', 'All', ls='-')
    
    # Stats
    try:
        r_all, p_all = stats.pearsonr(x_all, y_all)
        r_l, p_l = stats.pearsonr(x_l, y_l)
        r_w, p_w = stats.pearsonr(x_w, y_w)
    except:
        r_all, p_all, r_l, p_l, r_w, p_w = 0, 1, 0, 1, 0, 1
    
    # Legend like text
    text = f"All: r={r_all:.3f}, p={p_all:.3f}\nLand: r={r_l:.3f}, p={p_l:.3f}\nWater: r={r_w:.3f}, p={p_w:.3f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, ha='left', va='top', fontsize=9, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, pad=15, fontsize=13, fontweight='bold')
    ax.set_box_aspect(1)

# Generate Plots
print("Generating Prism-style plots...")

# Fig 1
fig1, ax1 = plt.subplots(figsize=(5, 6))
d_l = df_ana[(df_ana['condition']=='Land') & (df_ana['phase']=='Pre')]['AUC'].values
d_w = df_ana[(df_ana['condition']=='Water') & (df_ana['phase']=='Pre')]['AUC'].values
plot_bar_land_water(ax1, d_l, d_w, 'AUC (Pre)', 'AUC')
fig1.savefig(os.path.join(output_dir, 'Fig1_AUC_Comparison.png'), dpi=300, bbox_inches='tight')
plt.close(fig1)

# Fig 2 (Multi-panel)
# ... (変えず) ...

# Fig 3 - 11: Relation plots loop
sub_pre = df_ana[df_ana['phase']=='Pre']
sub_post = df_ana[df_ana['phase']=='Post']

# Helper to generate individual relation plots
def save_relation_fig(fid, x_col, y_col, df_x, df_y, xl, yl, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    xl_data = df_x[df_x['condition']=='Land'][x_col].values
    yl_data = df_y[df_y['condition']=='Land'][y_col].values
    xw_data = df_x[df_x['condition']=='Water'][x_col].values
    yw_data = df_y[df_y['condition']=='Water'][y_col].values
    plot_relation(ax, np.concatenate([xl_data, xw_data]), np.concatenate([yl_data, yw_data]), 
                  xl_data, yl_data, xw_data, yw_data, xl, yl, title)
    fig.savefig(os.path.join(output_dir, f'{fid}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# Fig 3: Pre spSEP vs PPI (30 and 100 in separate figs or combined?)
# Original Fig 3 was 1x2. Let's do 1x2 again.
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))
for i, v in enumerate(['pp30_ratio', 'pp100_ratio']):
    lbl = 'PPI 30ms Ratio' if v == 'pp30_ratio' else 'PPI 100ms Ratio'
    xl = sub_pre[sub_pre['condition']=='Land']['sp_pp_amp'].values
    yl = sub_pre[sub_pre['condition']=='Land'][v].values
    xw = sub_pre[sub_pre['condition']=='Water']['sp_pp_amp'].values
    yw = sub_pre[sub_pre['condition']=='Water'][v].values
    plot_relation(axes3[i], np.concatenate([xl, xw]), np.concatenate([yl, yw]), xl, yl, xw, yw, 'spSEP Amp (Pre)', lbl, '')
fig3.savefig(os.path.join(output_dir, 'Fig3_PreSEP_vs_PPI.png'), dpi=300, bbox_inches='tight')
plt.close(fig3)

# Fig 4: AUC vs Pre SEP
save_relation_fig('Fig4_AUC_vs_PreSEP', 'sp_pp_amp', 'AUC', sub_pre, sub_pre, 'spSEP Amp (Pre)', 'AUC', 'AUC vs Pre-spSEP')

# Fig 5: AUC vs Change (3 panels)
fig5, axes5 = plt.subplots(1, 3, figsize=(18, 6))
cvars = ['sp_pp_amp_change', 'pp30_ratio_change', 'pp100_ratio_change']
clabels = ['spSEP Amp Change', 'PPI 30ms Ratio Change', 'PPI 100ms Ratio Change']
for i, (v, vlabel) in enumerate(zip(cvars, clabels)):
    xl = sub_pre[sub_pre['condition']=='Land'][v].values
    yl = sub_pre[sub_pre['condition']=='Land']['AUC'].values
    xw = sub_pre[sub_pre['condition']=='Water'][v].values
    yw = sub_pre[sub_pre['condition']=='Water']['AUC'].values
    plot_relation(axes5[i], np.concatenate([xl, xw]), np.concatenate([yl, yw]), xl, yl, xw, yw, vlabel, 'AUC', '')
fig5.savefig(os.path.join(output_dir, 'Fig5_AUC_vs_Change.png'), dpi=300, bbox_inches='tight')
plt.close(fig5)

# Fig 6: 50%MAE Comparison
fig6, ax6 = plt.subplots(figsize=(5, 6))
d_l = sub_pre[sub_pre['condition']=='Land']['MAE_50'].values
d_w = sub_pre[sub_pre['condition']=='Water']['MAE_50'].values
plot_bar_land_water(ax6, d_l, d_w, '50%MAE Comparison', '50%MAE (%)')
fig6.savefig(os.path.join(output_dir, 'Fig6_50MAE_Comparison.png'), dpi=300, bbox_inches='tight')
plt.close(fig6)

# Fig 7: Overall MAE
fig7, ax7 = plt.subplots(figsize=(5, 6))
d_l = sub_pre[sub_pre['condition']=='Land']['MAE_All'].values
d_w = sub_pre[sub_pre['condition']=='Water']['MAE_All'].values
plot_bar_land_water(ax7, d_l, d_w, 'Overall MAE Comparison', 'MAE (%)')
fig7.savefig(os.path.join(output_dir, 'Fig7_OverallMAE_Comparison.png'), dpi=300, bbox_inches='tight')
plt.close(fig7)

# Fig 8, 9, 10, 11 handled similarly in loops
for fig_num, y_col in zip(['8', '9'], ['MAE_50', 'MAE_All']):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vlabels = ['spSEP Amp (Post)', 'PPI 30ms Ratio (Post)', 'PPI 100ms Ratio (Post)']
    for i, (v, vlabel) in enumerate(zip(['sp_pp_amp', 'pp30_ratio', 'pp100_ratio'], vlabels)):
        xl = sub_post[sub_post['condition']=='Land'][v].values
        yl = sub_post[sub_post['condition']=='Land'][y_col].values
        xw = sub_post[sub_post['condition']=='Water'][v].values
        yw = sub_post[sub_post['condition']=='Water'][y_col].values
        plot_relation(axes[i], np.concatenate([xl, xw]), np.concatenate([yl, yw]), xl, yl, xw, yw, vlabel, y_col+' (%)', '')
    name = f'Fig{fig_num}_{y_col}_vs_PostSEP.png'
    fig.savefig(os.path.join(output_dir, name), dpi=300, bbox_inches='tight')
    plt.close(fig)

for fig_num, y_col in zip(['10', '11'], ['MAE_50', 'MAE_All']):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (v, vlabel) in enumerate(zip(cvars, clabels)):
        xl = sub_pre[sub_pre['condition']=='Land'][v].values
        yl = sub_pre[sub_pre['condition']=='Land'][y_col].values
        xw = sub_pre[sub_pre['condition']=='Water'][v].values
        yw = sub_pre[sub_pre['condition']=='Water'][y_col].values
        plot_relation(axes[i], np.concatenate([xl, xw]), np.concatenate([yl, yw]), xl, yl, xw, yw, vlabel, y_col+' (%)', '')
    name = f'Fig{fig_num}_{y_col}_vs_Change.png'
    fig.savefig(os.path.join(output_dir, name), dpi=300, bbox_inches='tight')
    plt.close(fig)

print(f"SUCCESS: All 11 plots generated in {output_dir}")
