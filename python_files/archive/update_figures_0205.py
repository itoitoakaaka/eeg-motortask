import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.polynomial.polynomial import Polynomial
import os
import matplotlib as mpl

# ==========================================
# GraphPad Prism Style Configuration
# ==========================================
def set_prism_style():
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['xtick.major.size'] = 6
    mpl.rcParams['ytick.major.size'] = 6
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['grid.alpha'] = 0.0
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['pdf.fonttype'] = 42

set_prism_style()

# Settings
csv_path = 'mne/SEP_processed/measurements.csv'
output_dir = 'mne/SEP_processed/Figures_0205'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df_ana = df[~df['file_id'].str.contains('id001')].copy()
df_ana['sid'] = df_ana['file_id'].str[:5]

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
    
    ax.bar(x[0], means[0], color=L_COLOR, width=0.6, label='Land', alpha=0.9, zorder=2)
    ax.bar(x[1], means[1], color=W_COLOR, width=0.6, label='Water', alpha=0.9, zorder=2)
    ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black', capsize=0, linewidth=2, zorder=5)
    
    np.random.seed(42)
    # Using independent t-test as N differ
    p = stats.ttest_ind(data_l, data_w)[1]
    
    xj_l = np.random.normal(x[0], 0.06, size=len(data_l))
    xj_w = np.random.normal(x[1], 0.06, size=len(data_w))
    ax.scatter(xj_l, data_l, facecolors='white', edgecolors='black', linewidths=1.2, s=40, zorder=4, alpha=0.8)
    ax.scatter(xj_w, data_w, facecolors='white', edgecolors='black', linewidths=1.2, s=40, zorder=4, alpha=0.8)
    
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
    ax.set_box_aspect(1.2)

def plot_relation(ax, x_all, y_all, x_l, y_l, x_w, y_w, xlabel, ylabel, title):
    ax.scatter(x_l, y_l, color=L_COLOR, edgecolors='black', s=60, label='Land', zorder=5, alpha=0.9, linewidths=1.0)
    ax.scatter(x_w, y_w, color=W_COLOR, edgecolors='black', s=60, label='Water', zorder=5, alpha=0.9, linewidths=1.0)
    
    def add_reg(x, y, color, ls='-'):
        if len(x) > 1 and np.std(x) > 1e-9:
            p = Polynomial.fit(x, y, 1)
            xr = np.linspace(min(x_all)*0.95, max(x_all)*1.05, 100)
            ax.plot(xr, p(xr), color=color, ls=ls, lw=2.5, zorder=4)
            
    add_reg(x_l, y_l, 'orange', ls='--')
    add_reg(x_w, y_w, 'skyblue', ls='--')
    add_reg(x_all, y_all, 'black', ls='-')
    
    try:
        r_all, p_all = stats.pearsonr(x_all, y_all)
        r_l, p_l = stats.pearsonr(x_l, y_l)
        r_w, p_w = stats.pearsonr(x_w, y_w)
    except:
        r_all, p_all, r_l, p_l, r_w, p_w = 0, 1, 0, 1, 0, 1
    
    text = f"All: r={r_all:.3f}, p={p_all:.3f}\nLand: r={r_l:.3f}, p={p_l:.3f}\nWater: r={r_w:.3f}, p={p_w:.3f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, ha='left', va='top', fontsize=9, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, pad=15, fontsize=13, fontweight='bold')
    ax.set_box_aspect(1)

# --- Generate ---
sub_pre = df_ana[df_ana['phase']=='Pre']
sub_post = df_ana[df_ana['phase']=='Post']

print("Updating Figures_0205 (Independent T-Tests)...")

# Fig 1: AUC
fig1, ax1 = plt.subplots(figsize=(5, 6))
d_l = sub_pre[sub_pre['condition']=='Land']['AUC'].values
d_w = sub_pre[sub_pre['condition']=='Water']['AUC'].values
plot_bar_land_water(ax1, d_l, d_w, 'AUC Comparison', 'AUC')
fig1.savefig(os.path.join(output_dir, 'Fig1_AUC_Comparison.png'), dpi=300, bbox_inches='tight')

# Fig 2: Pre vs Post spSEP
fig2, ax2 = plt.subplots(figsize=(5, 6))
d_pre = sub_pre['sp_pp_amp'].values
d_post = sub_post['sp_pp_amp'].values
x = [0, 1]
means = [np.mean(d_pre), np.mean(d_post)]
sems = [stats.sem(d_pre), stats.sem(d_post)]
ax2.bar(x[0], means[0], color='gray', width=0.6, alpha=0.9)
ax2.bar(x[1], means[1], color='silver', width=0.6, alpha=0.9)
ax2.errorbar(x, means, yerr=sems, fmt='none', ecolor='black', capsize=0, linewidth=2)
# Here Pre/Post is same N for all subjects so we can draw lines
for i in range(len(d_pre)): ax2.plot(x, [d_pre[i], d_post[i]], color='gray', alpha=0.3, lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(['Pre', 'Post'], fontweight='bold')
ax2.set_ylabel('spSEP Amp (uV)', fontweight='bold')
ax2.set_title('Pre vs Post spSEP Amp', fontweight='bold')
fig2.savefig(os.path.join(output_dir, 'Fig2_PrePost_SEP_Comparison.png'), dpi=300, bbox_inches='tight')

# Other Figures
fig3, ax3 = plt.subplots(figsize=(6, 6))
xl = sub_pre[sub_pre['condition']=='Land']['sp_pp_amp'].values
yl = sub_pre[sub_pre['condition']=='Land']['pp30_ratio'].values
xw = sub_pre[sub_pre['condition']=='Water']['sp_pp_amp'].values
yw = sub_pre[sub_pre['condition']=='Water']['pp30_ratio'].values
plot_relation(ax3, np.concatenate([xl, xw]), np.concatenate([yl, yw]), xl, yl, xw, yw, 'spSEP Amp (Pre)', 'PPI 30ms Ratio (%)', 'Pre-spSEP vs PPI Ratio')
fig3.savefig(os.path.join(output_dir, 'Fig3_PrePPI_vs_spSEP.png'), dpi=300, bbox_inches='tight')

# Fig 4: AUC vs PreSEP
xl = sub_pre[sub_pre['condition']=='Land']['sp_pp_amp'].values
yl = sub_pre[sub_pre['condition']=='Land']['AUC'].values
xw = sub_pre[sub_pre['condition']=='Water']['sp_pp_amp'].values
yw = sub_pre[sub_pre['condition']=='Water']['AUC'].values
fig4, ax4 = plt.subplots(figsize=(6, 6))
plot_relation(ax4, np.concatenate([xl, xw]), np.concatenate([yl, yw]), xl, yl, xw, yw, 'spSEP Amp (Pre)', 'AUC', 'AUC vs Pre-spSEP')
fig4.savefig(os.path.join(output_dir, 'Fig4_AUC_vs_PreSEP.png'), dpi=300, bbox_inches='tight')

# Fig 5: AUC vs Change
xl = sub_post[sub_post['condition']=='Land']['sp_pp_amp_change'].values
yl = sub_pre[sub_pre['condition']=='Land']['AUC'].values
xw = sub_post[sub_post['condition']=='Water']['sp_pp_amp_change'].values
yw = sub_pre[sub_pre['condition']=='Water']['AUC'].values
fig5, ax5 = plt.subplots(figsize=(6, 6))
plot_relation(ax5, np.concatenate([xl, xw]), np.concatenate([yl, yw]), xl, yl, xw, yw, 'spSEP Amp Change (%)', 'AUC', 'AUC vs spSEP Change')
fig5.savefig(os.path.join(output_dir, 'Fig5_AUC_vs_Change.png'), dpi=300, bbox_inches='tight')

# Fig 6: 50MAE
fig6, ax6 = plt.subplots(figsize=(5, 6))
d_l = sub_pre[sub_pre['condition']=='Land']['MAE_50'].values
d_w = sub_pre[sub_pre['condition']=='Water']['MAE_50'].values
plot_bar_land_water(ax6, d_l, d_w, '50%MAE Comparison', '50%MAE (%)')
fig6.savefig(os.path.join(output_dir, 'Fig6_50MAE_Comparison.png'), dpi=300, bbox_inches='tight')

# Fig 7: Overall MAE
fig7, ax7 = plt.subplots(figsize=(5, 6))
d_l = sub_pre[sub_pre['condition']=='Land']['MAE_All'].values
d_w = sub_pre[sub_pre['condition']=='Water']['MAE_All'].values
plot_bar_land_water(ax7, d_l, d_w, 'Overall MAE Comparison', 'MAE (%)')
fig7.savefig(os.path.join(output_dir, 'Fig7_allMAE_Comparison.png'), dpi=300, bbox_inches='tight')

for fid, xc, yc, tl, xl, yl in [
    ('Fig8_50MAE_vs_PostSEP', 'sp_pp_amp', 'MAE_50', '50%MAE vs Post-spSEP', 'spSEP Amp (Post)', '50%MAE (%)'),
    ('Fig9_allMAE_vs_PostSEP', 'sp_pp_amp', 'MAE_All', 'Overall MAE vs Post-spSEP', 'spSEP Amp (Post)', 'MAE (%)'),
    ('Fig10_50MAE_vs_Change', 'sp_pp_amp_change', 'MAE_50', '50%MAE vs spSEP Change', 'spSEP Change (%)', '50%MAE (%)'),
    ('Fig11_allMAE_vs_Change', 'sp_pp_amp_change', 'MAE_All', 'Overall MAE vs spSEP Change', 'spSEP Change (%)', 'MAE (%)')
]:
    fig, ax = plt.subplots(figsize=(6, 6))
    if 'Post' in tl:
        xl_d = sub_post[sub_post['condition']=='Land'][xc].values
        yl_d = sub_post[sub_post['condition']=='Land'][yc].values
        xw_d = sub_post[sub_post['condition']=='Water'][xc].values
        yw_d = sub_post[sub_post['condition']=='Water'][yc].values
    else:
        xl_d = sub_post[sub_post['condition']=='Land'][xc].values
        yl_d = sub_pre[sub_pre['condition']=='Land'][yc].values
        xw_d = sub_post[sub_post['condition']=='Water'][xc].values
        yw_d = sub_pre[sub_pre['condition']=='Water'][yc].values
        
    plot_relation(ax, np.concatenate([xl_d, xw_d]), np.concatenate([yl_d, yw_d]), xl_d, yl_d, xw_d, yw_d, xl, yl, tl)
    fig.savefig(os.path.join(output_dir, f'{fid}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

print(f"Update of {output_dir} complete.")
