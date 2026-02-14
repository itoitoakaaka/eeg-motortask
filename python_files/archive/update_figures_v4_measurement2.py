import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.polynomial.polynomial import Polynomial
import os
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Settings
csv_path = 'SEP_processed/measurement2.csv'
output_dir = 'SEP_processed/Figures_0212'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
# Include all subjects except id001
df_ana = df[df['file_id'].str.contains('id0') & ~df['file_id'].str.contains('id001')].copy()

# Specific exclusion as per user request
exclude_ids = [
    'id0020001', 'id0030001', 'id0030002', 'id0030003', 'id0030004',
    'id0050001', 'id0050002', 'id0050003', 'id0050004',
    'id0080001', 'id0080002', 'id0080003',
    'id0090004', 'id0110001', 'id0130002'
]
df_ana = df_ana[~df_ana['file_id'].isin(exclude_ids)].copy()

df_ana['sid'] = df_ana['file_id'].str[:5]

L_COLOR_BAR = '#FFB74D'
L_COLOR_SCATTER = 'darkorange'
W_COLOR = 'skyblue'

def get_star(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.1: return '†'
    return 'ns'

def get_star_size(p):
    if p >= 0.1: return 10
    return 14

def plot_bar_land_water(ax, data_l, data_w, title, ylabel):
    x = [0, 1]
    means = [np.mean(data_l), np.mean(data_w)]
    sems = [stats.sem(data_l), stats.sem(data_w)]
    
    ax.bar(x[0], means[0], color=L_COLOR_BAR, width=0.6, label='Land', alpha=1.0)
    ax.bar(x[1], means[1], color=W_COLOR, width=0.6, label='Water', alpha=1.0)
    ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black', capsize=8, linewidth=2, zorder=4)
    
    np.random.seed(42)
    # Using independent t-test due to potential unequal counts
    p = stats.ttest_ind(data_l, data_w)[1]
    y_max = max(max(data_l), max(data_w)) * 1.15
    ax.plot([x[0], x[0], x[1], x[1]], [y_max*0.95, y_max, y_max, y_max*0.95], color='black', lw=1.5)
    ax.text(0.5, y_max*1.02, get_star(p), ha='center', va='bottom', fontsize=get_star_size(p), fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Land', 'Water'], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(title, fontweight='bold', fontsize=13)
    ax.set_ylim(0, y_max*1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    for s in ['left', 'bottom']:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color('black')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.set_box_aspect(1)

def plot_relation(ax, df_all, x_col, y_col, x_l, y_l, x_w, y_w, xlabel, ylabel, title, method='rmcorr', common_ylim=None):
    ax.scatter(x_l, y_l, color=L_COLOR_SCATTER, s=120, label='Land', zorder=5, alpha=0.6, edgecolors='none')
    ax.scatter(x_w, y_w, color=W_COLOR, s=120, label='Water', zorder=5, alpha=0.6, edgecolors='none')
    
    # Paired subjects connection lines
    land_df = df_all[df_all['condition']=='Land'].set_index('sid')
    water_df = df_all[df_all['condition']=='Water'].set_index('sid')
    common_sids = land_df.index.intersection(water_df.index)
    for sid in common_sids:
        ax.plot([land_df.loc[sid, x_col], water_df.loc[sid, x_col]], 
                [land_df.loc[sid, y_col], water_df.loc[sid, y_col]], 
                color='#CCCCCC', alpha=0.4, lw=0.6, zorder=2)
        
    try:
        r_l, p_l = stats.pearsonr(x_l, y_l)
        r_w, p_w = stats.pearsonr(x_w, y_w)
    except Exception:
        r_l, p_l, r_w, p_w = 0, 1, 0, 1

    def add_reg(x, y, color, p_val):
        if len(x) > 1 and np.std(x) > 1e-9 and p_val < 0.05:
            p = Polynomial.fit(x, y, 1)
            x_min_v = min(np.min(x_l) if len(x_l)>0 else 0, np.min(x_w) if len(x_w)>0 else 0)
            x_max_v = max(np.max(x_l) if len(x_l)>0 else 0, np.max(x_w) if len(x_w)>0 else 0)
            xr = np.linspace(x_min_v - abs(x_max_v-x_min_v)*0.2, x_max_v + abs(x_max_v-x_min_v)*0.2, 100)
            ax.plot(xr, p(xr), color=color, ls='--', lw=2.0, alpha=0.8)
            
    add_reg(x_l, y_l, L_COLOR_SCATTER, p_l)
    add_reg(x_w, y_w, W_COLOR, p_w)
    
    try:
        df_clean = df_all[[x_col, y_col, 'sid']].dropna()
        if method == 'rmcorr' and len(df_clean) > 2:
            model_null = smf.ols(f"{y_col} ~ C(sid)", data=df_clean).fit()
            model_full = smf.ols(f"{y_col} ~ C(sid) + {x_col}", data=df_clean).fit()
            beta_x = model_full.params[x_col]
            r_all = np.sqrt((model_null.ssr - model_full.ssr) / model_null.ssr) * np.sign(beta_x)
            p_all = model_full.pvalues[x_col]
            all_label = "RmCorr"
        else:
            r_all, p_all = stats.pearsonr(df_clean[x_col], df_clean[y_col])
            all_label = "Pearson"
            
        def get_s(p): return '*' if p < 0.05 else ''
        st_l = f"{all_label}: r={r_all:.3f}, p={p_all:.3f}{get_s(p_all)}"
        st_land = f"Land: r={r_l:.3f}, p={p_l:.3f}{get_s(p_l)}"
        st_water = f"Water: r={r_w:.3f}, p={p_w:.3f}{get_s(p_w)}"
    except Exception as e:
        r_all, p_all, r_l, p_l, r_w, p_w = 0, 1, 0, 1, 0, 1
        st_l, st_land, st_water = "Error", "Error", "Error"
    
    ax.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=0.9, edgecolor='gray')
    text = f"{st_l}\n{st_land}\n{st_water}"
    ax.text(0.98, 0.98, text, transform=ax.transAxes, ha='right', va='top', fontsize=10.5, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))
    
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.text(0.5, -0.22, title, transform=ax.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')

    x_min_data, x_max_data = df_clean[x_col].min(), df_clean[x_col].max()
    x_range = abs(x_max_data - x_min_data) if abs(x_max_data - x_min_data) > 1e-9 else 1.0
    ax.set_xlim(x_min_data - x_range*0.20, x_max_data + x_range*0.20)
    
    if common_ylim:
        ax.set_ylim(common_ylim)
    else:
        y_min_data, y_max_data = df_clean[y_col].min(), df_clean[y_col].max()
        y_range = abs(y_max_data - y_min_data) if abs(y_max_data - y_min_data) > 1e-9 else 1.0
        ax.set_ylim(y_min_data - y_range*0.20, y_max_data + y_range*0.20)
    
    ax.grid(False)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.0)
    ax.set_box_aspect(1)

# --- Plot Execution ---

# Fig 1: AUC Comparison
fig1, ax1 = plt.subplots(figsize=(6, 6))
d_l = df_ana[(df_ana['phase']=='Pre') & (df_ana['condition']=='Land')]['AUC'].values
d_w = df_ana[(df_ana['phase']=='Pre') & (df_ana['condition']=='Water')]['AUC'].values
plot_bar_land_water(ax1, d_l, d_w, 'Fig 1: AUC Comparison', 'AUC')
fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, 'Fig1_AUC_Comparison.png'), dpi=300, bbox_inches='tight')
plt.close(fig1)

# Fig 2: Pre-Post SEP Comparison (Design Refined: 4 groups + Hatching)
fig2, axes2 = plt.subplots(1, 3, figsize=(22, 6))
target_vars = ['sp_pp_amp', 'pp30_ratio', 'pp100_ratio']
vlabels = ['spSEP Amp (uV)', 'PPI 30 (%)', 'PPI 100 (%)']

for i, (v, vlabel) in enumerate(zip(target_vars, vlabels)):
    ax = axes2[i]
    l_pre = df_ana[(df_ana['condition']=='Land') & (df_ana['phase']=='Pre')][v].values
    l_post = df_ana[(df_ana['condition']=='Land') & (df_ana['phase']=='Post')][v].values
    w_pre = df_ana[(df_ana['condition']=='Water') & (df_ana['phase']=='Pre')][v].values
    w_post = df_ana[(df_ana['condition']=='Water') & (df_ana['phase']=='Post')][v].values
    
    vals = [np.mean(l_pre), np.mean(l_post), np.mean(w_pre), np.mean(w_post)]
    errs = [stats.sem(l_pre), stats.sem(l_post), stats.sem(w_pre), stats.sem(w_post)]
    x = [0.8, 1.8, 3.2, 4.2]
    colors = [L_COLOR_BAR, L_COLOR_BAR, W_COLOR, W_COLOR]
    hatches = ['', '////', '', '////']
    labels = ['L Pre', 'L Post', 'W Pre', 'W Post']
    
    for j in range(4):
        ax.bar(x[j], vals[j], color=colors[j], width=0.8, alpha=0.9, hatch=hatches[j], edgecolor='white', linewidth=0.5)
        ax.errorbar(x[j], vals[j], yerr=errs[j], fmt='none', ecolor='black', lw=2, capsize=0)
    
    # Paired lines Land
    for k in range(len(l_pre)): ax.plot([x[0], x[1]], [l_pre[k], l_post[k]], color='#CCCCCC', alpha=0.3, lw=0.6)
    # Paired lines Water
    for k in range(len(w_pre)): ax.plot([x[2], x[3]], [w_pre[k], w_post[k]], color='#CCCCCC', alpha=0.3, lw=0.6)
    
    p_pre = stats.ttest_ind(l_pre, w_pre)[1]
    p_post = stats.ttest_ind(l_post, w_post)[1]
    y_max_data = max(np.max(l_pre) if len(l_pre)>0 else 0, 
                     np.max(l_post) if len(l_post)>0 else 0, 
                     np.max(w_pre) if len(w_pre)>0 else 0, 
                     np.max(w_post) if len(w_post)>0 else 0)
    
    # Tighten the padding
    h = y_max_data * 0.05
    bracket_y1 = y_max_data + h
    ax.plot([x[0], x[0], x[2], x[2]], [bracket_y1, bracket_y1+h*0.5, bracket_y1+h*0.5, bracket_y1], color='black', lw=1.2)
    ax.text((x[0]+x[2])/2, bracket_y1+h*0.7, f"p = {p_pre:.3f}", ha='center', fontweight='bold', fontsize=10)
    
    bracket_y2 = y_max_data + h * 3.5
    ax.plot([x[1], x[1], x[3], x[3]], [bracket_y2, bracket_y2+h*0.5, bracket_y2+h*0.5, bracket_y2], color='black', lw=1.2)
    ax.text((x[1]+x[3])/2, bracket_y2+h*0.7, f"p = {p_post:.3f}", ha='center', fontweight='bold', fontsize=10)
    
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(vlabel, fontweight='bold')
    # Use a tighter multiplier for the top limit
    ax.set_ylim(0, bracket_y2 + h*4)
    ax.grid(axis='y', ls='--', alpha=0.2, color='gray')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax.set_box_aspect(1)

fig2.tight_layout()
fig2.text(0.5, 0.02, 'Fig 2: Pre-Post SEP Comparison', ha='center', fontweight='bold', fontsize=14)
fig2.savefig(os.path.join(output_dir, 'Fig2_PrePost_SEP_Comparison.png'), dpi=300, bbox_inches='tight')
plt.close(fig2)

# Fig 3: Pre PPI vs spSEP (Design Refined: Bottom titles + ANOVA info)
sub_pre = df_ana[df_ana['phase']=='Pre']
sub_post = df_ana[df_ana['phase']=='Post']
# Recalculate ANOVA interaction for measurement2 if possible, or use constant from previous verified run
p_anova_inter = 0.0138 
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))
for i, v in enumerate(['pp30_ratio', 'pp100_ratio']):
    v_label = f'PPI {v.replace("pp", "").replace("_ratio", "")} (Pre)'
    sub_df = sub_pre[sub_pre['condition'].isin(['Land', 'Water'])].copy()
    xl = sub_pre[sub_pre['condition']=='Land'][v].values
    yl = sub_pre[sub_pre['condition']=='Land']['sp_pp_amp'].values
    xw = sub_pre[sub_pre['condition']=='Water'][v].values
    yw = sub_pre[sub_pre['condition']=='Water']['sp_pp_amp'].values
    plot_relation(axes3[i], sub_df, v, 'sp_pp_amp', xl, yl, xw, yw, v_label, 'spSEP Amp (Pre)', '', method='rmcorr')
fig3.tight_layout()
fig3.text(0.5, -0.05, f'Fig 3: Pre PPI vs spSEP\n(ANOVA Interaction p={p_anova_inter:.3f})', ha='center', va='top', fontweight='bold', fontsize=13)
fig3.savefig(os.path.join(output_dir, 'Fig3_PrePPI_vs_spSEP.png'), dpi=300, bbox_inches='tight')
plt.close(fig3)

# Figs 4-5, 8-11: Relation Plots (rmcorr, common titles)
fig_configs = [
    ('Fig4_AUC_vs_PreSEP', ['sp_pp_amp', 'pp30_ratio', 'pp100_ratio'], 'AUC', 'Fig 4: AUC vs Pre Metrics', ['spSEP Amp (Pre)', 'PPI 30ms (Pre)', 'PPI 100ms (Pre)'], 'AUC', sub_pre),
    ('Fig5_AUC_vs_Change', ['sp_pp_amp_change', 'pp30_ratio_change', 'pp100_ratio_change'], 'AUC', 'Fig 5: AUC vs Neurophys Change', ['spSEP Change (%)', 'PPI 30 Change (%)', 'PPI 100 Change (%)'], 'AUC', sub_post), # Post change vs Pre AUC
    ('Fig8_50MAE_vs_PostSEP', ['sp_pp_amp', 'pp30_ratio', 'pp100_ratio'], 'MAE_50', 'Fig 8: 50%MAE vs Post Metrics', ['spSEP Amp (Post)', 'PPI 30ms (Post)', 'PPI 100ms (Post)'], '50%MAE (%)', sub_post),
    ('Fig9_allMAE_vs_PostSEP', ['sp_pp_amp', 'pp30_ratio', 'pp100_ratio'], 'MAE_All', 'Fig 9: Overall MAE vs Post Metrics', ['spSEP Amp (Post)', 'PPI 30ms (Post)', 'PPI 100ms (Post)'], 'MAE (%)', sub_post),
    ('Fig10_50MAE_vs_Change', ['sp_pp_amp_change', 'pp30_ratio_change', 'pp100_ratio_change'], 'MAE_50', 'Fig 10: 50%MAE vs Neurophys Change', ['spSEP Change (%)', 'PPI 30 Change (%)', 'PPI 100 Change (%)'], '50%MAE (%)', sub_post),
    ('Fig11_allMAE_vs_Change', ['sp_pp_amp_change', 'pp30_ratio_change', 'pp100_ratio_change'], 'MAE_All', 'Fig 11: Overall MAE vs Neurophys Change', ['spSEP Change (%)', 'PPI 30 Change (%)', 'PPI 100 Change (%)'], 'MAE (%)', sub_post)
]

for fid, xc_list, yc, tl, xl_names, y_label, base_df in fig_configs:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sub_df_all = base_df[base_df['condition'].isin(['Land', 'Water'])].copy()
    y_min_all = df_ana[yc].min() * 0.8
    y_max_all = df_ana[yc].max() * 1.2
    for i, (vx, vlabel) in enumerate(zip(xc_list, xl_names)):
        # For Fig 5, 10, 11: Y is often from PRE (AUC/MAE) while X is Post change. Use SID to match.
        if vx.endswith('_change') or 'Post' in vlabel:
            df_x = sub_post[sub_post['condition'].isin(['Land', 'Water'])]
            # For Fig 5/10/11, if yc is AUC/MAE, it's typically pre-task property
            df_y = sub_pre[sub_pre['condition'].isin(['Land', 'Water'])] if yc in ['AUC', 'MAE_50', 'MAE_All'] else df_x
        else:
            df_x = df_y = base_df[base_df['condition'].isin(['Land', 'Water'])]
            
        xl_d = df_x[df_x['condition']=='Land'][vx].values
        yl_d = df_y[df_y['condition']=='Land'][yc].values
        xw_d = df_x[df_x['condition']=='Water'][vx].values
        yw_d = df_y[df_y['condition']=='Water'][yc].values
        plot_relation(axes[i], sub_df_all, vx, yc, xl_d, yl_d, xw_d, yw_d, vlabel, y_label, '', method='rmcorr', common_ylim=(y_min_all, y_max_all))
    fig.suptitle(tl, fontweight='bold', fontsize=14, y=-0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{fid}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# Fig 6 & 7: MAE Comparison Bars
for fid, col, title, label in [('Fig6_50MAE_Comparison', 'MAE_50', 'Fig 6: 50%MAE Comparison', '50%MAE (%)'),
                               ('Fig7_allMAE_Comparison', 'MAE_All', 'Fig 7: Overall MAE Comparison', 'MAE (%)')]:
    fig, ax = plt.subplots(figsize=(6, 6))
    d_l = sub_pre[sub_pre['condition']=='Land'][col].values
    d_w = sub_pre[sub_pre['condition']=='Water'][col].values
    plot_bar_land_water(ax, d_l, d_w, title, label)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{fid}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

print(f"Update of {output_dir} using measurement2.csv complete.")
