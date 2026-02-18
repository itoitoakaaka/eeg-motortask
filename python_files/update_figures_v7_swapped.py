import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.polynomial.polynomial import Polynomial
import os
import statsmodels.formula.api as smf

# Settings
excel_path = '/Users/itoakane/Research/SEP_processed/measurement2_swapped.xlsx'
outputOUTPUT_DIR = '/Users/itoakane/Research/SEP_processed/Figures_0217'
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_excel(excel_path)

# Include all subjects except id001 and id014
df_ana = df[df['file_id'].astype(str).str.contains('id0') & 
            ~df['file_id'].astype(str).str.contains('id001') & 
            ~df['file_id'].astype(str).str.contains('id014')].copy()
df_ana['sid'] = df_ana['file_id'].astype(str).str[:5]

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
    data_l = data_l[~np.isnan(data_l)]
    data_w = data_w[~np.isnan(data_w)]
    
    x = [0, 1]
    means = [np.mean(data_l) if len(data_l)>0 else 0, np.mean(data_w) if len(data_w)>0 else 0]
    sems = [stats.sem(data_l) if len(data_l)>1 else 0, stats.sem(data_w) if len(data_w)>1 else 0]
    
    ax.bar(x[0], means[0], color=L_COLOR_BAR, width=0.6, label='Land', alpha=1.0)
    ax.bar(x[1], means[1], color=W_COLOR, width=0.6, label='Water', alpha=1.0)
    ax.errorbar(x, means, yerr=sems, fmt='none', ecolor='black', capsize=8, linewidth=2, zorder=4)
    
    p = stats.ttest_ind(data_l, data_w)[1] if len(data_l)>1 and len(data_w)>1 else 1.0
    y_max = max(max(data_l) if len(data_l)>0 else 0, max(data_w) if len(data_w)>0 else 0) * 1.15
    y_max = max(y_max, 1.0)
    ax.plot([x[0], x[0], x[1], x[1]], [y_max*0.95, y_max, y_max, y_max*0.95], color='black', lw=1.5)
    ax.text(0.5, y_max*1.02, get_star(p), ha='center', va='bottom', fontsize=get_star_size(p), fontweight='bold')
    
    ax.set_xticks(x); ax.set_xticklabels(['Land', 'Water'], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(title, fontweight='bold', fontsize=13)
    ax.set_ylim(0, y_max*1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    for s in ['left', 'bottom']:
        ax.spines[s].set_visible(True); ax.spines[s].set_color('black')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax.set_box_aspect(1)

def plot_relation(ax, df_all, x_col, y_col, x_l, y_l, x_w, y_w, xlabel, ylabel, title, method='rmcorr', common_ylim=None):
    mask_l = ~np.isnan(x_l) & ~np.isnan(y_l)
    mask_w = ~np.isnan(x_w) & ~np.isnan(y_w)
    x_l_c, y_l_c = x_l[mask_l], y_l[mask_l]
    x_w_c, y_w_c = x_w[mask_w], y_w[mask_w]

    ax.scatter(x_l_c, y_l_c, color=L_COLOR_SCATTER, s=120, label='Land', zorder=5, alpha=0.6)
    ax.scatter(x_w_c, y_w_c, color=W_COLOR, s=120, label='Water', zorder=5, alpha=0.6)
    
    land_df = df_all[df_all['condition']=='Land'].set_index('sid')
    water_df = df_all[df_all['condition']=='Water'].set_index('sid')
    common_sids = land_df.index.intersection(water_df.index)
    for sid in common_sids:
        try:
            xl_v, yl_v = land_df.loc[sid, x_col], land_df.loc[sid, y_col]
            xw_v, yw_v = water_df.loc[sid, x_col], water_df.loc[sid, y_col]
            if not np.isnan([xl_v, yl_v, xw_v, yw_v]).any():
                ax.plot([xl_v, xw_v], [yl_v, yw_v], color='#CCCCCC', alpha=0.4, lw=0.6, zorder=2)
        except Exception: pass
        
    try:
        r_l, p_l = stats.pearsonr(x_l_c, y_l_c) if len(x_l_c)>2 else (0, 1)
        r_w, p_w = stats.pearsonr(x_w_c, y_w_c) if len(x_w_c)>2 else (0, 1)
    except Exception: r_l, p_l, r_w, p_w = 0, 1, 0, 1

    def add_reg(xvec, yvec, color, p_val):
        if len(xvec) > 2 and np.std(xvec) > 1e-9 and p_val < 0.05:
            p = Polynomial.fit(xvec, yvec, 1)
            x_min_v = min(np.min(x_l_c) if len(x_l_c)>0 else 0, np.min(x_w_c) if len(x_w_c)>0 else 0)
            x_max_v = max(np.max(x_l_c) if len(x_l_c)>0 else 0, np.max(x_w_c) if len(x_w_c)>0 else 0)
            xr = np.linspace(x_min_v - abs(x_max_v-x_min_v)*0.2, x_max_v + abs(x_max_v-x_min_v)*0.2, 100)
            ax.plot(xr, p(xr), color=color, ls='--', lw=2.0, alpha=0.8)
            
    add_reg(x_l_c, y_l_c, L_COLOR_SCATTER, p_l)
    add_reg(x_w_c, y_w_c, W_COLOR, p_w)
    
    try:
        df_clean = df_all[[x_col, y_col, 'sid']].dropna()
        if method == 'rmcorr' and len(df_clean) > 2 and df_clean['sid'].nunique() > 1:
            model_null = smf.ols(f"{y_col} ~ C(sid)", data=df_clean).fit()
            model_full = smf.ols(f"{y_col} ~ C(sid) + {x_col}", data=df_clean).fit()
            beta_x = model_full.params[x_col]
            r_all = np.sqrt(max(0, (model_null.ssr - model_full.ssr) / model_null.ssr)) * np.sign(beta_x)
            p_all = model_full.pvalues[x_col]
            all_label = "RmCorr"
            if p_all < 0.05:
                intercepts = model_full.params.filter(like='C(sid)')
                avg_i = intercepts.mean() if not intercepts.empty else model_full.params['Intercept']
                xr = np.linspace(df_clean[x_col].min()*0.8, df_clean[x_col].max()*1.2, 100)
                ax.plot(xr, avg_i + beta_x * xr, color='gray', ls='--', lw=2.5, alpha=0.6, zorder=1)
        else:
            r_all, p_all = stats.pearsonr(df_clean[x_col], df_clean[y_col]) if len(df_clean)>2 else (0,1)
            all_label = "Pearson"
            if p_all < 0.05:
                p_ov = Polynomial.fit(df_clean[x_col], df_clean[y_col], 1)
                xr = np.linspace(df_clean[x_col].min()*0.8, df_clean[x_col].max()*1.2, 100)
                ax.plot(xr, p_ov(xr), color='gray', ls='--', lw=2.5, alpha=0.6, zorder=1)
        def get_s(p): return '*' if p < 0.05 else ''
        st_l = f"{all_label}: r={r_all:.3f}, p={p_all:.3f}{get_s(p_all)}"
        st_land = f"Land: r={r_l:.3f}, p={p_l:.3f}{get_s(p_l)}"
        st_water = f"Water: r={r_w:.3f}, p={p_w:.3f}{get_s(p_w)}"
    except Exception: st_l, st_land, st_water = "Error", "Error", "Error"
    
    ax.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.98, f"{st_l}\n{st_land}\n{st_water}", transform=ax.transAxes, ha='right', va='top', fontsize=10.5, 
            fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))
    ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel(ylabel, fontsize=13)
    ax.text(0.5, -0.22, title, transform=ax.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Scaling
    x_min, x_max = df_clean[x_col].min(), df_clean[x_col].max()
    x_rng = abs(x_max - x_min) if abs(x_max-x_min)>1e-9 else 1.0
    ax.set_xlim(x_min - x_rng*0.2, x_max + x_rng*0.2)
    if common_ylim: ax.set_ylim(common_ylim)
    else:
        y_min, y_max = df_clean[y_col].min(), df_clean[y_col].max()
        y_rng = abs(y_max - y_min) if abs(y_max-y_min)>1e-9 else 1.0
        ax.set_ylim(y_min - y_rng*0.2, y_max + y_rng*0.2)
    ax.set_box_aspect(1)
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(True); ax.spines[s].set_color('black')

# Execution
# Fig 1
fig1, ax1 = plt.subplots(figsize=(6,6))
sub_pre = df_ana[df_ana['phase']=='Pre']
d_l = sub_pre[sub_pre['condition']=='Land']['AUC'].values
d_w = sub_pre[sub_pre['condition']=='Water']['AUC'].values
plot_bar_land_water(ax1, d_l, d_w, 'Fig 1: AUC Comparison', 'AUC')
fig1.savefig(os.path.join(output_dir, 'Fig1_AUC_Comparison.png'), dpi=300, bbox_inches='tight')

# Fig 2
fig2, axes2 = plt.subplots(1, 3, figsize=(22, 6))
vars_2 = ['sp_pp_amp', 'ppi30', 'ppi100']
vlabels = ['spSEP Amp (uV)', 'PPI 30 (%)', 'PPI 100 (%)']
for i, (v, vlabel) in enumerate(zip(vars_2, vlabels)):
    ax = axes2[i]
    l_pre = df_ana[(df_ana['condition']=='Land') & (df_ana['phase']=='Pre')][v].values
    l_post = df_ana[(df_ana['condition']=='Land') & (df_ana['phase']=='Post')][v].values
    w_pre = df_ana[(df_ana['condition']=='Water') & (df_ana['phase']=='Pre')][v].values
    w_post = df_ana[(df_ana['condition']=='Water') & (df_ana['phase']=='Post')][v].values
    
    vals = [np.nanmean(l_pre), np.nanmean(l_post), np.nanmean(w_pre), np.nanmean(w_post)]
    errs = [stats.sem(l_pre, nan_policy='omit'), stats.sem(l_post, nan_policy='omit'), stats.sem(w_pre, nan_policy='omit'), stats.sem(w_post, nan_policy='omit')]
    x = [0.8, 1.8, 3.2, 4.2]
    colors = [L_COLOR_BAR, L_COLOR_BAR, W_COLOR, W_COLOR]
    hatches = ['', '////', '', '////']
    for j in range(4):
        ax.bar(x[j], vals[j], color=colors[j], width=0.8, alpha=0.9, hatch=hatches[j], edgecolor='white', linewidth=0.5)
        ax.errorbar(x[j], vals[j], yerr=errs[j], fmt='none', ecolor='black', lw=2, capsize=0)
    for cond, xc in [('Land', [0.8, 1.8]), ('Water', [3.2, 4.2])]:
        c_pre = df_ana[(df_ana['condition']==cond) & (df_ana['phase']=='Pre')].set_index('sid')[v]
        c_post = df_ana[(df_ana['condition']==cond) & (df_ana['phase']=='Post')].set_index('sid')[v]
        for sid in c_pre.index.intersection(c_post.index):
            if not np.isnan([c_pre[sid], c_post[sid]]).any(): ax.plot(xc, [c_pre[sid], c_post[sid]], color='#CCCCCC', alpha=0.3, lw=0.6)
    y_max = np.nanmax(np.concatenate([l_pre, l_post, w_pre, w_post]))
    h = y_max * 0.05
    ax.plot([x[0], x[0], x[2], x[2]], [y_max+h, y_max+h*1.5, y_max+h*1.5, y_max+h], color='black', lw=1.2)
    ax.text((x[0]+x[2])/2, y_max+h*1.7, f"p={stats.ttest_ind(l_pre, w_pre, nan_policy='omit')[1]:.3f}", ha='center', fontweight='bold')
    ax.plot([x[1], x[1], x[3], x[3]], [y_max+h*3.5, y_max+h*4, y_max+h*4, y_max+h*3.5], color='black', lw=1.2)
    ax.text((x[1]+x[3])/2, y_max+h*4.2, f"p={stats.ttest_ind(l_post, w_post, nan_policy='omit')[1]:.3f}", ha='center', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(['L Pre', 'L Post', 'W Pre', 'W Post'])
    ax.set_ylabel(vlabel, fontweight='bold'); ax.set_box_aspect(1)
fig2.savefig(os.path.join(output_dir, 'Fig2_PrePost_SEP_Comparison.png'), dpi=300, bbox_inches='tight')

# Relations
sub_post = df_ana[df_ana['phase']=='Post']
configs = [
    ('Fig3_PrePPI_vs_spSEP', ['ppi30', 'ppi100'], 'sp_pp_amp', 'Fig 3: Pre PPI vs spSEP', ['PPI 30 (Pre)', 'PPI 100 (Pre)'], 'spSEP Amp (Pre)', sub_pre),
    ('Fig4_AUC_vs_PreSEP', ['sp_pp_amp', 'ppi30', 'ppi100'], 'AUC', 'Fig 4: AUC vs Pre Metrics', ['spSEP Amp (Pre)', 'PPI 30 (Pre)', 'PPI 100 (Pre)'], 'AUC', sub_pre),
    ('Fig5_AUC_vs_Change', ['sp_pp_amp_change', 'ppi30_change', 'ppi100_change'], 'AUC', 'Fig 5: AUC vs Neurophys Change', ['spSEP Change (%)', 'PPI 30 Change (%)', 'PPI 100 Change (%)'], 'AUC', sub_post),
    ('Fig8_50MAE_vs_PostSEP', ['sp_pp_amp', 'ppi30', 'ppi100'], 'MAE_50', 'Fig 8: 50%MAE vs Post Metrics', ['spSEP Amp (Post)', 'PPI 30 (Post)', 'PPI 100 (Post)'], '50%MAE (%)', sub_post),
    ('Fig9_allMAE_vs_PostSEP', ['sp_pp_amp', 'ppi30', 'ppi100'], 'MAE_All', 'Fig 9: Overall MAE vs Post Metrics', ['spSEP Amp (Post)', 'PPI 30 (Post)', 'PPI 100 (Post)'], 'MAE (%)', sub_post),
    ('Fig10_50MAE_vs_Change', ['sp_pp_amp_change', 'ppi30_change', 'ppi100_change'], 'MAE_50', 'Fig 10: 50%MAE vs Neurophys Change', ['spSEP Change (%)', 'PPI 30 Change (%)', 'PPI 100 Change (%)'], '50%MAE (%)', sub_post),
    ('Fig11_allMAE_vs_Change', ['sp_pp_amp_change', 'ppi30_change', 'ppi100_change'], 'MAE_All', 'Fig 11: Overall MAE vs Neurophys Change', ['spSEP Change (%)', 'PPI 30 Change (%)', 'PPI 100 Change (%)'], 'MAE (%)', sub_post)
]
for fid, xc, yc, tl, xl, yl, base_df in configs:
    n_cols = len(xc)
    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 6))
    if n_cols == 1: axes = [axes]
    y_min, y_max = df_ana[yc].min(), df_ana[yc].max()
    y_r = abs(y_max-y_min); ylim = (y_min-y_r*0.2, y_max+y_r*0.2)
    for i, (vx, vlabel) in enumerate(zip(xc, xl)):
        df_x = sub_post if (vx.endswith('_change') or 'Post' in vlabel) else sub_pre
        df_y = sub_pre if yc in ['AUC', 'MAE_50', 'MAE_All'] else df_x
        xl_d, yl_d = df_x[df_x['condition']=='Land'][vx].values, df_y[df_y['condition']=='Land'][yc].values
        xw_d, yw_d = df_x[df_x['condition']=='Water'][vx].values, df_y[df_y['condition']=='Water'][yc].values
        plot_relation(axes[i], base_df, vx, yc, xl_d, yl_d, xw_d, yw_d, vlabel, yl, '', method='rmcorr', common_ylim=ylim)
    fig.suptitle(tl, fontweight='bold', fontsize=14, y=-0.02); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{fid}.png'), dpi=300, bbox_inches='tight')

# MAE Comparison
for fid, col, tl, lbl in [('Fig6_50MAE_Comparison', 'MAE_50', 'Fig 6: 50%MAE Comparison', '50%MAE (%)'), ('Fig7_allMAE_Comparison', 'MAE_All', 'Fig 7: Overall MAE Comparison', 'MAE (%)')]:
    fig, ax = plt.subplots(figsize=(6,6))
    plot_bar_land_water(ax, sub_pre[sub_pre['condition']=='Land'][col].values, sub_pre[sub_pre['condition']=='Water'][col].values, tl, lbl)
    fig.savefig(os.path.join(output_dir, f'{fid}.png'), dpi=300, bbox_inches='tight')

print(f"Figures_0217_v2 update complete using {excel_path}.")
