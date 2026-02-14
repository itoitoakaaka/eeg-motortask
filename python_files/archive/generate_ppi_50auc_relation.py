import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

print("PPI ratio変化量 vs Adaptation AUC 相関図を作成中...")

# ===== 設定 =====
output_dir = "SEP_raw_temp/Figures"
output_filename = "Relation_AdaptAUC_vs_PPIchange.png"
task_excel = "task/task.xlsx"
sep_csv = "SEP_raw_temp/SEP_processed/measurements.csv"

# ===== データ読み込み =====
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_task = pd.read_excel(task_excel)
df_sep = pd.read_csv(sep_csv)

# ID正規化 (id002-010)
def normalize_id(text_id):
    if pd.isna(text_id): return None
    s_id = str(text_id)
    if 'test' in s_id:
        num = s_id.replace('test', '').split('-')[0]
        return f'id{num}'
    if s_id.startswith('id') and len(s_id) >= 5:
        return s_id[:5]
    return None

df_task['Subject'] = df_task['ID'].apply(normalize_id)
df_sep['Subject'] = df_sep['file_id'].apply(normalize_id)

subjects = [f'id{i:03d}' for i in range(2, 11)] # id002-id010
df_task = df_task[df_task['Subject'].isin(subjects)]
df_sep = df_sep[df_sep['Subject'].isin(subjects)]

# ===== データ加工 =====
# 1. SEP変化量 (Post - Pre)
sep_pivot = df_sep.pivot_table(
    index=['Subject', 'condition'],
    columns='phase',
    values=['pp30_ratio', 'pp100_ratio']
)

changes = []
for subject in subjects:
    for condition in ['Land', 'Water']:
        try:
            pp30_pre = sep_pivot.loc[(subject, condition), ('pp30_ratio', 'Pre')]
            pp30_post = sep_pivot.loc[(subject, condition), ('pp30_ratio', 'Post')]
            pp100_pre = sep_pivot.loc[(subject, condition), ('pp100_ratio', 'Pre')]
            pp100_post = sep_pivot.loc[(subject, condition), ('pp100_ratio', 'Post')]
            
            changes.append({
                'Subject': subject,
                'Condition': condition,
                'pp30_ratio_change': pp30_post - pp30_pre,
                'pp100_ratio_change': pp100_post - pp100_pre
            })
        except KeyError:
            continue

df_change = pd.DataFrame(changes)

# 2. Merge with Task (AUC)
# Check columns
if 'AUC' not in df_task.columns:
    print("Error: 'AUC' column not found in task.xlsx")
    exit()

df_auc = df_task[['Subject', 'Condition', 'AUC']].copy()
merged = pd.merge(df_change, df_auc, on=['Subject', 'Condition'], how='inner')

print(f"データ数: {len(merged)} (Land: {len(merged[merged['Condition']=='Land'])}, Water: {len(merged[merged['Condition']=='Water'])})")

# ===== 統計計算 =====
def calc_corr(df_sub, x_col, y_col):
    if len(df_sub) < 3: return np.nan, np.nan
    return stats.pearsonr(df_sub[x_col], df_sub[y_col])

# pp30 correlations
all_r30, all_p30 = calc_corr(merged, 'pp30_ratio_change', 'AUC')
land_r30, land_p30 = calc_corr(merged[merged['Condition']=='Land'], 'pp30_ratio_change', 'AUC')
water_r30, water_p30 = calc_corr(merged[merged['Condition']=='Water'], 'pp30_ratio_change', 'AUC')

# pp100 correlations
all_r100, all_p100 = calc_corr(merged, 'pp100_ratio_change', 'AUC')
land_r100, land_p100 = calc_corr(merged[merged['Condition']=='Land'], 'pp100_ratio_change', 'AUC')
water_r100, water_p100 = calc_corr(merged[merged['Condition']=='Water'], 'pp100_ratio_change', 'AUC')

print(f"\npp30_ratio変化量 vs AUC:")
print(f"  Land:  r={land_r30:.3f}, p={land_p30:.3f}")
print(f"  Water: r={water_r30:.3f}, p={water_p30:.3f}")
print(f"  All:   r={all_r30:.3f}, p={all_p30:.3f}")

print(f"\npp100_ratio変化量 vs AUC:")
print(f"  Land:  r={land_r100:.3f}, p={land_p100:.3f}")
print(f"  Water: r={water_r100:.3f}, p={water_p100:.3f}")
print(f"  All:   r={all_r100:.3f}, p={all_p100:.3f}")

# ===== 可視化 =====
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# データ準備
land_data = merged[merged['Condition'] == 'Land']
water_data = merged[merged['Condition'] == 'Water']
paired_subjects = set(land_data['Subject']) & set(water_data['Subject'])

# --- 左パネル: PPI30 ---
ax1 = axes[0]
# 散布図
ax1.scatter(land_data['pp30_ratio_change'], land_data['AUC'], 
           color='orange', alpha=0.7, s=80, edgecolors='white', zorder=3, label='Land')
ax1.scatter(water_data['pp30_ratio_change'], water_data['AUC'],
           color='skyblue', alpha=0.7, s=80, edgecolors='white', zorder=3, label='Water')

# 回帰直線
for data, color in [(land_data, 'orange'), (water_data, 'skyblue')]:
    if len(data) > 2:
        m, b = np.polyfit(data['pp30_ratio_change'], data['AUC'], 1)
        x_range = np.array([data['pp30_ratio_change'].min(), data['pp30_ratio_change'].max()])
        ax1.plot(x_range, m*x_range + b, color=color, linestyle='--', linewidth=1.5, alpha=0.8)

# 線で結ぶ
for subj in paired_subjects:
    l = land_data[land_data['Subject']==subj]
    w = water_data[water_data['Subject']==subj]
    if not l.empty and not w.empty:
        ax1.plot([l.iloc[0]['pp30_ratio_change'], w.iloc[0]['pp30_ratio_change']],
                 [l.iloc[0]['AUC'], w.iloc[0]['AUC']],
                 color='gray', alpha=0.4, linewidth=1, zorder=2)

# 統計情報 - Allを追加
stats_text = f"All: r={all_r30:.3f}, p={all_p30:.3f}\nLand: r={land_r30:.3f}, p={land_p30:.3f}\nWater: r={water_r30:.3f}, p={water_p30:.3f}"
ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
         verticalalignment='top', horizontalalignment='right',
         fontsize=10, fontweight='bold', color='black')

ax1.set_xlabel('PPI30 Ratio Change (%)', fontsize=14)
ax1.set_ylabel('Adaptation AUC', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(loc='lower left', fontsize=12, frameon=False)


# --- 右パネル: PPI100 ---
ax2 = axes[1]
# 散布図
ax2.scatter(land_data['pp100_ratio_change'], land_data['AUC'], 
           color='orange', alpha=0.7, s=80, edgecolors='white', zorder=3, label='Land')
ax2.scatter(water_data['pp100_ratio_change'], water_data['AUC'],
           color='skyblue', alpha=0.7, s=80, edgecolors='white', zorder=3, label='Water')

# 回帰直線
for data, color in [(land_data, 'orange'), (water_data, 'skyblue')]:
    if len(data) > 2:
        m, b = np.polyfit(data['pp100_ratio_change'], data['AUC'], 1)
        x_range = np.array([data['pp100_ratio_change'].min(), data['pp100_ratio_change'].max()])
        ax2.plot(x_range, m*x_range + b, color=color, linestyle='--', linewidth=1.5, alpha=0.8)

# 線で結ぶ
for subj in paired_subjects:
    l = land_data[land_data['Subject']==subj]
    w = water_data[water_data['Subject']==subj]
    if not l.empty and not w.empty:
        ax2.plot([l.iloc[0]['pp100_ratio_change'], w.iloc[0]['pp100_ratio_change']],
                 [l.iloc[0]['AUC'], w.iloc[0]['AUC']],
                 color='gray', alpha=0.4, linewidth=1, zorder=2)

# 統計情報 - Allを追加
stats_text = f"All: r={all_r100:.3f}, p={all_p100:.3f}\nLand: r={land_r100:.3f}, p={land_p100:.3f}\nWater: r={water_r100:.3f}, p={water_p100:.3f}"
ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
         verticalalignment='top', horizontalalignment='right',
         fontsize=10, fontweight='bold', color='black')

ax2.set_xlabel('PPI100 Ratio Change (%)', fontsize=14)
ax2.set_ylabel('Adaptation AUC', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)

# 凡例をグラフ外の上部中央に配置 (ユーザー要望のスタイル維持)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=12, frameon=False)
# サブプロットの凡例は非表示に
ax1.get_legend().remove()

plt.subplots_adjust(left=0.15, right=0.9, top=0.85, bottom=0.20, wspace=0.3)

save_path = os.path.join(output_dir, output_filename)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"図を保存: {save_path}")
