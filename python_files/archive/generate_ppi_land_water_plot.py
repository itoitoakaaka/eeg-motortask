import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

print("PPI Ratio水陸比較グラフを作成中...")

# ===== データ読み込み =====
sep_path = "SEP_raw_temp/SEP_processed/measurements.csv"
sep_df = pd.read_csv(sep_path)

# ID正規化
def normalize_sep_id(file_id):
    return str(file_id)[:5]

sep_df['Subject'] = sep_df['file_id'].apply(normalize_sep_id)

# 対象被験者フィルタ
subjects = [f'id{i:03d}' for i in range(2, 11)]
sep_df = sep_df[sep_df['Subject'].isin(subjects)]

# ===== 可視化関数 =====
def plot_paired_comparison(ax, df, variable, title, output_p_values=False):
    # データ準備
    pre_land = df[(df['phase'] == 'Pre') & (df['condition'] == 'Land')].set_index('Subject')[variable]
    pre_water = df[(df['phase'] == 'Pre') & (df['condition'] == 'Water')].set_index('Subject')[variable]
    post_land = df[(df['phase'] == 'Post') & (df['condition'] == 'Land')].set_index('Subject')[variable]
    post_water = df[(df['phase'] == 'Post') & (df['condition'] == 'Water')].set_index('Subject')[variable]
    
    # 共通の被験者のみ
    subjects_pre = pre_land.index.intersection(pre_water.index)
    subjects_post = post_land.index.intersection(post_water.index)
    
    pre_land = pre_land.loc[subjects_pre]
    pre_water = pre_water.loc[subjects_pre]
    post_land = post_land.loc[subjects_post]
    post_water = post_water.loc[subjects_post]
    
    # Data List Ordered: Land Pre, Land Post, Water Pre, Water Post
    # Requested Order: "LandPre LandPost and WaterPre WaterPost"
    data = [pre_land, post_land, pre_water, post_water]
    
    means = [d.mean() for d in data]
    sems = [d.std(ddof=1) / np.sqrt(len(d)) for d in data]
    
    # X Positions: Group by Condition? 
    # Land Group: Pre(0), Post(1)
    # Water Group: Pre(2.5), Post(3.5)
    x_pos = [0, 1, 2.5, 3.5]
    
    colors = ['orange', 'orange', 'skyblue', 'skyblue']
    labels = ['Pre', 'Post', 'Pre', 'Post']
    
    # Bars
    bars = ax.bar(x_pos, means, yerr=sems, align='center', alpha=0.8, 
                 color=colors, capsize=5, width=0.8, edgecolor='white')
    
    # Hatch for Post (indices 1 and 3)
    bars[1].set_hatch('//')
    bars[3].set_hatch('//')
    bars[1].set_alpha(0.6)
    bars[3].set_alpha(0.6)
    
    # Individual Points
    for i, d in enumerate(data):
        x = np.random.normal(x_pos[i], 0.04, size=len(d))
        ax.scatter(x, d, color='gray', alpha=0.5, s=20, zorder=3)

    # Connecting Lines (Paired Subjects)
    # Compare Land Pre vs Water Pre (Indices 0 vs 2)
    # Compare Land Post vs Water Post (Indices 1 vs 3)
    # Note: Lines will span across.
    for subj in subjects_pre:
        ax.plot([x_pos[0], x_pos[2]], [pre_land[subj], pre_water[subj]], 
               color='gray', alpha=0.2, linewidth=0.5, zorder=2)
               
    for subj in subjects_post:
        ax.plot([x_pos[1], x_pos[3]], [post_land[subj], post_water[subj]], 
               color='gray', alpha=0.2, linewidth=0.5, zorder=2)

    # Statistics / Significance
    t_pre, p_pre = stats.ttest_rel(pre_land, pre_water)
    t_post, p_post = stats.ttest_rel(post_land, post_water)
    
    max_y = max([d.max() for d in data])
    
    # Function for brackets
    def add_significance(x1, x2, p_val, h_offset):
        y = max_y + h_offset
        h = max_y * 0.05
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c='black')
        
        if p_val < 0.001: sig = '***'
        elif p_val < 0.01: sig = '**'
        elif p_val < 0.05: sig = '*'
        elif p_val < 0.1: sig = f'† (p={p_val:.3f})'
        else: sig = 'n.s.'
        
        if p_val < 0.1:
            ax.text((x1+x2)/2, y+h, sig, ha='center', va='bottom', fontsize=10)

    # Brackets: Land Pre vs Water Pre (0 vs 2.5)
    add_significance(x_pos[0], x_pos[2], p_pre, max_y * 0.1)
    
    # Land Post vs Water Post (1 vs 3.5)
    # Offset higher to avoid collision if needed, or same if separate enough?
    # They overlap in X range.
    add_significance(x_pos[1], x_pos[3], p_post, max_y * 0.25)
    
    # Category Labels
    # Draw text for "Land" and "Water"
    # Land center: (0+1)/2 = 0.5
    # Water center: (2.5+3.5)/2 = 3.0
    ylim = ax.get_ylim()
    # Adjust ticks
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    
    # Group Labels defined in main section or here via Text

    
    # 軸設定
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Ratio (%)', fontsize=12)
    
    # セクションラベル (Pre / Post)
    ylim = ax.get_ylim()
    ax.text(0.5, ylim[0] - (ylim[1]-ylim[0])*0.15, 'Pre', ha='center', fontsize=14, fontweight='bold')
    ax.text(3.0, ylim[0] - (ylim[1]-ylim[0])*0.15, 'Post', ha='center', fontsize=14, fontweight='bold')

    # 背景グリッド
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 枠を消す
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ===== メイン処理 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

plot_paired_comparison(axes[0], sep_df, 'pp30_ratio', 'PPI 30ms Ratio')
plot_paired_comparison(axes[1], sep_df, 'pp100_ratio', 'PPI 100ms Ratio')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Pre/Postラベルのために下側を空ける

# 保存
output_path = "SEP_raw_temp/Figures/PPI_Ratio_Land_vs_Water.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"図を保存: {output_path}")
