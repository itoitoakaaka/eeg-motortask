import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_EXCEL = os.path.join(BASE_DIR, 'task.xlsx')
OUTPUT_PLOT_DIR = os.path.join(BASE_DIR, 'output_plots')
ROOT_RESEARCH = os.path.dirname(BASE_DIR)

def get_base_id(full_id):
    if '-' in full_id:
        return full_id.split('-')[0]
    return full_id

def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def create_combined_mae_plot(df):
    percentages = [10, 30, 50, 70, 90]
    mae_cols = [f'{p}%MAE' for p in percentages]
    means = df.groupby('Condition')[mae_cols].mean()
    stds = df.groupby('Condition')[mae_cols].std()
    counts = df.groupby('Condition')[mae_cols].count()
    ses = stds / np.sqrt(counts)
    
    plt.figure(figsize=(7, 7.5))
    if 'Land' in means.index:
        plt.errorbar(percentages, means.loc['Land'], yerr=ses.loc['Land'], 
                     label='Land', color='orange', marker='o', capsize=5, linewidth=2)
    if 'Water' in means.index:
        plt.errorbar(percentages, means.loc['Water'], yerr=ses.loc['Water'], 
                     label='Water', color='skyblue', marker='o', capsize=5, linewidth=2)
    
    plt.xlabel('Stimulus Speed (%)', fontsize=14)
    plt.ylabel('MAE (%)', fontsize=14)
    plt.title('Mean Absolute Error (MAE) by Speed', fontweight='bold', fontsize=16)
    plt.xticks(percentages, [f'{p}%' for p in percentages], fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(ROOT_RESEARCH, "Fig9_All_MAE.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_overall_comparison_plot(df, metric, title, filename):
    base_ids = sorted(df['BaseID'].unique())
    land_vals, water_vals, valid_ids = [], [], []
    for bid in base_ids:
        subset = df[df['BaseID'] == bid]
        l_row = subset[subset['Condition'] == 'Land']
        w_row = subset[subset['Condition'] == 'Water']
        if not l_row.empty and not w_row.empty:
            land_vals.append(l_row.iloc[0][metric])
            water_vals.append(w_row.iloc[0][metric])
            valid_ids.append(bid)
    
    if not valid_ids: return
    plt.figure(figsize=(6, 7.5))
    plt.bar([0, 1], [np.mean(land_vals), np.mean(water_vals)], color=['orange', 'skyblue'], alpha=0.3, width=0.5)
    cmap = plt.get_cmap('tab20')
    for i, bid in enumerate(valid_ids):
        plt.plot([0, 1], [land_vals[i], water_vals[i]], marker='o', color=cmap(i%20), lw=2, label=bid)
    plt.xticks([0, 1], ['Land', 'Water'], fontsize=14, fontweight='bold')
    plt.ylabel(title, fontsize=14)
    plt.title(title, fontweight='bold', fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_RESEARCH, filename), dpi=300)
    plt.close()

def process_files():
    if not os.path.exists(OUTPUT_PLOT_DIR): os.makedirs(OUTPUT_PLOT_DIR)
    results = []
    for folder in ['land', 'water']:
        cond_dir = os.path.join(BASE_DIR, folder)
        files_50 = glob.glob(os.path.join(cond_dir, '*+50%.CSV'))
        for f_50 in files_50:
            id_val = os.path.basename(f_50).split('+')[0]
            if 'test001' in id_val: continue
            f_3mean = f_50.replace('+50%.CSV', '+3MEAN.CSV')
            f_tsk2 = f_50.replace('+50%.CSV', '+tsk2.CSV')
            try:
                max_mean = pd.read_csv(f_3mean, header=6).iloc[0, 5]
                df_50 = pd.read_csv(f_50, header=6)
                trials = df_50.iloc[-1]['No [N]']
                err_abs = (df_50['SP [deg/sec]'] - max_mean*0.5).abs()
                pct_err = (err_abs / max_mean) * 100
                auc = np.trapz(pct_err, df_50['No [N]'])
                autocorr = (df_50['SP [deg/sec]'] - max_mean*0.5).autocorr(lag=1)
                
                mae_vals = {}
                df_tsk = pd.read_csv(f_tsk2, header=6)
                for p in [10, 30, 50, 70, 90]:
                    sub = df_tsk[df_tsk['Stim [%]'] == p]
                    mae_vals[f'{p}%MAE'] = ((sub['deg/sec [deg/sec]'] - max_mean*p/100).abs() / max_mean).mean() * 100
                
                results.append({
                    'ID': id_val, 'BaseID': get_base_id(id_val), 'Condition': folder.capitalize(),
                    'Max_Mean': max_mean, 'Trials': trials, 'AUC': auc, 'AutoCorr': autocorr,
                    **mae_vals, 'Overall_MAE': np.mean(list(mae_vals.values()))
                })
            except Exception as e: print(f"Error {id_val}: {e}")

    df_res = pd.DataFrame(results).sort_values(['BaseID', 'Condition'])
    df_res.to_excel(OUTPUT_EXCEL, index=False)
    create_combined_mae_plot(df_res)
    create_overall_comparison_plot(df_res, 'Overall_MAE', 'Overall MAE (%)', 'Fig10_MAE_Comparison.png')
    create_overall_comparison_plot(df_res, 'AUC', 'Adaptation AUC', 'Fig11_AUC_Comparison.png')
    print("Done.")

if __name__ == "__main__":
    process_files()
