import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 0. 全局导出与字体设置 =================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Arial'] # 统一使用 Arial 字体

# ================= 1. 数据读取与实时计算 =================
# 改为相对路径：请确保此脚本与 CSV 文件放在同一个文件夹内
raw_shap_path = "SHAP_Values_SOS_PVZone.csv"
output_dir = "./Output_Figures"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    shap_df = pd.read_csv(raw_shap_path)

    # 第一步：计算全场所有特征的 SHAP 绝对值总和 (作为分母)
    grand_total_abs = 0
    for col in shap_df.columns:
        if col in ['ID', 'Unnamed: 0'] or shap_df[col].dtype == 'object':
            continue
        grand_total_abs += np.abs(shap_df[col].values).sum()

    results = []
    # 第二步：按照文献公式计算每个特征的贡献
    for col in shap_df.columns:
        if col in ['ID', 'Unnamed: 0'] or shap_df[col].dtype == 'object':
            continue

        vals = shap_df[col].values

        # 1. 计算总绝对贡献百分比 (柱子总长度)
        total_abs_val = np.abs(vals).sum()
        relative_contrib = (total_abs_val / grand_total_abs) * 100

        # 2. 提取中位数的符号 (决定柱子颜色)
        median_val = np.median(vals)
        sign = 1 if median_val > 0 else -1

        # 3. 分配给正向或负向
        pos_contrib = relative_contrib if sign == 1 else 0
        neg_contrib = relative_contrib if sign == -1 else 0

        results.append({
            'Feature': col,
            'Positive': pos_contrib,
            'Negative': neg_contrib,
            'Relative_Contribution': relative_contrib
        })

    rose_data = pd.DataFrame(results)

    # 按照重要性从大到小排序
    rose_data = rose_data.sort_values(by='Relative_Contribution', ascending=False).reset_index(drop=True)

    features = rose_data['Feature'].values
    pos_vals = rose_data['Positive'].values
    neg_vals = rose_data['Negative'].values

except Exception as e:
    print(f"Error: {e}")
    exit()

# ================= 2. 绘制英文玫瑰图 =================
COLOR_NEG = '#A0BCE0'  # 水蓝色 (Negative)
COLOR_POS = '#ECA8A8'  # 水粉色 (Positive)

fig, ax = plt.subplots(figsize=(8.5, 8), subplot_kw={'projection': 'polar'})
fig.subplots_adjust(left=0.15)

ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
ax.set_thetamin(0)
ax.set_thetamax(90)

N = len(features)
width = (np.pi / 2) / N
theta = np.linspace(0 + width / 2, np.pi / 2 - width / 2, N)

ax.bar(theta, pos_vals, width=width, color=COLOR_POS,
       edgecolor='black', linewidth=1.0, alpha=0.95, label="Positive", zorder=3)

ax.bar(theta, neg_vals, width=width, bottom=pos_vals, color=COLOR_NEG,
       edgecolor='black', linewidth=1.0, alpha=0.95, label="Negative", zorder=3)

ax.set_xticks(theta)
ax.set_xticklabels(features, fontsize=13, fontweight='bold', ha='left', va='bottom')
ax.tick_params(axis='x', pad=8)

max_pct = max(np.array(pos_vals) + np.array(neg_vals))
ax.set_ylim(0, max_pct * 1.05)

fig.text(0.06, 0.5, "Relative contributions (%)", rotation=90,
         va='center', ha='center', fontsize=16, fontweight='bold')

ax.grid(True, which='major', axis='y', linestyle='--', color='gray', alpha=0.5)
ax.grid(True, which='major', axis='x', linestyle='-', color='gray', alpha=0.3)
ax.set_axisbelow(True)

ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15),
          frameon=True, edgecolor='black', fancybox=False, fontsize=12)

# 统一输出文件名
pdf_path = os.path.join(output_dir, "RoseChart_Contributions_SOS.pdf")
png_path = os.path.join(output_dir, "RoseChart_Contributions_SOS.png")

plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.savefig(png_path, format='png', dpi=900, bbox_inches='tight')
plt.close()

print("Plotting complete! Figures saved to Output_Figures folder.")