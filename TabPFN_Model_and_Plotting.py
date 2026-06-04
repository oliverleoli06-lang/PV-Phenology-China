import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shap
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

# ================= 0. 全局导出设置 =================
# 保证 PDF 和 PS 导出真实字体
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Arial'] # 统一使用 Arial 字体

# ================= 1. 路径设置与数据读取 =================
# 使用相对路径：请确保此脚本与 CSV 文件放在同一个文件夹内
shap_values_path = "SHAP_Values_SOS_PVZone.csv"
x_test_path = "Feature_Values_SOS_PVZone.csv"
output_dir = "./Output_Figures"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

shap_values = pd.read_csv(shap_values_path).values
X_test = pd.read_csv(x_test_path)

# ================= 2. 绘制英文配图 =================
plt.figure()

ref_colors = ['#5885B5', '#F8F8F8', '#D65B4C']
custom_cmap = LinearSegmentedColormap.from_list('custom_red_blue', ref_colors)

shap.summary_plot(shap_values, X_test, show=False, cmap=custom_cmap,
                  plot_size=(14, 6), color_bar=False, alpha=0.85)

fig1 = plt.gcf()
ax_main = plt.gca()

# 设置正确的英文标签
ax_main.set_xlabel("SHAP value (impact on ΔSOS_PV)", fontsize=14)

for tick in ax_main.get_yticklabels():
    tick.set_fontproperties(plt.matplotlib.font_manager.FontProperties(family='Arial'))
    tick.set_fontsize(13)

for spine in ['top', 'bottom', 'left', 'right']:
    ax_main.spines[spine].set_visible(True)
    ax_main.spines[spine].set_color('black')
    ax_main.spines[spine].set_linewidth(1.2)

ax_main.tick_params(axis='y', which='major', left=True, right=True, direction='in', length=4, pad=5)
ax_main.tick_params(axis='x', which='major', bottom=True, top=True, direction='in', length=6)
ax_main.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_main.tick_params(axis='x', which='minor', bottom=True, top=True, direction='in', length=3)

ax_main.grid(True, which='major', axis='x', linestyle='--', color='#b0b0b0', alpha=0.6, zorder=0)
ax_main.grid(True, which='minor', axis='x', linestyle='--', color='#b0b0b0', alpha=0.6, zorder=0)
ax_main.set_axisbelow(True)

divider = make_axes_locatable(ax_main)
cax = divider.append_axes("top", size="3%", pad=0.1)

norm = plt.Normalize(vmin=-1, vmax=1)
sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])

cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
cb.set_ticks([])
cb.outline.set_linewidth(1.2)

cax.text(0.0, 1.8, "Low", transform=cax.transAxes, ha='left', va='bottom', fontsize=12)
cax.text(1.0, 1.8, "High", transform=cax.transAxes, ha='right', va='bottom', fontsize=12)

# 统一输出文件名
pdf_path = os.path.join(output_dir, "SHAP_Summary_SOS_PVZone.pdf")
png_path = os.path.join(output_dir, "SHAP_Summary_SOS_PVZone.png")

plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.savefig(png_path, format='png', dpi=900, bbox_inches='tight')
plt.close()

# ================= 3. 完成提示 =================
print("绘图完成！图片已保存至 Output_Figures 文件夹。")