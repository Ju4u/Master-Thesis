import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ============================================================
# GLOBAL FONT SETTINGS (Individual X and Y Tick Control)
# ============================================================
FONT = {
    "base": 14,
    "figure_title": 18,
    "panel_title": 22,
    "axis_label": 17,
    "xtick_label": 15,  # Adjust compound names (x-axis tick size) here
    "ytick_label": 12,  # Adjust numbers (y-axis tick size) here
    "legend": 13,
    "significance": 12,
    "ns_label": 12,
    "panel_label": 20,
}

plt.rcParams.update({
    "font.size": FONT["base"],
    "axes.titlesize": FONT["panel_title"],
    "axes.labelsize": FONT["axis_label"],
    "xtick.labelsize": FONT["xtick_label"],
    "ytick.labelsize": FONT["ytick_label"],
    "legend.fontsize": FONT["legend"],
    "figure.titlesize": FONT["figure_title"],
})

# ================================================================
# DATA ANALYSIS
# ================================================================

csv_filename = "qPCR_AP1_Ct.csv"

if not os.path.exists(csv_filename):
    raise FileNotFoundError(f"Could not find '{csv_filename}' in current directory.")

df = pd.read_csv(csv_filename)

# Parse numeric Cq values
df['Cq_num'] = pd.to_numeric(df['Cq'], errors='coerce')


# Parse sample name and biological replicate number (#1 vs #2)
def parse_sample(s):
    if pd.isna(s) or s == "":
        return None, None
    s = str(s).strip()
    if "#1" in s:
        return s.replace("#1", "").strip(), "1"
    elif "#2" in s:
        return s.replace("#2", "").strip(), "2"
    return s, None


df[['Sample_Group', 'Bio_Rep']] = df['Sample'].apply(lambda x: pd.Series(parse_sample(x)))
df_valid = df.dropna(subset=['Cq_num', 'Sample_Group']).copy()

# Exclude specific outlier well O05 (X24442, Rep 2, GAPDH, Cq ~ 26.05)
if 'Well' in df_valid.columns:
  df_valid = df_valid[df_valid['Well'] != 'O05']
elif 'well' in df_valid.columns:
  df_valid = df_valid[df_valid['well'] != 'O05']
else:
  # Fallback filter matching the sample group, biological replicate, target, and exact Cq value
  df_valid = df_valid[
      ~(
          (df_valid['Sample_Group'] == 'X24442')
          & (df_valid['Bio_Rep'] == '2')
          & (df_valid['Target'] == 'GAPDH')
          & (np.isclose(df_valid['Cq_num'], 26.05, atol=1e-2))
      )
  ]

# Average technical replicates (same Content row)
tech_means = df_valid.groupby(['Sample_Group', 'Bio_Rep', 'Target', 'Content'])['Cq_num'].mean().reset_index()
tech_means.rename(columns={'Cq_num': 'Cq_tech'}, inplace=True)

# Pivot targets to columns
piv = tech_means.pivot(index=['Sample_Group', 'Bio_Rep'], columns='Target', values='Cq_tech').reset_index()

ref_target = 'GAPDH'
pos_control = 'C+'
neg_control = 'C-'
roi_targets = ['COX-2', 'mPGES-1', 'COX-1']  # Target order

# Calculate dCq = Cq(Target) - Cq(GAPDH)
for target in roi_targets:
    piv[f'dCq_{target}'] = piv[target] - piv[ref_target]

# Mean dCq of Control (C+)
control_dCq_means = {tgt: piv[piv['Sample_Group'] == pos_control][f'dCq_{tgt}'].mean() for tgt in roi_targets}

# Calculate ddCq & Relative Expression (2^-ddCq)
results = []
for target in roi_targets:
    for _, row in piv.iterrows():
        sg = row['Sample_Group']
        bio_rep = row['Bio_Rep']
        dcq = row[f'dCq_{target}']
        ddcq = dcq - control_dCq_means[target]
        rel_exp = 2 ** (-ddcq)
        results.append({
            'Sample_Group': sg,
            'Bio_Rep': bio_rep,
            'Target': target,
            'dCq': dcq,
            'ddCq': ddcq,
            'Rel_Exp': rel_exp
        })

df_res = pd.DataFrame(results)

# Filtering sample groups (including X24444)
sample_order = ['C-', 'C+', 'KIT C', 'X24475', 'X24437', 'X24444', 'X24442', 'X24435']
df_filtered = df_res[df_res['Sample_Group'].isin(sample_order)].copy()

# Keep ONLY biological replicate #2 for X24444 (exclude replicate #1)
df_filtered = df_filtered[~((df_filtered['Sample_Group'] == 'X24444') & (df_filtered['Bio_Rep'] != '2'))]

# Statistical Analysis: One-way ANOVA & Dunnett's test vs C+ (excluding X24444 from Dunnett's)
p_values = {}
print("\n" + "=" * 75)
print("             STATISTICAL ANALYSIS SUMMARY (ANOVA & Dunnett's)")
print("=" * 75)

for tgt in roi_targets:
    p_values[tgt] = {}
    tgt_df = df_filtered[df_filtered['Target'] == tgt]

    groups = [g for g in sample_order]
    samples_dict = {g: tgt_df[tgt_df['Sample_Group'] == g]['Rel_Exp'].dropna().values for g in groups if
                    len(tgt_df[tgt_df['Sample_Group'] == g]['Rel_Exp'].dropna().values) > 0}

    valid_groups_for_anova = [vals for vals in samples_dict.values() if len(vals) >= 1]
    if len(valid_groups_for_anova) >= 2:
        f_stat, anova_p = stats.f_oneway(*valid_groups_for_anova)
    else:
        f_stat, anova_p = np.nan, np.nan

    print(f"\nTarget Gene: {tgt}")
    print(f"  One-Way ANOVA: F = {f_stat:.4f}, p = {anova_p:.4f}")

    ctrl_vals = samples_dict.get(pos_control, np.array([]))
    other_groups = [g for g in sample_order if g != pos_control and g != 'X24444' and g in samples_dict]
    group_vals = [samples_dict[g] for g in other_groups]

    if len(ctrl_vals) >= 1 and len(group_vals) > 0:
        dunnett_res = stats.dunnett(*group_vals, control=ctrl_vals)
        for grp, p_val in zip(other_groups, dunnett_res.pvalue):
            p_values[tgt][grp] = p_val
            mean_val = tgt_df[tgt_df['Sample_Group'] == grp]['Rel_Exp'].mean()
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            print(f"  {grp} vs {pos_control:<10} | Mean Rel. Exp: {mean_val:.4f} | p-value: {p_val:.4f} ({sig})")

print("=" * 75 + "\n")

# Compute summary means and SEMs
summary_list = []
for tgt in roi_targets:
    for sg in sample_order:
        sub = df_filtered[(df_filtered['Target'] == tgt) & (df_filtered['Sample_Group'] == sg)]
        vals = sub['Rel_Exp'].dropna().values
        mean_val = vals.mean() if len(vals) > 0 else np.nan
        sem_val = stats.sem(vals, ddof=1) if len(vals) > 1 else 0.0
        summary_list.append({'Target': tgt, 'Sample_Group': sg, 'Mean': mean_val, 'SEM': sem_val})
df_summary = pd.DataFrame(summary_list)

# ================================================================
# PLOTTING
# ================================================================

blot_image_path = "x_axis_table.png"  # Image containing Panel B (table + gel bands)
has_img = blot_image_path is not None

ratios = [3.2]
if has_img:
    ratios.append(0.4)
    ratios.append(0.65)

fig = plt.figure(figsize=(7, 7.2))

gs = fig.add_gridspec(
    len(ratios),
    1,
    height_ratios=ratios,
    hspace=0.05,
    left=0.17,
    right=0.94,
    top=0.93,
    bottom=0.02
)

x_min = -0.5
x_max = len(sample_order) - 0.5
x_pos = np.arange(len(sample_order))

ax = fig.add_subplot(gs[0, 0])

sns.set_style("white")

width = 0.22
separation = 0.27

x_t1 = x_pos - separation  # COX-2
x_t2 = x_pos  # mPGES-1
x_t3 = x_pos + separation  # COX-1

# Compound Colors
compound_colors = {
    'KIT C': '#E0E0E0',
    'X24475': '#C0C0C0',
    'X24437': '#A0A0A0',
    'X24444': '#808080',
    'X24442': '#606060',
    'X24435': '#404040'
}

bar_colors = [
    'white' if n == neg_control else 'black' if n == pos_control else compound_colors[n]
    for n in sample_order
]

target_config = {
    'COX-2': {'x_coords': x_t1, 'hatch': ''},
    'mPGES-1': {'x_coords': x_t2, 'hatch': '///'},
    'COX-1': {'x_coords': x_t3, 'hatch': '...'}
}

# Plot Bars
for i, sg in enumerate(sample_order):
    for tgt in roi_targets:
        cfg = target_config[tgt]
        xc = cfg['x_coords'][i]
        hatch_val = cfg['hatch']

        m_vals = df_summary[(df_summary['Target'] == tgt) & (df_summary['Sample_Group'] == sg)]['Mean'].values
        s_vals = df_summary[(df_summary['Target'] == tgt) & (df_summary['Sample_Group'] == sg)]['SEM'].values

        if len(m_vals) > 0 and not np.isnan(m_vals[0]):
            m = m_vals[0]
            s = s_vals[0]

            bar = ax.bar(xc, m, width, yerr=s, color=bar_colors[i], edgecolor='black', linewidth=1.5, capsize=3,
                         hatch=hatch_val)

            if sg == pos_control and hatch_val != '':
                ax.bar(xc, m, width, color='none', edgecolor='white', linewidth=0, hatch=hatch_val,
                       zorder=bar[0].get_zorder() + 1)

# Plot Individual Replicate Points with White Borders
rep_offsets = [-0.04, 0.04]
for rep_idx, rep_name in enumerate(['1', '2']):
    for i, sg in enumerate(sample_order):
        for tgt in roi_targets:
            cfg = target_config[tgt]
            xc = cfg['x_coords'][i]

            sub = df_filtered[(df_filtered['Target'] == tgt) & (df_filtered['Sample_Group'] == sg) & (
                    df_filtered['Bio_Rep'] == rep_name)]
            if not sub.empty:
                v = sub['Rel_Exp'].values[0]
                ax.scatter(xc + rep_offsets[rep_idx], v, facecolor='black', s=28, zorder=3, alpha=0.8,
                           edgecolors='white', linewidth=0.8)

# Custom Legend Patches
t1_patch = mpatches.Patch(facecolor='#E0E0E0', edgecolor='black', linewidth=1.5)
t2_patch = mpatches.Patch(facecolor='#E0E0E0', edgecolor='black', linewidth=1.5, hatch='///')
t3_patch = mpatches.Patch(facecolor='#E0E0E0', edgecolor='black', linewidth=1.5, hatch='...')

ax.legend(
    [t1_patch, t2_patch, t3_patch],
    roi_targets,
    frameon=False,
    loc='upper left',
    handlelength=1.2,
    handleheight=1.2
)

# Significance Markers & Brackets (Skip X24444)
max_data_top = 0
for i, sg in enumerate(sample_order):
    if sg == pos_control or sg == 'X24444':
        continue
    group_max = 0
    for tgt in roi_targets:
        sub_m = df_summary[(df_summary['Target'] == tgt) & (df_summary['Sample_Group'] == sg)]['Mean'].values
        sub_s = df_summary[(df_summary['Target'] == tgt) & (df_summary['Sample_Group'] == sg)]['SEM'].values
        if len(sub_m) > 0 and not np.isnan(sub_m[0]):
            group_max = max(group_max, sub_m[0] + sub_s[0])
    max_data_top = max(max_data_top, group_max)

dynamic_top_limit = max(max_data_top * 1.08, 4.0)

for i, sg in enumerate(sample_order):
    if sg == pos_control or sg == 'X24444':
        continue

    group_p_vals = [p_values[tgt].get(sg, 1.0) for tgt in roi_targets if sg in p_values[tgt]]
    all_ns = len(group_p_vals) > 0 and all(p >= 0.05 for p in group_p_vals)

    if all_ns:
        group_max = 0
        for tgt in roi_targets:
            sub_m = df_summary[(df_summary['Target'] == tgt) & (df_summary['Sample_Group'] == sg)]['Mean'].values
            sub_s = df_summary[(df_summary['Target'] == tgt) & (df_summary['Sample_Group'] == sg)]['SEM'].values
            if len(sub_m) > 0 and not np.isnan(sub_m[0]):
                group_max = max(group_max, sub_m[0] + sub_s[0])

        b_left = x_t1[i] - (width / 2)
        b_right = x_t3[i] + (width / 2)
        b_y = group_max + 0.10
        b_drop = 0.05

        ax.plot([b_left, b_right], [b_y, b_y], color='black', lw=1.2)
        ax.plot([b_left, b_left], [b_y, b_y - b_drop], color='black', lw=1.2)
        ax.plot([b_right, b_right], [b_y, b_y - b_drop], color='black', lw=1.2)
        ax.text((b_left + b_right) / 2, b_y + 0.01, 'ns', ha='center', va='bottom', fontsize=FONT["ns_label"],
                fontweight='bold')
    else:
        for tgt in roi_targets:
            cfg = target_config[tgt]
            xc = cfg['x_coords'][i]
            p = p_values[tgt].get(sg, None)
            if p is not None:
                sub_m = df_summary[(df_summary['Target'] == tgt) & (df_summary['Sample_Group'] == sg)]['Mean'].values
                sub_s = df_summary[(df_summary['Target'] == tgt) & (df_summary['Sample_Group'] == sg)]['SEM'].values
                if len(sub_m) > 0 and not np.isnan(sub_m[0]):
                    y_pos = sub_m[0] + sub_s[0] + 0.05

                    if p < 0.05:
                        sig_text = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
                    else:
                        sig_text = 'ns'

                    ax.text(xc, y_pos, sig_text, ha='center', va='bottom', fontsize=FONT["significance"], fontweight='bold')

# Axis Labels and Formatting
ax.set_ylabel(f'Relative normalized expression \n Target / GAPDH (fold change of C+)', fontweight='bold')
ax.set_title('COX-2, mPGES-1 & COX-1 qPCR', fontweight='bold', pad=15)
ax.set_xlim(x_min, x_max)
ax.set_xticks(x_pos)
ax.set_xticklabels(sample_order, rotation=45, ha='center', fontsize=FONT["xtick_label"])
ax.set_ylim(0, dynamic_top_limit)

# Color the 'X24444' x-axis tick label red
for tick in ax.get_xticklabels():
    if tick.get_text() == 'X24444':
        tick.set_color('red')
        tick.set_fontweight('bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Reference line representing C+ = 1
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# ================================================================
# TABLE / IMAGE PANEL (Panel B)
# ================================================================

if has_img:
    ax_img = fig.add_subplot(
        gs[2, 0],
        sharex=ax
    )

    try:
        img = mpimg.imread(blot_image_path)
        h_orig, w_orig = img.shape[:2]

        ax_img.imshow(
            img,
            extent=[x_min - 1.75, x_max - 0.3, 0, h_orig],
            aspect='auto',
            interpolation='bilinear',
            clip_on=False
        )
    except FileNotFoundError:
        ax_img.text(
            0.5,
            0.5,
            'Image file not found',
            ha='center',
            va='center',
            fontsize=12
        )

    ax_img.set_xlim(x_min, x_max)
    ax_img.axis('off')

# Save and Display
plt.savefig('qPCR_AP1.png', dpi=600, bbox_inches='tight')
plt.show()