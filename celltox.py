import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import dunnett
import seaborn as sns

# ============================================================
# GLOBAL FONT SETTINGS (Matched with ELISA script Panel A)
# ============================================================
FONT = {
    "base": 14,
    "figure_title": 22,
    "panel_title": 28,
    "axis_label": 22,
    "tick_label": 18,
    "legend": 14,
    "panel_label": 35,
    "significance": 19,
    "custom_axis_label": 18,
    "concentration_label": 16,
    "compound_group_label": 18,
}

plt.rcParams.update({
    "font.size": FONT["base"],
    "axes.titlesize": FONT["panel_title"],
    "axes.labelsize": FONT["axis_label"],
    "xtick.labelsize": FONT["tick_label"],
    "ytick.labelsize": FONT["tick_label"],
    "legend.fontsize": FONT["legend"],
    "figure.titlesize": FONT["figure_title"],
})

# ============================================================
# 1. LOAD DATA FROM EXCEL
# ============================================================
file_path = "CellTox_AP1.xlsx"

df_raw = pd.read_excel(
    file_path,
    sheet_name=0,
    skiprows=29,
    nrows=8,
    usecols="B:M",
    header=None
)

raw_matrix = df_raw.values.astype(float)

# Disregard values of A5 and A6
#raw_matrix[0, 1] = np.nan
raw_matrix[0, 4] = np.nan
raw_matrix[0, 5] = np.nan

# Background values from H7-H9
background_vals = raw_matrix[7, 6:9]

if np.isnan(background_vals).any() or background_vals[0] < 1000:
    bg_mean = np.mean([7045, 7076, 7067])
else:
    bg_mean = np.nanmean(background_vals)

# ============================================================
# 2. BACKGROUND SUBTRACTION & NORMALIZATION
# ============================================================
data_bg_sub = raw_matrix - bg_mean

# Viability control values
via_ctrl_vals = data_bg_sub[6, 0:3]
via_ctrl_mean = np.mean(via_ctrl_vals)

# Normalize relative to viability control mean
data_norm = data_bg_sub / via_ctrl_mean

# ============================================================
# 3. MAP DATA TO GROUPS
# ============================================================
compound_map = {
    "KC": "KIT C",
    "75": "X24475",
    "37": "X24437",
    "44": "X24444",
    "42": "X24442",
    "35": "X24435"
}

short_compounds = ["KC", "75", "37", "44", "42", "35"]
groups = {}

# Controls
groups["Via"] = data_norm[6, 0:3]
groups["Tox"] = data_norm[6, 3:6]
groups["IL"] = data_norm[7, 0:3]
groups["DMSO"] = data_norm[7, 3:6]

# Test compounds at 5 µM (Rows A-C)
for col_idx, sc in enumerate(short_compounds):
    groups[f"{compound_map[sc]} 5 uM"] = data_norm[0:3, col_idx]

# Test compounds at 20 µM (Rows D-F)
for col_idx, sc in enumerate(short_compounds):
    groups[f"{compound_map[sc]} 20 uM"] = data_norm[3:6, col_idx]

control_groups = ["Tox", "Via", "IL", "DMSO"]
compound_full_names = list(compound_map.values())

ordered_keys = (
    control_groups
    + [f"{c} 5 uM" for c in compound_full_names]
    + [f"{c} 20 uM" for c in compound_full_names]
)

# ============================================================
# 4. STATISTICAL ANALYSIS (ANOVA & DUNNETT'S TEST)
# ============================================================
clean_samples = [
    groups[k][~np.isnan(groups[k])]
    for k in ordered_keys
]

f_stat, p_val_anova = stats.f_oneway(*clean_samples)
print(f"One-way ANOVA: F = {f_stat:.4f}, p = {p_val_anova:.4e}")

# Dunnett's post-hoc test against Via control
treatment_keys = [k for k in ordered_keys if k != "Via"]
treatment_samples = [
    groups[k][~np.isnan(groups[k])]
    for k in treatment_keys
]
control_sample = groups["Via"][~np.isnan(groups["Via"])]

res_dunnett = dunnett(*treatment_samples, control=control_sample)

p_values = {"Via": 1.0}
for k, p in zip(treatment_keys, res_dunnett.pvalue):
    p_values[k] = p


def get_sig_label(p):
    if pd.isna(p) or p >= 0.05:
        return "ns"
    elif p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    else:
        return "*"


# Calculate summary statistics
group_summary = {}

for k in ordered_keys:
    vals = groups[k]
    valid_vals = vals[~np.isnan(vals)]

    group_summary[k] = {
        "mean": np.nanmean(vals),
        "std": np.std(valid_vals, ddof=1) if len(valid_vals) > 1 else 0.0,
        "sem": (
            np.std(valid_vals, ddof=1) / np.sqrt(len(valid_vals))
            if len(valid_vals) > 1 else 0.0
        ),
        "sig": get_sig_label(p_values[k])
    }

# ============================================================
# 5. PRINT VERIFICATION TABLE
# ============================================================
table_data = []

for k in ordered_keys:
    vals = groups[k]

    if k in ["Via", "Tox", "IL", "DMSO"]:
        comp_name = k
        conc_val = "Control"
    else:
        parts = k.split(" ")
        comp_name = " ".join(parts[:-2])
        conc_val = f"{parts[-2]} {parts[-1]}"

    table_data.append({
        "Group": comp_name,
        "Condition/Conc": conc_val,
        "Rep 1": vals[0] if len(vals) > 0 else np.nan,
        "Rep 2": vals[1] if len(vals) > 1 else np.nan,
        "Rep 3": vals[2] if len(vals) > 2 else np.nan,
        "Mean (Fold)": group_summary[k]["mean"],
        "SEM": group_summary[k]["sem"],
        "p-value": p_values[k],
        "Significance": group_summary[k]["sig"]
    })

summary_df = pd.DataFrame(table_data)

print("\n" + "=" * 115)
print("CELLTOX GREEN ASSAY - STATISTICAL SUMMARY & REPLICATE VERIFICATION TABLE")
print("=" * 115)
print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
    )
)
print("=" * 115 + "\n")

# ============================================================
# 6. PLOTTING (PORTRAIT FIGURE, REDUCED LEFT GAP)
# ============================================================
sns.set_style("white")

# Portrait aspect ratio
fig, ax1 = plt.subplots(figsize=(8, 10))

CONTROL_COLORS = {
    "Via": "mediumseagreen",
    "Tox": "crimson",
    "IL": "orange",
    "DMSO": "khaki"
}

COMPOUND_COLORS = {
    "KIT C": "#E0E0E0",
    "X24475": "#C0C0C0",
    "X24437": "#A0A0A0",
    "X24444": "#808080",
    "X24442": "#606060",
    "X24435": "#404040"
}

current_x = 0
bar_width = 0.38
separation = 0.22
control_step = 1.0
group_step = 1.0

all_bar_x = []
all_comps = []
all_concs = []
group_centers = []
group_labels = []

# Updated to use SEM
max_y_global = max(
    s["mean"] + s["sem"] for s in group_summary.values()
)
sig_data = []

# Plot controls
for ctrl in control_groups:
    row = group_summary[ctrl]

    ax1.bar(
        current_x,
        row["mean"],
        width=bar_width,
        yerr=row["sem"],  # Updated to SEM
        color=CONTROL_COLORS[ctrl],
        edgecolor="black",
        linewidth=1.5,
        capsize=5
    )

    all_bar_x.append(current_x)
    all_comps.append(ctrl)
    all_concs.append("N/A")

    if ctrl != "Via" and row["sig"] != "ns":
        sig_data.append({
            "pos": current_x,
            "sig": row["sig"],
            "comp": ctrl,
            "y_val": row["mean"] + row["sem"]  # Updated to SEM
        })

    group_centers.append(current_x)
    group_labels.append(ctrl)
    current_x += control_step

# Plot compounds: 5 and 20 µM bars around each group center
for comp in compound_full_names:
    center_x = current_x

    for conc, offset in [(5, -separation), (20, separation)]:
        bar_x = center_x + offset
        key = f"{comp} {conc} uM"
        row = group_summary[key]

        ax1.bar(
            bar_x,
            row["mean"],
            width=bar_width,
            yerr=row["sem"],  # Updated to SEM
            color=COMPOUND_COLORS[comp],
            edgecolor="black",
            linewidth=1.5,
            capsize=5
        )

        all_bar_x.append(bar_x)
        all_comps.append(comp)
        all_concs.append(conc)

        if row["sig"] != "ns":
            sig_data.append({
                "pos": bar_x,
                "sig": row["sig"],
                "comp": comp,
                "y_val": row["mean"] + row["sem"]  # Updated to SEM
            })

    group_centers.append(center_x)
    group_labels.append(comp)
    current_x += group_step

# Significance annotations
for item in sig_data:
    ax1.text(
        item["pos"],
        item["y_val"] + (max_y_global * 0.03),
        item["sig"],
        ha="center",
        va="bottom",
        fontsize=FONT["significance"],
        fontweight="bold"
    )

# Table-style x-axis labels
trans = ax1.get_xaxis_transform()
row_names = ["1 ng/mL IL-1β", "Conc. (µM)"]

# CHANGE: position row labels closer to the first bar
x_label_pos = min(all_bar_x) - 0.25

y_start = -0.05
y_step = 0.06

ax1.text(
    x_label_pos, y_start, row_names[0],
    transform=trans,
    ha="right",
    va="center",
    fontsize=FONT["custom_axis_label"],
    clip_on=False
)

ax1.text(
    x_label_pos, y_start - y_step, row_names[1],
    transform=trans,
    ha="right",
    va="center",
    fontsize=FONT["custom_axis_label"],
    clip_on=False
)

for bx, bcomp, bconc in zip(all_bar_x, all_comps, all_concs):
    il1b_val = "+" if (
        bcomp in compound_full_names or bcomp == "IL"
    ) else "-"
    conc_val = str(bconc) if bconc != "N/A" else "-"

    ax1.text(
        bx, y_start, il1b_val,
        transform=trans,
        ha="center",
        va="center",
        fontsize=FONT["concentration_label"],
        clip_on=False
    )

    # Mark the 5 µM labels for X24435 and X24442 in red
    conc_label_color = (
        "red"
        if bcomp in ["X24435", "X24442"] and bconc == 5
        else "black"
    )

    ax1.text(
        bx, y_start - y_step, conc_val,
        transform=trans,
        ha="center",
        va="center",
        fontsize=FONT["concentration_label"],
        fontstretch="condensed",
        color=conc_label_color,
        clip_on=False
    )

for x_pos, label in zip(group_centers, group_labels):
    label_x = x_pos + 0.4 if label in compound_full_names else x_pos

    ax1.text(
        label_x, y_start - (2 * y_step) - 0.02, label,
        transform=trans,
        ha="right",
        va="top",
        rotation=40,
        rotation_mode="anchor",
        fontsize=FONT["compound_group_label"],
        fontweight="bold",
        clip_on=False
    )

# Reference line and axis formatting
ax1.axhline(
    1.0,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="Viability Control (1-fold)"
)

ax1.set_ylabel(
    "Fluorescence \n(time-fold of Via)",
    fontsize=FONT["axis_label"],
    fontweight="bold"
)

ax1.set_title(
    "Cytotoxicity Assay",
    fontsize=FONT["panel_title"],
    fontweight="bold",
    pad=14
)

ax1.set_xticks(all_bar_x)
ax1.set_xticklabels([])
ax1.set_ylim(0, max_y_global * 1.25)

# CHANGE: reduce excess space to the left of the first bar
ax1.set_xlim(min(all_bar_x) - 0.6, max(all_bar_x) + 0.5)

# Spines
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_linewidth(1.5)
ax1.spines["bottom"].set_linewidth(1.5)

# Keep sufficient room for rotated compound labels
fig.subplots_adjust(
    bottom=0.22,
    top=0.93,
    left=0.17,
    right=0.98
)

plt.savefig(
    "CellTox.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()