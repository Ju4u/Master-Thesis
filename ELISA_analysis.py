import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
import re
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ============================================================
# GLOBAL FONT SETTINGS
# All font sizes are in points.
# Change these values to control the entire figure.
# ============================================================

FONT = {
    # General Matplotlib defaults
    "base": 14,

    # Titles
    "figure_title": 22,
    "panel_title": 28,

    # Standard axis formatting
    "axis_label": 22,
    "tick_label": 18,
    "legend": 14,

    # Panel identifiers (A, B, C)
    "panel_label": 35,

    # Panel A: bar chart
    "significance": 19,
    "custom_axis_label": 18,
    "concentration_label": 16,
    "compound_group_label": 22,

    # Panel C: regression plot
    "compound_annotation": 16,
}

# Set global defaults as a fallback.
# Explicit font sizes below also reference FONT, so these
# settings are no longer overridden by hard-coded numbers.
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
# 1. HELPER FUNCTIONS FOR EXCEL EXTRACTION
# ============================================================

def col2num(col_str):
    num = 0
    for c in col_str.upper():
        num = num * 26 + (ord(c) - ord("A")) + 1
    return num - 1


def extract_range(df, range_str):
    if ":" in range_str:
        start, end = range_str.split(":")

        start_col = col2num(
            re.findall(r"[A-Za-z]+", start)[0]
        )
        start_row = int(
            re.findall(r"\d+", start)[0]
        ) - 1

        end_col = col2num(
            re.findall(r"[A-Za-z]+", end)[0]
        )
        end_row = int(
            re.findall(r"\d+", end)[0]
        ) - 1

        vals = df.iloc[
            start_row:end_row + 1,
            start_col:end_col + 1
        ].values.flatten()

    else:
        col = col2num(
            re.findall(r"[A-Za-z]+", range_str)[0]
        )
        row = int(
            re.findall(r"\d+", range_str)[0]
        ) - 1

        vals = [df.iloc[row, col]]

    cleaned_vals = [
        str(v).replace(",", ".") if isinstance(v, str) else v
        for v in vals
    ]

    return pd.to_numeric(cleaned_vals, errors="coerce")


def get_val(df, cell_id):
    """Extract a single cell value and gracefully handle empty/None."""
    if cell_id is None:
        return np.nan

    try:
        return extract_range(df, cell_id)[0]
    except Exception:
        return np.nan


# ============================================================
# 2. MAIN ANALYSIS AND PLOTTING
# ============================================================

def analyze_elisa():

    df_raw = pd.read_excel(
        "ELISA_data.xlsx",
        sheet_name="ELISA_data",
        header=None
    )

    concentrations = [0.01, 0.1, 1, 5, 10, 20]

    # ========================================================
    # DATA MAPPING
    # ========================================================

    MAPPING = {
        "C-": {
            "type": "control",
            "rep1": "G36",
            "rep2": "H36"
        },
        "C+": {
            "type": "control",
            "rep1": "G37",
            "rep2": "H37"
        },
        "KIT C": {
            "type": "compound",
            "rep1": "G38:G43",
            "rep2": "H38:H43"
        },
        "X24475": {
            "type": "compound",
            "rep1": "M38:M43",
            "rep2": "N38:N43"
        },
        "X24437": {
            "type": "compound",
            "rep1": "I38:I43",
            "rep2": "J38:J43"
        },
        "X24444": {
            "type": "compound",
            "rep1": "K38:K43",
            "rep2": "L38:L43"
        },
        "X24442": {
            "type": "compound",
            "rep1": "I36:N36",
            "rep2": "I37:N37"
        },
        "X24435": {
            "type": "compound",
            "rep1": "D38:D43",
            "rep2": "E38:E43"
        }
    }

    NEW_MAPPING = {
        "C-": {
            "rep3": "G71",
            "rep4": "L76"
        },
        "C+": {
            "rep3": "G72",
            "rep4": "L77"
        },
        "KIT C": {
            "rep3": ["G73", "G74", "G76", "G77", "H71", "H73"],
            "rep4": [None, "G75", None, "G78", "H72", None]
        },
        "X24475": {
            "rep3": ["H74", "H75", "H76", "H77", "H78", "I71"],
            "rep4": [None, None, None, None, None, None]
        },
        "X24437": {
            "rep3": ["I72", "I74", "I75", "I77", "I78", "J71"],
            "rep4": ["I73", None, "I76", None, None, None]
        },
        "X24444": {
            "rep3": ["J72", "J74", "J75", "J77", "J78", "K71"],
            "rep4": ["J73", None, "J76", None, None, None]
        },
        "X24442": {
            "rep3": ["K72", "K73", "K74", "K75", "K76", "K77"],
            "rep4": [None, None, None, None, None, None]
        },
        "X24435": {
            "rep3": ["K78", "L71", "L72", "L73", "L74", "L75"],
            "rep4": [None, None, None, None, None, None]
        }
    }

    plot_order = [
        "C-", "C+", "KIT C", "X24475",
        "X24437", "X24444", "X24442", "X24435"
    ]

    COLORS = {
        "C-": "white",
        "C+": "black",
        "KIT C": "#E0E0E0",
        "X24475": "#C0C0C0",
        "X24437": "#A0A0A0",
        "X24444": "#808080",
        "X24442": "#606060",
        "X24435": "#404040"
    }

    COMPOUND_IMAGES = {
        "KIT C": "KITC.png",
        "X24475": "X24475.png",
        "X24437": "X24437.png",
        "X24444": "X24444.png",
        "X24442": "X24442.png",
        "X24435": "X24435.png"
    }

    compound_imgs = {
        comp: mpimg.imread(filename)
        for comp, filename in COMPOUND_IMAGES.items()
    }

    # ========================================================
    # 3. DATA EXTRACTION
    # ========================================================

    data = []

    for comp in plot_order:
        info = MAPPING[comp]
        new_info = NEW_MAPPING[comp]

        r1_vals = extract_range(df_raw, info["rep1"])
        r2_vals = extract_range(df_raw, info["rep2"])

        if info["type"] == "control":
            r3_val = get_val(df_raw, new_info["rep3"])
            r4_val = get_val(df_raw, new_info["rep4"])

            data.append({
                "Compound": comp,
                "Conc": "N/A",
                "Rep1": r1_vals[0],
                "Rep2": r2_vals[0],
                "Rep3": r3_val,
                "Rep4": r4_val
            })

        else:
            for i, conc in enumerate(concentrations):
                r3_val = get_val(df_raw, new_info["rep3"][i])
                r4_val = get_val(df_raw, new_info["rep4"][i])

                data.append({
                    "Compound": comp,
                    "Conc": conc,
                    "Rep1": r1_vals[i],
                    "Rep2": r2_vals[i],
                    "Rep3": r3_val,
                    "Rep4": r4_val
                })

    df = pd.DataFrame(data)

    # ========================================================
    # 4. C+ NORMALIZATION REFERENCES
    # ========================================================

    c_plus_mean = (
        df[df["Compound"] == "C+"][["Rep1", "Rep2"]]
        .mean(axis=1)
        .values[0]
    )

    # X24435-specific C+
    x24435_c_plus_rep1 = get_val(df_raw, "D37")
    x24435_c_plus_rep2 = get_val(df_raw, "E37")
    x24435_c_plus_mean = np.nanmean([
        x24435_c_plus_rep1, x24435_c_plus_rep2
    ])

    # New experiments: Rep3 + Rep4
    c_plus_rep3 = df[df["Compound"] == "C+"]["Rep3"].values[0]
    c_plus_rep4 = df[df["Compound"] == "C+"]["Rep4"].values[0]
    c_plus_new_mean = np.nanmean([c_plus_rep3, c_plus_rep4])

    # ========================================================
    # 5. NORMALIZE ALL EXPERIMENTAL DATA TO C+
    # ========================================================

    df["Norm_Rep1"] = np.where(
        df["Compound"] == "X24435",
        df["Rep1"] / x24435_c_plus_mean,
        df["Rep1"] / c_plus_mean
    )

    df["Norm_Rep2"] = np.where(
        df["Compound"] == "X24435",
        df["Rep2"] / x24435_c_plus_mean,
        df["Rep2"] / c_plus_mean
    )

    df["Norm_Rep3"] = df["Rep3"] / c_plus_new_mean
    df["Norm_Rep4"] = df["Rep4"] / c_plus_new_mean

    # ========================================================
    # 6. C- REFERENCES FOR EMAX
    # ========================================================

    c_minus_old_rep1 = get_val(df_raw, "G36")
    c_minus_old_rep2 = get_val(df_raw, "H36")
    c_minus_old_mean = np.nanmean([
        c_minus_old_rep1, c_minus_old_rep2
    ])
    c_minus_old_norm = c_minus_old_mean / c_plus_mean

    # X24435-specific C-
    x24435_c_minus_rep1 = get_val(df_raw, "D36")
    x24435_c_minus_rep2 = get_val(df_raw, "E36")
    x24435_c_minus_mean = np.nanmean([
        x24435_c_minus_rep1, x24435_c_minus_rep2
    ])
    x24435_c_minus_norm = x24435_c_minus_mean / x24435_c_plus_mean

    # New C-
    c_minus_new_rep3 = get_val(df_raw, "G71")
    c_minus_new_rep4 = get_val(df_raw, "L76")
    c_minus_new_mean = np.nanmean([
        c_minus_new_rep3, c_minus_new_rep4
    ])
    c_minus_new_norm = c_minus_new_mean / c_plus_new_mean

    # ========================================================
    # 7. COMPOUND-SPECIFIC C- REFERENCE
    # ========================================================

    compound_c_minus = {
        "KIT C": [c_minus_old_norm, c_minus_new_norm],
        "X24475": [c_minus_old_norm, c_minus_new_norm],
        "X24437": [c_minus_old_norm, c_minus_new_norm],
        "X24444": [c_minus_old_norm, c_minus_new_norm],
        "X24442": [c_minus_old_norm, c_minus_new_norm],
        "X24435": [x24435_c_minus_norm, c_minus_new_norm]
    }

    # ========================================================
    # 8. MEAN AND SEM
    # ========================================================

    norm_cols = [
        "Norm_Rep1", "Norm_Rep2",
        "Norm_Rep3", "Norm_Rep4"
    ]

    df["Mean"] = df[norm_cols].mean(axis=1)
    df["SEM"] = df[norm_cols].sem(axis=1)

    # ========================================================
    # 9. STATISTICAL ANALYSIS: DUNNETT'S TEST
    # ========================================================

    df["p_val"] = np.nan

    control_vals = (
        df[df["Compound"] == "C+"][norm_cols]
        .values.flatten()
    )
    control_vals = control_vals[~pd.isna(control_vals)]

    treatment_arrays = []
    treatment_indices = []

    for idx, row in df.iterrows():
        if row["Compound"] == "C+":
            continue

        vals = np.array([
            row["Norm_Rep1"], row["Norm_Rep2"],
            row["Norm_Rep3"], row["Norm_Rep4"]
        ])
        vals = vals[~pd.isna(vals)]

        if len(vals) >= 2:
            treatment_arrays.append(vals)
            treatment_indices.append(idx)

    if hasattr(stats, "dunnett"):
        try:
            res = stats.dunnett(
                *treatment_arrays,
                control=control_vals,
                alternative="two-sided"
            )

            for i, p in zip(treatment_indices, res.pvalue):
                df.at[i, "p_val"] = p

        except Exception as e:
            print(f"Dunnett's test calculation failed: {e}")

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

    df["sig"] = df["p_val"].apply(get_sig_label)

    # ========================================================
    # 10. CREATE ONE FIGURE WITH THREE PANELS
    # ========================================================

    sns.set_style("white")

    fig = plt.figure(figsize=(22, 19))

    # A spans the full width.
    # B and C share the bottom row.
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1.15, 1],
        hspace=0.78
    )

    ax1 = fig.add_subplot(gs[0, 0])

    gs_bottom = gs[1, 0].subgridspec(
        1, 2,
        width_ratios=[1, 1],
        wspace=0.30
    )

    ax2 = fig.add_subplot(gs_bottom[0, 0])
    ax3 = fig.add_subplot(gs_bottom[0, 1])

    # Panel labels
    for ax, label in [(ax1, "A"), (ax2, "B"), (ax3, "C")]:
        ax.text(
            -0.055, 1.025, label,
            transform=ax.transAxes,
            fontsize=FONT["panel_label"],
            fontweight="bold",
            ha="right",
            va="bottom"
        )

    # ========================================================
    # 11. PANEL A: BAR CHART
    # ========================================================

    current_x = 0
    bar_width = 1.2
    bar_step = 1.7
    group_gap = 2.5

    all_bar_x = []
    all_comps = []
    all_concs = []
    group_centers = []
    group_labels = []

    max_y_global = (df["Mean"] + df["SEM"]).max()
    sig_data = []

    for comp in plot_order:
        comp_data = df[df["Compound"] == comp]
        start_x = current_x

        for idx, row in comp_data.iterrows():
            ax1.bar(
                current_x,
                row["Mean"],
                width=bar_width,
                yerr=row["SEM"],
                color=COLORS[comp],
                edgecolor="black",
                linewidth=1.5,
                capsize=5
            )

            all_bar_x.append(current_x)
            all_comps.append(comp)
            all_concs.append(row["Conc"])

            if MAPPING[comp]["type"] != "control" or comp == "C-":
                sig_data.append({
                    "pos": current_x,
                    "sig": row["sig"],
                    "comp": comp,
                    "y_val": row["Mean"] + row["SEM"]
                })

            current_x += bar_step

        group_centers.append((start_x + current_x - bar_step) / 2)
        group_labels.append(comp)
        current_x += group_gap

    # Group significance annotations
    current_group = []
    groups = []

    for item in sig_data:
        if not current_group:
            current_group.append(item)
        else:
            if (
                item["comp"] == current_group[-1]["comp"]
                and item["sig"] == current_group[-1]["sig"]
            ):
                current_group.append(item)
            else:
                groups.append(current_group)
                current_group = [item]

    if current_group:
        groups.append(current_group)

    for group in groups:
        if group[0]["sig"] == "ns":
            continue

        start_x = group[0]["pos"]
        end_x = group[-1]["pos"]
        mid_x = (start_x + end_x) / 2
        local_max_y = max(item["y_val"] for item in group)

        if len(group) > 1:
            sig_height = local_max_y + max_y_global * 0.05

            ax1.plot(
                [start_x, end_x],
                [sig_height, sig_height],
                color="black",
                lw=1.5
            )

            ax1.text(
                mid_x,
                sig_height + max_y_global * 0.01,
                group[0]["sig"],
                ha="center",
                va="bottom",
                fontsize=FONT["significance"],
                fontweight="bold"
            )

        else:
            y_pos = local_max_y + max_y_global * 0.03

            ax1.text(
                mid_x, y_pos, group[0]["sig"],
                ha="center",
                va="bottom",
                fontsize=FONT["significance"],
                fontweight="bold"
            )

    # Custom x-axis labels
    trans = ax1.get_xaxis_transform()

    row_names = ["1 ng/µL IL-1β", "Conc. (µM)"]
    x_label_pos = min(all_bar_x) - 1.8
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
        il1b_val = "-" if bcomp == "C-" else "+"
        conc_val = str(bconc) if bconc != "N/A" else "-"

        ax1.text(
            bx, y_start, il1b_val,
            transform=trans,
            ha="center",
            va="center",
            fontsize=FONT["concentration_label"],
            clip_on=False
        )

        ax1.text(
            bx, y_start - y_step, conc_val,
            transform=trans,
            ha="center",
            va="center",
            fontsize=FONT["concentration_label"],
            fontstretch="condensed",
            clip_on=False
        )

    for x_pos, label in zip(group_centers, group_labels):
        ax1.text(
            x_pos, y_start - (2 * y_step) - 0.02, label,
            transform=trans,
            ha="center",
            va="top",
            fontsize=FONT["compound_group_label"],
            fontweight="bold",
            clip_on=False
        )

        if label in compound_imgs:
            image = OffsetImage(compound_imgs[label], zoom=0.1)

            annotation = AnnotationBbox(
                image,
                (x_pos + 0.5, -0.45),
                xycoords=("data", "axes fraction"),
                frameon=False,
                box_alignment=(0.5, 0.5)
            )

            ax1.add_artist(annotation)

    ax1.set_ylabel(
        "PGE2 Level\n(time-fold of C+)",
        fontsize=FONT["axis_label"],
        fontweight="bold"
    )

    ax1.set_title(
        "PGE2 Release",
        fontsize=FONT["panel_title"],
        fontweight="bold",
        pad=20
    )

    ax1.set_xticks(all_bar_x)
    ax1.set_xticklabels([])
    ax1.set_ylim(0, max_y_global * 1.25)
    ax1.set_xlim(min(all_bar_x) - 4.5, max(all_bar_x) + 1.5)

    ax1.tick_params(
        axis="both",
        labelsize=FONT["tick_label"]
    )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.5)
    ax1.spines["bottom"].set_linewidth(1.5)

    # ========================================================
    # 12. PANEL B: DOSE-RESPONSE CURVES
    # ========================================================

    print("\n" + "=" * 90)
    print(
        f"{'Compound':<15} | {'IC50 (µM)':<12} | "
        f"{'Bottom':<10} | {'C- norm.':<10} | {'Emax (%)':<10}"
    )
    print("=" * 90)

    compounds_to_fit = [
        "KIT C", "X24475", "X24437",
        "X24444", "X24442", "X24435"
    ]

    for comp in compounds_to_fit:
        comp_df = df[df["Compound"] == comp]

        x_vals_for_fit = []
        y_vals_for_fit = []
        x_vals_for_plot = []
        y_vals_for_plot = []

        # Collect individual replicate values for regression
        for _, row in comp_df.iterrows():
            for rep_col in [
                "Norm_Rep1", "Norm_Rep2",
                "Norm_Rep3", "Norm_Rep4"
            ]:
                if not pd.isna(row[rep_col]):
                    x_vals_for_fit.append(float(row["Conc"]))
                    y_vals_for_fit.append(float(row[rep_col]))

            # Mean values used for plotting
            if not pd.isna(row["Mean"]) and row["Conc"] != "N/A":
                x_vals_for_plot.append(float(row["Conc"]))
                y_vals_for_plot.append(float(row["Mean"]))

        x_vals_for_fit = np.array(x_vals_for_fit)
        y_vals_for_fit = np.array(y_vals_for_fit)

        if len(x_vals_for_fit) > 3:

            # 4PL model with Top fixed at 1.0
            def model_4pl_fixed_top(x_log, bottom, log_ic50, hill_slope):
                top = 1.0

                return (
                    bottom
                    + (top - bottom)
                    / (1 + 10 ** ((x_log - log_ic50) * hill_slope))
                )

            x_log = np.log10(x_vals_for_fit)

            p0 = [
                np.min(y_vals_for_fit),
                np.median(x_log),
                1.0
            ]

            bounds = (
                [-0.2, -4.0, 0.1],
                [0.8, 4.0, 10.0]
            )

            ax2.scatter(
                x_vals_for_plot,
                y_vals_for_plot,
                facecolor=COLORS[comp],
                edgecolor="black",
                s=50,
                zorder=3,
                alpha=0.9
            )

            try:
                popt, _ = curve_fit(
                    model_4pl_fixed_top,
                    x_log,
                    y_vals_for_fit,
                    p0=p0,
                    bounds=bounds,
                    maxfev=10000
                )

                bottom_asymptote = popt[0]
                log_ic50 = popt[1]
                hill_slope = popt[2]
                ic50 = 10 ** log_ic50

                # Determine compound-specific C- baseline
                c_minus_values = np.array(
                    compound_c_minus[comp],
                    dtype=float
                )
                c_minus_norm = np.nanmean(c_minus_values)

                # Emax relative to C+ -> C-
                max_inhib = (
                    (1.0 - bottom_asymptote)
                    / (1.0 - c_minus_norm)
                ) * 100

                # Clamp only for display
                max_inhib = max(0, min(100, max_inhib))

                # Smooth fitted curve
                x_fit_log = np.linspace(
                    np.log10(0.001),
                    np.log10(max(concentrations)),
                    200
                )

                y_fit = model_4pl_fixed_top(x_fit_log, *popt)
                x_fit = 10 ** x_fit_log

                if ic50 > max(concentrations):
                    print(
                        f"{comp:<15} | > {max(concentrations):<9} | "
                        f"{bottom_asymptote:<10.3f} | "
                        f"{c_minus_norm:<10.3f} | {max_inhib:>7.1f}"
                    )

                    curve_label = (
                        f"{comp} (IC50 > {max(concentrations)} µM, "
                        f"Emax = {max_inhib:.0f}%)"
                    )

                else:
                    print(
                        f"{comp:<15} | {ic50:<10.3f} | "
                        f"{bottom_asymptote:<10.3f} | "
                        f"{c_minus_norm:<10.3f} | {max_inhib:>7.1f}"
                    )

                    curve_label = (
                        f"{comp} (IC50 = {ic50:.2f} µM, "
                        f"Emax = {max_inhib:.0f}%)"
                    )

                ax2.plot(
                    x_fit, y_fit,
                    color=COLORS[comp],
                    lw=2.5,
                    label=curve_label
                )

            except Exception as e:
                print(f"{comp:<15} | Fit Failed | N/A")
                print(f"    Reason: {e}")

                ax2.plot(
                    [],
                    [],
                    color=COLORS[comp],
                    lw=2.5,
                    label=f"{comp} (Fit Failed)"
                )

    # ========================================================
    # 13. PANEL B FORMATTING
    # ========================================================

    ax2.set_xscale("log")

    ax2.set_xlabel(
        "Conc. (µM)",
        fontsize=FONT["axis_label"],
        fontweight="bold"
    )

    ax2.set_ylabel(
        "PGE2 Level\n(time-fold of C+)",
        fontsize=FONT["axis_label"],
        fontweight="bold"
    )

    ax2.set_title(
        "Dose-Response Curves",
        fontsize=FONT["panel_title"],
        fontweight="bold",
        pad=20
    )

    # 50% inhibition reference level
    c_minus_plot = np.nanmean([
        c_minus_old_norm, c_minus_new_norm
    ])

    fifty_percent_response = (1.0 + c_minus_plot) / 2

    ax2.axhline(
        fifty_percent_response,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        zorder=1
    )

    # C+ reference line
    ax2.axhline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.2,
        alpha=0.5,
        zorder=1
    )

    # C- reference line
    ax2.axhline(
        c_minus_plot,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        alpha=0.5,
        zorder=1
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(1.5)
    ax2.spines["bottom"].set_linewidth(1.5)

    ax2.tick_params(
        axis="both",
        labelsize=FONT["tick_label"]
    )

    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=False,
        fontsize=FONT["legend"],
        ncol=2
    )

    # ========================================================
    # 13B. PANEL C: C3 CHAIN-LENGTH REGRESSION
    # ========================================================

    # Signigicance stars
    def significance_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    # Compound properties
    regression_df = pd.DataFrame({
        "Compound": [
            "X24437", "KIT C", "X24475",
            "X24435", "X24444", "X24442"
        ],
        "Group_8": [
            "8-iso-Propyl", "8-iso-Propyl", "8-iso-Propyl",
            "8-tert-Butyl", "8-tert-Butyl", "8-tert-Butyl"
        ],
        "C3_length": [0, 1, 3, 0, 1, 3],
        "IC50": [17.76, 2.41, 0.03, 3.83, 13.72, 0.39]
    })

    # Natural logarithm is used for the regression. The plotted y-axis
    # remains IC50 in µM; fitted log(IC50) values are exponentiated below.
    regression_df["ln_IC50"] = np.log(regression_df["IC50"])

    # Grayscale styling consistent with the ELISA pipeline
    GROUP_STYLES = {
        "8-iso-Propyl": {
            "color": "#C0C0C0",
            "marker": "o",
            "linestyle": "--"
        },
        "8-tert-Butyl": {
            "color": "#606060",
            "marker": "s",
            "linestyle": ":"
        }
    }

    overall_color = "#222222"

    # Fit the three linear regressions on ln(IC50).
    # A negative slope means IC50 decreases as C3 chain length increases,
    # i.e. potency increases.
    res_all = stats.linregress(
        regression_df["C3_length"],
        regression_df["ln_IC50"]
    )

    group_results = {}

    for group in GROUP_STYLES:
        subset = regression_df[
            regression_df["Group_8"] == group
        ]

        group_results[group] = stats.linregress(
            subset["C3_length"],
            subset["ln_IC50"]
        )

    # Center C3 so the ANCOVA group coefficient compares residues at the
    # mean C3 chain length (rather than at C3 = 0).
    mean_c3 = regression_df["C3_length"].mean()
    regression_df["C3_centered"] = regression_df["C3_length"] - mean_c3

    ancova_model = smf.ols(
        "ln_IC50 ~ C3_centered * C(Group_8)",
        data=regression_df
    ).fit()

    # Smooth x domain
    x_smooth = np.linspace(0, 3, 250)

    # Overall fit, back-transformed from ln(IC50) to IC50.
    y_all_smooth = np.exp(
        res_all.intercept + res_all.slope * x_smooth
    )

    ax3.plot(
        x_smooth,
        y_all_smooth,
        color=overall_color,
        linestyle="-",
        linewidth=2.5,
        zorder=2,
        label=(
            "Overall fit\n"
            f"slope = {res_all.slope:.2f} ln(IC50)/C, \n"
            f"R² = {res_all.rvalue**2:.2f}, "
            f"p = {res_all.pvalue:.2g}"
        )
    )

    # Group-specific regression fits
    for group, result in group_results.items():
        style = GROUP_STYLES[group]

        y_smooth = np.exp(
            result.intercept + result.slope * x_smooth
        )

        ax3.plot(
            x_smooth,
            y_smooth,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            zorder=2,
            label=(
                f"{group} fit\n"
                f"slope = {result.slope:.2f} ln(IC50)/C, \n"
                f"R² = {result.rvalue**2:.2f}, "
                f"p = {result.pvalue:.2g}"
            )
        )

    # Plot individual compounds using the same colors as panels A and B
    for _, row in regression_df.iterrows():
        group = row["Group_8"]
        marker = GROUP_STYLES[group]["marker"]

        ax3.scatter(
            row["C3_length"],
            row["IC50"],
            color=COLORS[row["Compound"]],
            marker=marker,
            s=85,
            zorder=5,
            edgecolor="black",
            linewidth=1.0
        )

    # Compound labels and IC50 values
    text_positions = {
        "X24437": (0, -23),
        "X24435": (0, 12),
        "X24444": (0, 12),
        "KIT C": (0, -23),
        "X24442": (0, 12),
        "X24475": (0, -23)
    }

    for _, row in regression_df.iterrows():
        ax3.annotate(
            f"{row['Compound']}\n({row['IC50']:g} µM)",
            xy=(row["C3_length"], row["IC50"]),
            xytext=text_positions[row["Compound"]],
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=FONT["compound_annotation"],
            fontweight="bold",
            color=COLORS[row["Compound"]],
            annotation_clip=False
        )

    # Axis formatting
    ax3.set_yscale("log")

    ax3.set_xlabel(
        "C3 Chain Length (Number of Carbons)",
        fontsize=FONT["axis_label"],
        fontweight="bold",
        labelpad=10
    )

    ax3.set_ylabel(
        "IC50 (µM)",
        fontsize=FONT["axis_label"],
        fontweight="bold",
        labelpad=8
    )

    ax3.set_title(
        "C3 Chain Length vs. Inhibition Potency",
        fontsize=FONT["panel_title"],
        fontweight="bold",
        pad=18
    )

    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xlim(-0.45, 3.45)
    ax3.set_ylim(0.01, 50)

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_linewidth(1.5)
    ax3.spines["bottom"].set_linewidth(1.5)

    ax3.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.7,
        alpha=0.5
    )

    ax3.grid(
        True,
        which="minor",
        axis="y",
        linestyle=":",
        linewidth=0.5,
        alpha=0.25
    )

    ax3.tick_params(
        axis="both",
        labelsize=FONT["tick_label"]
    )

    # Legend: compact and inside the plot
    handles, labels = ax3.get_legend_handles_labels()

    ax3.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=FONT["legend"],
        borderpad=0.7,
        labelspacing=0.7,
        handlelength=2.5,
        handletextpad=0.5,
        columnspacing=2.5
    )

    # ========================================================
    # PRINT REGRESSION STATISTICS + 95% CONFIDENCE INTERVALS
    # ========================================================

    print("\n" + "=" * 100)
    print("C3 CHAIN-LENGTH REGRESSION ANALYSIS")
    print("Response: ln(IC50); negative slopes indicate increasing potency")
    print("95% confidence intervals use the residual degrees of freedom")
    print("=" * 100)

    regression_sets = [("Overall", res_all, regression_df)]
    for group, result in group_results.items():
        subset = regression_df[regression_df["Group_8"] == group]
        regression_sets.append((group, result, subset))

    for name, result, subset in regression_sets:
        n = len(subset)
        df_resid = n - 2
        t_crit = stats.t.ppf(0.975, df_resid)

        slope_ci = (
            result.slope - t_crit * result.stderr,
            result.slope + t_crit * result.stderr
        )
        intercept_ci = (
            result.intercept - t_crit * result.intercept_stderr,
            result.intercept + t_crit * result.intercept_stderr
        )
        fold_change_per_carbon = np.exp(result.slope)
        fold_ci = np.exp(slope_ci)

        print(f"\n{name} (n = {n}, residual df = {df_resid})")
        print(f"  Intercept (ln IC50): {result.intercept:.4f}")
        print(
            f"    95% CI: [{intercept_ci[0]:.4f}, "
            f"{intercept_ci[1]:.4f}]"
        )
        print(f"  Slope (ln IC50 per C3 carbon): {result.slope:.4f}")
        print(
            f"    95% CI: [{slope_ci[0]:.4f}, "
            f"{slope_ci[1]:.4f}]"
        )
        print(
            f"  IC50 fold-change per additional C3 carbon: "
            f"{fold_change_per_carbon:.4f}"
        )
        print(
            f"    95% CI: [{fold_ci[0]:.4f}, {fold_ci[1]:.4f}]"
        )
        print(f"  R² = {result.rvalue**2:.4f}")
        print(f"  p = {result.pvalue:.5g}")

    print("\n" + "=" * 100)
    print("ANCOVA: C3 chain length, C8 residue, and their interaction")
    print("Response: ln(IC50)")
    print(f"C3 centered at its mean: {mean_c3:.4f}")
    print("=" * 100)

    ancova_ci = ancova_model.conf_int()
    ancova_ci.columns = ["CI_lower", "CI_upper"]
    ancova_table = pd.DataFrame({
        "Estimate": ancova_model.params,
        "CI_lower": ancova_ci["CI_lower"],
        "CI_upper": ancova_ci["CI_upper"],
        "p_value": ancova_model.pvalues
    })
    print(ancova_table.to_string(float_format=lambda x: f"{x:.5g}"))

    print("\nANCOVA model fit:")
    print(f"  R² = {ancova_model.rsquared:.4f}")
    print(f"  Adjusted R² = {ancova_model.rsquared_adj:.4f}")
    print(f"  F({int(ancova_model.df_model)}, {int(ancova_model.df_resid)}) = "
          f"{ancova_model.fvalue:.4f}")
    print(f"  Overall model p = {ancova_model.f_pvalue:.5g}")
    print(f"  Residual df = {int(ancova_model.df_resid)}")

    # Convert the centered C8 group coefficient into a ratio of geometric
    # IC50 values at mean C3. With iso-propyl as the reference group,
    # exp(coefficient) is tert-butyl IC50 / iso-propyl IC50.
    group_term = "C(Group_8)[T.8-tert-Butyl]"
    if group_term in ancova_model.params.index:
        group_beta = ancova_model.params[group_term]
        group_ci_low, group_ci_high = ancova_ci.loc[
            group_term, ["CI_lower", "CI_upper"]
        ]
        ratio = np.exp(group_beta)
        ratio_ci = (np.exp(group_ci_low), np.exp(group_ci_high))

        print("\nC8 residue comparison at mean C3 chain length:")
        print("  Ratio = IC50(tert-butyl) / IC50(iso-propyl)")
        print(f"  IC50 ratio = {ratio:.4f}")
        print(f"  95% CI = [{ratio_ci[0]:.4f}, {ratio_ci[1]:.4f}]")
        print(f"  p = {ancova_model.pvalues[group_term]:.5g}")
        print("  Ratio < 1 favors tert-butyl; ratio > 1 favors iso-propyl.")

    # Interaction is the difference between the two C3 slopes.
    interaction_terms = [
        term for term in ancova_model.params.index
        if ":" in term and "C3_centered" in term
    ]
    if interaction_terms:
        interaction_term = interaction_terms[0]
        interaction_beta = ancova_model.params[interaction_term]
        interaction_ci_low, interaction_ci_high = ancova_ci.loc[
            interaction_term, ["CI_lower", "CI_upper"]
        ]
        print("\nC3 slope difference (tert-butyl minus iso-propyl):")
        print(f"  Difference in slopes = {interaction_beta:.4f} ln(IC50)/C")
        print(
            f"  95% CI = [{interaction_ci_low:.4f}, "
            f"{interaction_ci_high:.4f}]"
        )
        print(
            f"  Ratio of per-carbon IC50 fold-changes = "
            f"{np.exp(interaction_beta):.4f}"
        )
        print(f"  p = {ancova_model.pvalues[interaction_term]:.5g}")

    print("\nNote: IC50 values are entered as fixed estimates; these CIs do not")
    print("propagate uncertainty from the original dose-response curve fits.")
    print("With only three compounds per residue group, subgroup regression CIs")
    print("are especially unstable (one residual degree of freedom per group).")
    print("=" * 100 + "\n")

    # ========================================================
    # 14. DATA EXPORT & PRINTING
    # ========================================================

    df["n_reps"] = df[
        ["Rep1", "Rep2", "Rep3", "Rep4"]
    ].count(axis=1)

    print_df = df[
        [
            "Compound", "Conc", "n_reps",
            "Rep1", "Rep2", "Rep3", "Rep4",
            "Norm_Rep1", "Norm_Rep2",
            "Norm_Rep3", "Norm_Rep4",
            "Mean", "SEM", "p_val", "sig"
        ]
    ].copy()

    print(print_df.to_string(index=False))
    print("=" * 90 + "\n")

    # ========================================================
    # 15. FINAL LAYOUT AND SAVE
    # ========================================================

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.96,
        bottom=0.15,
        hspace=0.78,
        wspace=0.30
    )

    fig.savefig(
        "ELISA_PGE2_combined.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 16. RUN ANALYSIS
# ============================================================

if __name__ == "__main__":
    analyze_elisa()
