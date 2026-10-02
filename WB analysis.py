# WB analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.legend_handler import HandlerTuple


def extract_data(df_full, row_dict):
    sample_names = list(row_dict.keys())
    data_rows = []
    for name in sample_names:
        r = row_dict[name] - 1

        row_data = []
        for col_idx in range(1, 4):
            if r < df_full.shape[0] and col_idx < df_full.shape[1]:
                val = df_full.iloc[r, col_idx]
            else:
                val = np.nan
            row_data.append(val)

        row_data = pd.to_numeric(row_data, errors='coerce')
        data_rows.append(row_data)

    return pd.DataFrame(data_rows, index=sample_names)


def analyze_western_blot(
        excel_file,
        target1_rows, lc1_rows,
        target2_rows, lc2_rows,
        target_names=("COX-2", "mPGES-1"),
        pos_control='C+',
        neg_control='C-',
        blot_image_path=None,
        plot_left_boundary=0.10,
        plot_right_boundary=0.9,
        plot_top_boundary=0.9,
        plot_bottom_boundary=0.10
):
    df_full = pd.read_excel(excel_file, header=None)
    sample_names = list(target1_rows.keys())
    num_samples = len(sample_names)

    if pos_control not in sample_names:
        raise ValueError(f"Positive control '{pos_control}' must be in the sample dictionary.")

    def process_target(t_rows, l_rows):
        df = extract_data(df_full, t_rows).reindex(sample_names)

        if l_rows:
            df_lc = extract_data(df_full, l_rows).reindex(sample_names)
            if df_lc.isna().all().all():
                df_calibrated = df.copy()
            else:
                df_lc_norm = df_lc.copy()
                for col in range(3):
                    c_plus_lc = df_lc.loc[pos_control].iloc[col]
                    if pd.isna(c_plus_lc) or c_plus_lc == 0:
                        df_lc_norm.iloc[:, col] = np.nan
                    else:
                        df_lc_norm.iloc[:, col] = df_lc.iloc[:, col] / c_plus_lc
                df_calibrated = df * (1.0 / df_lc_norm)
        else:
            df_calibrated = df.copy()

        norm_data = df_calibrated.copy()
        for col in range(3):
            c_plus_val = df_calibrated.loc[pos_control].iloc[col]
            if pd.isna(c_plus_val) or c_plus_val == 0:
                norm_data.iloc[:, col] = np.nan
            else:
                norm_data.iloc[:, col] = df_calibrated.iloc[:, col] / c_plus_val

        means = norm_data.mean(axis=1).values
        sems = norm_data.sem(axis=1).values
        p_vals = []

        for name in sample_names:
            sample_vals = norm_data.loc[name].dropna().values
            if len(sample_vals) < 2:
                p_vals.append(None)
                continue

            _, p = stats.ttest_1samp(sample_vals, 1.0)
            p_vals.append(p)

        return norm_data, means, sems, p_vals

    norm_t1, means_t1, sems_t1, p_vals_t1 = process_target(target1_rows, lc1_rows)
    norm_t2, means_t2, sems_t2, p_vals_t2 = process_target(target2_rows, lc2_rows)

    has_img = blot_image_path is not None
    ratios = [3.2]

    if has_img:
        ratios.append(0.2)
        ratios.append(1.8)

    fig = plt.figure(figsize=(6, 8))

    gs = fig.add_gridspec(
        len(ratios), 1,
        height_ratios=ratios,
        hspace=0.1,
        left=plot_left_boundary,
        right=plot_right_boundary,
        top=plot_top_boundary,
        bottom=plot_bottom_boundary
    )

    x_min = -0.5
    x_max = num_samples - 0.5
    x_pos = np.arange(num_samples)

    ax = fig.add_subplot(gs[0, 0])
    ax.text(-0.23, 1.05, 'A', transform=ax.transAxes, fontsize=20, fontweight='bold', va='bottom')
    sns.set_style("white")

    width = 0.38
    seperation = 0.22
    x_t1 = x_pos - seperation
    x_t2 = x_pos + seperation

    # Color mapping for compound bar fills
    compound_colors = {
        'KIT C': '#E0E0E0',
        'X24475': '#C0C0C0',
        'X24437': '#A0A0A0',
        'X24444': '#808080',
        'X24442': '#606060',
        'X24435': '#404040'
    }

    # Bar fill colors: compound-specific colors for test compounds, white/black for controls
    c_t1 = ['white' if n in [pos_control, neg_control] else compound_colors[n] for n in sample_names]  # COX-2
    c_t2 = ['black' if n in [pos_control, neg_control] else compound_colors[n] for n in sample_names]  # mPGES-1

    plot_sems_t1 = np.nan_to_num(sems_t1)
    plot_sems_t2 = np.nan_to_num(sems_t2)

    # Plot bars individually with black outlines
    for i in range(num_samples):
        # COX-2 bars: solid fill with black outline
        ax.bar(x_t1[i], means_t1[i], width, yerr=plot_sems_t1[i], color=c_t1[i],
               edgecolor='black', linewidth=1.5, capsize=3)

        # mPGES-1 bars: hatched pattern with black outline
        ax.bar(x_t2[i], means_t2[i], width, yerr=plot_sems_t2[i], color=c_t2[i],
               edgecolor='black', linewidth=1.5, capsize=3, hatch='///')

    offsets = [-0.08, 0, 0.08]

    for rep in range(3):

        v_t1 = norm_t1.iloc[:, rep].values
        valid_t1 = ~np.isnan(v_t1)
        if valid_t1.any():
            # Plain black points on the white/light-gray bars
            ax.scatter(x_t1[valid_t1] + offsets[rep], v_t1[valid_t1], facecolor='black', s=30, zorder=3, alpha=0.8,
                       edgecolors='none')

        v_t2 = norm_t2.iloc[:, rep].values
        valid_t2 = ~np.isnan(v_t2)
        if valid_t2.any():
            # Black points with white borders so they show up on black/dark-gray bars
            ax.scatter(x_t2[valid_t2] + offsets[rep], v_t2[valid_t2], facecolor='black', s=30, zorder=3, alpha=0.8,
                       edgecolors='white', linewidth=0.8)

        # ===== CUSTOM LEGEND =====
        # COX-2: Small square box with white fill and black edge
        t1_patch = mpatches.Patch(facecolor='#E0E0E0', edgecolor='black', linewidth=1.5)

        # mPGES-1: Small square box with white fill, black edge, and hatch pattern
        t2_patch = mpatches.Patch(facecolor='#E0E0E0', edgecolor='black', linewidth=1.5, hatch='///')

        by_label = {target_names[0]: t1_patch, target_names[1]: t2_patch}

        # Render inside the plot on the top left with smaller, square markers
        ax.legend(by_label.values(), by_label.keys(),
                  frameon=False, fontsize=11, loc='upper left',
                  handlelength=1, handleheight=1)

        by_label = {target_names[0]: t1_patch, target_names[1]: t2_patch}

        # Render inside the plot on the top left with smaller, square markers
        ax.legend(by_label.values(), by_label.keys(),
                  frameon=False, fontsize=11, loc='upper left',
                  handlelength=1, handleheight=1)

        by_label = {target_names[0]: t1_patch, target_names[1]: t2_patch}

        # Render inside the plot on the top left with smaller, square markers
        ax.legend(by_label.values(), by_label.keys(),
                  frameon=False, fontsize=11, loc='upper left',
                  handlelength=1, handleheight=1)

    max_y = max(np.nanmax(means_t1 + plot_sems_t1), np.nanmax(means_t2 + plot_sems_t2))

    def add_significance_asterisks(x_coords, means, sems, p_vals):
        for i, p in enumerate(p_vals):
            if sample_names[i] == pos_control or p is None or np.isnan(p):
                continue
            if p < 0.05:
                y_pos = means[i] + sems[i] + (max_y * 0.05)
                sig_text = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                ax.text(x_coords[i], y_pos, sig_text, ha='center', va='bottom', fontsize=12, fontweight='bold')

    add_significance_asterisks(x_t1, means_t1, plot_sems_t1, p_vals_t1)
    add_significance_asterisks(x_t2, means_t2, plot_sems_t2, p_vals_t2)

    # Add summary spanning bracket for 'ns'
    if num_samples > 2:
        line_start = x_t1[2] - (width / 2)
        line_end = x_t2[-1] + (width / 2)
        line_y = max_y * 1.10
        tick_drop = max_y * 0.03

        ax.plot([line_start, line_end], [line_y, line_y], color='black', lw=1.5)
        ax.plot([line_start, line_start], [line_y, line_y - tick_drop], color='black', lw=1.5)
        ax.plot([line_end, line_end], [line_y, line_y - tick_drop], color='black', lw=1.5)
        ax.text((line_start + line_end) / 2, line_y + (max_y * 0.02), 'ns', ha='center', va='bottom', fontsize=12,
                fontweight='bold')

    ax.set_ylabel('COX-2 or mPGES-1 / α-Tubulin \n(time-fold of C+)', fontsize=14, fontweight='bold')
    ax.set_title('COX-2 & mPGES-1 Western Blot', fontsize=14, fontweight='bold', pad=15)

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sample_names, fontsize=12, rotation=45, ha='center')

    ax.set_ylim(0, max_y * 1.25 if max_y > 0 else 1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    if has_img:
        ax_img = fig.add_subplot(gs[2, 0], sharex=ax)
        ax_img.text(-0.23, 1.05, 'B', transform=ax_img.transAxes, fontsize=20, fontweight='bold', va='bottom')

        try:
            img = mpimg.imread(blot_image_path)

            crop_left = 0
            crop_right = 0
            crop_top = 0
            crop_bottom = 0

            h_orig, w_orig = img.shape[:2]
            img = img[crop_top: h_orig - crop_bottom, crop_left: w_orig - crop_right]

            img_start_position = -2.3
            img_end_position = 7.5

            h_new, w_new = img.shape[:2]

            ax_img.imshow(
                img,
                extent=[img_start_position, img_end_position, 0, h_new],
                aspect='auto',
                interpolation='bilinear',
                clip_on=False
            )
        except FileNotFoundError:
            ax_img.text(0.5, 0.5, 'Image file not found', ha='center', va='center', fontsize=12)

        ax_img.set_xlim(x_min, x_max)
        ax_img.axis('off')

    plt.savefig('WB_analysis.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    t1_samples = {
        "C-": 3, "C+": 4, "KIT C": 5, "X24475": 6,
        "X24437": 7, "X24444": 8, "X24442": 9, "X24435": 10,
    }

    t1_loading = {
        "C-": 12, "C+": 13, "KIT C": 14, "X24475": 15,
        "X24437": 16, "X24444": 17, "X24442": 18, "X24435": 19,
    }

    t2_samples = {
        "C-": 21, "C+": 22, "KIT C": 23, "X24475": 24,
        "X24437": 25, "X24444": 26, "X24442": 27, "X24435": 28,
    }

    t2_loading = {
        "C-": 30, "C+": 31, "KIT C": 32, "X24475": 33,
        "X24437": 34, "X24444": 35, "X24442": 36, "X24435": 37,
    }

    analyze_western_blot(
        excel_file="WB_data_new.xlsx",
        target1_rows=t1_samples, lc1_rows=t1_loading,
        target2_rows=t2_samples, lc2_rows=t2_loading,
        target_names=("COX-2", "mPGES-1"),
        pos_control="C+", neg_control="C-",
        blot_image_path="bands.png",
        plot_left_boundary=0.2,
        plot_right_boundary=0.9,
        plot_top_boundary=0.9,
        plot_bottom_boundary=0.05
    )