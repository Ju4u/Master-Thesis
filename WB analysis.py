# WB analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.patches as mpatches


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

    # ============================================================
    # LOAD DATA
    # ============================================================

    df_full = pd.read_excel(excel_file, header=None)

    sample_names = list(target1_rows.keys())
    num_samples = len(sample_names)

    if pos_control not in sample_names:
        raise ValueError(
            f"Positive control '{pos_control}' must be in the sample dictionary."
        )

    # ============================================================
    # DATA PROCESSING AND STATISTICS
    # ============================================================

    def process_target(t_rows, l_rows):

        # --------------------------------------------------------
        # Extract target-protein densitometry
        # --------------------------------------------------------

        df = extract_data(
            df_full,
            t_rows
        ).reindex(sample_names)

        # --------------------------------------------------------
        # Extract and apply loading-control normalization
        # --------------------------------------------------------

        if l_rows:

            df_lc = extract_data(
                df_full,
                l_rows
            ).reindex(sample_names)

            # If no loading-control data are available,
            # retain the original target-protein data.
            if df_lc.isna().all().all():

                df_calibrated = df.copy()

            else:

                df_lc_norm = df_lc.copy()

                # Normalize each loading-control replicate
                # to the corresponding C+ loading-control value.
                for col in range(3):

                    c_plus_lc = df_lc.loc[pos_control].iloc[col]

                    if pd.isna(c_plus_lc) or c_plus_lc == 0:

                        df_lc_norm.iloc[:, col] = np.nan

                    else:

                        df_lc_norm.iloc[:, col] = (
                            df_lc.iloc[:, col] / c_plus_lc
                        )

                # Correct target-protein signal for relative
                # loading-control differences.
                df_calibrated = df * (1.0 / df_lc_norm)

        else:

            df_calibrated = df.copy()

        # --------------------------------------------------------
        # Normalize target/loading-control ratio to C+
        # --------------------------------------------------------

        norm_data = df_calibrated.copy()

        for col in range(3):

            c_plus_val = df_calibrated.loc[pos_control].iloc[col]

            if pd.isna(c_plus_val) or c_plus_val == 0:

                norm_data.iloc[:, col] = np.nan

            else:

                norm_data.iloc[:, col] = (
                    df_calibrated.iloc[:, col] / c_plus_val
                )

        # --------------------------------------------------------
        # Descriptive statistics
        # --------------------------------------------------------

        means = norm_data.mean(axis=1).values
        sems = norm_data.sem(axis=1).values

        # ========================================================
        # STATISTICAL ANALYSIS
        #
        # One-way ANOVA followed by Dunnett's multiple-comparison
        # test using C+ as the reference/control group.
        # ========================================================

        # Extract valid replicate values for each experimental group.
        groups = [
            norm_data.loc[name].dropna().values
            for name in sample_names
        ]

        # --------------------------------------------------------
        # One-way ANOVA
        # --------------------------------------------------------

        valid_groups = [
            group for group in groups
            if len(group) >= 2
        ]

        if len(valid_groups) >= 2:

            anova_stat, anova_p = stats.f_oneway(
                *valid_groups
            )

        else:

            anova_stat = np.nan
            anova_p = np.nan

        # --------------------------------------------------------
        # Dunnett's multiple-comparison test
        #
        # Every treatment group is compared with C+.
        # --------------------------------------------------------

        control_index = sample_names.index(pos_control)

        control_data = groups[control_index]

        treatment_groups = [
            groups[i]
            for i in range(len(groups))
            if i != control_index
        ]

        treatment_names = [
            sample_names[i]
            for i in range(len(sample_names))
            if i != control_index
        ]

        # Initialize p-values for all groups.
        p_vals = [None] * len(sample_names)

        if (
            len(control_data) >= 2
            and len(treatment_groups) > 0
        ):

            # Keep only treatment groups with sufficient
            # replicate numbers for statistical testing.
            valid_treatment_groups = []
            valid_treatment_names = []

            for group, name in zip(
                    treatment_groups,
                    treatment_names
            ):

                if len(group) >= 2:

                    valid_treatment_groups.append(group)
                    valid_treatment_names.append(name)

            if len(valid_treatment_groups) > 0:

                # Dunnett's test.
                #
                # scipy.stats.dunnett returns multiplicity-adjusted
                # p-values for each treatment vs. the control.
                dunnett_result = stats.dunnett(
                    *valid_treatment_groups,
                    control=control_data
                )

                # Store Dunnett-adjusted p-values according to the
                # original sample order.
                for name in valid_treatment_names:

                    idx = valid_treatment_names.index(name)

                    sample_index = sample_names.index(name)

                    p_vals[sample_index] = (
                        dunnett_result.pvalue[idx]
                    )

        # Print statistical results to the console.
        print("\n========================================")
        print(f"Target: {target_names[0]}")
        print("========================================")
        print(f"One-way ANOVA: F = {anova_stat:.4f}, "
              f"p = {anova_p:.4g}")

        print("\nDunnett's multiple-comparison test "
              f"(vs. {pos_control}):")

        for name, p in zip(sample_names, p_vals):

            if name == pos_control:
                continue

            if p is None:

                print(f"{name}: insufficient data")

            else:

                print(f"{name}: adjusted p = {p:.4g}")

        return (
            norm_data,
            means,
            sems,
            p_vals,
            anova_stat,
            anova_p
        )

    # ============================================================
    # PROCESS BOTH TARGET PROTEINS
    # ============================================================

    (
        norm_t1,
        means_t1,
        sems_t1,
        p_vals_t1,
        anova_t1,
        anova_p_t1
    ) = process_target(
        target1_rows,
        lc1_rows
    )

    (
        norm_t2,
        means_t2,
        sems_t2,
        p_vals_t2,
        anova_t2,
        anova_p_t2
    ) = process_target(
        target2_rows,
        lc2_rows
    )

    # ============================================================
    # FIGURE SETUP
    # ============================================================

    has_img = blot_image_path is not None

    ratios = [3.2]

    if has_img:

        ratios.append(0.2)
        ratios.append(1.8)

    fig = plt.figure(
        figsize=(6, 8)
    )

    gs = fig.add_gridspec(
        len(ratios),
        1,
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

    ax = fig.add_subplot(
        gs[0, 0]
    )

    ax.text(
        -0.23,
        1.05,
        'A',
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        va='bottom'
    )

    sns.set_style("white")

    # ============================================================
    # BAR POSITIONS
    # ============================================================

    width = 0.38
    separation = 0.22

    x_t1 = x_pos - separation
    x_t2 = x_pos + separation

    # ============================================================
    # COMPOUND COLORS
    # ============================================================

    compound_colors = {

        'KIT C': '#E0E0E0',
        'X24475': '#C0C0C0',
        'X24437': '#A0A0A0',
        'X24444': '#808080',
        'X24442': '#606060',
        'X24435': '#404040'

    }

    # COX-2:
    # controls = white
    # compounds = compound-specific gray

    c_t1 = [
        'white'
        if n in [pos_control, neg_control]
        else compound_colors[n]
        for n in sample_names
    ]

    # mPGES-1:
    # controls = black
    # compounds = compound-specific gray

    c_t2 = [
        'black'
        if n in [pos_control, neg_control]
        else compound_colors[n]
        for n in sample_names
    ]

    plot_sems_t1 = np.nan_to_num(
        sems_t1
    )

    plot_sems_t2 = np.nan_to_num(
        sems_t2
    )

    # ============================================================
    # PLOT BARS
    # ============================================================

    for i in range(num_samples):

        # COX-2
        ax.bar(
            x_t1[i],
            means_t1[i],
            width,
            yerr=plot_sems_t1[i],
            color=c_t1[i],
            edgecolor='black',
            linewidth=1.5,
            capsize=3
        )

        # mPGES-1
        ax.bar(
            x_t2[i],
            means_t2[i],
            width,
            yerr=plot_sems_t2[i],
            color=c_t2[i],
            edgecolor='black',
            linewidth=1.5,
            capsize=3,
            hatch='///'
        )

    # ============================================================
    # PLOT INDIVIDUAL REPLICATES
    # ============================================================

    offsets = [
        -0.08,
        0,
        0.08
    ]

    for rep in range(3):

        # --------------------------------------------------------
        # COX-2 replicate points
        # --------------------------------------------------------

        v_t1 = norm_t1.iloc[:, rep].values

        valid_t1 = ~np.isnan(
            v_t1
        )

        if valid_t1.any():

            ax.scatter(
                x_t1[valid_t1] + offsets[rep],
                v_t1[valid_t1],
                facecolor='black',
                s=30,
                zorder=3,
                alpha=0.8,
                edgecolors='none'
            )

        # --------------------------------------------------------
        # mPGES-1 replicate points
        # --------------------------------------------------------

        v_t2 = norm_t2.iloc[:, rep].values

        valid_t2 = ~np.isnan(
            v_t2
        )

        if valid_t2.any():

            ax.scatter(
                x_t2[valid_t2] + offsets[rep],
                v_t2[valid_t2],
                facecolor='black',
                s=30,
                zorder=3,
                alpha=0.8,
                edgecolors='white',
                linewidth=0.8
            )

    # ============================================================
    # CUSTOM LEGEND
    # ============================================================

    t1_patch = mpatches.Patch(
        facecolor='#E0E0E0',
        edgecolor='black',
        linewidth=1.5
    )

    t2_patch = mpatches.Patch(
        facecolor='#E0E0E0',
        edgecolor='black',
        linewidth=1.5,
        hatch='///'
    )

    by_label = {
        target_names[0]: t1_patch,
        target_names[1]: t2_patch
    }

    ax.legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
        fontsize=11,
        loc='upper left',
        handlelength=1,
        handleheight=1
    )

    # ============================================================
    # SIGNIFICANCE ASTERISKS
    # ============================================================

    max_y = max(
        np.nanmax(
            means_t1 + plot_sems_t1
        ),
        np.nanmax(
            means_t2 + plot_sems_t2
        )
    )

    def add_significance_asterisks(
            x_coords,
            means,
            sems,
            p_vals
    ):

        for i, p in enumerate(p_vals):

            # Do not annotate the positive control itself.
            if (
                sample_names[i] == pos_control
                or p is None
                or np.isnan(p)
            ):
                continue

            if p < 0.05:

                y_pos = (
                    means[i]
                    + sems[i]
                    + (max_y * 0.05)
                )

                if p < 0.001:

                    sig_text = '***'

                elif p < 0.01:

                    sig_text = '**'

                else:

                    sig_text = '*'

                ax.text(
                    x_coords[i],
                    y_pos,
                    sig_text,
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    fontweight='bold'
                )

    add_significance_asterisks(
        x_t1,
        means_t1,
        plot_sems_t1,
        p_vals_t1
    )

    add_significance_asterisks(
        x_t2,
        means_t2,
        plot_sems_t2,
        p_vals_t2
    )

    # ============================================================
    # SUMMARY BRACKET
    # ============================================================

    if num_samples > 2:

        line_start = (
            x_t1[2]
            - (width / 2)
        )

        line_end = (
            x_t2[-1]
            + (width / 2)
        )

        line_y = max_y * 1.10

        tick_drop = max_y * 0.03

        ax.plot(
            [line_start, line_end],
            [line_y, line_y],
            color='black',
            lw=1.5
        )

        ax.plot(
            [line_start, line_start],
            [line_y, line_y - tick_drop],
            color='black',
            lw=1.5
        )

        ax.plot(
            [line_end, line_end],
            [line_y, line_y - tick_drop],
            color='black',
            lw=1.5
        )

        ax.text(
            (line_start + line_end) / 2,
            line_y + (max_y * 0.02),
            'ns',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    # ============================================================
    # AXIS LABELS AND FORMATTING
    # ============================================================

    ax.set_ylabel(
        'COX-2 or mPGES-1 / α-Tubulin \n(time-fold of C+)',
        fontsize=14,
        fontweight='bold'
    )

    ax.set_title(
        'COX-2 & mPGES-1 Western Blot',
        fontsize=14,
        fontweight='bold',
        pad=15
    )

    ax.set_xlim(
        x_min,
        x_max
    )

    ax.set_xticks(
        x_pos
    )

    ax.set_xticklabels(
        sample_names,
        fontsize=12,
        rotation=45,
        ha='center'
    )

    ax.set_ylim(
        0,
        max_y * 1.25
        if max_y > 0
        else 1
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Reference line representing C+ = 1
    ax.axhline(
        y=1,
        color='gray',
        linestyle='--',
        linewidth=1,
        alpha=0.5
    )

    # ============================================================
    # WESTERN BLOT IMAGE
    # ============================================================

    if has_img:

        ax_img = fig.add_subplot(
            gs[2, 0],
            sharex=ax
        )

        ax_img.text(
            -0.23,
            1.05,
            'B',
            transform=ax_img.transAxes,
            fontsize=20,
            fontweight='bold',
            va='bottom'
        )

        try:

            img = mpimg.imread(
                blot_image_path
            )

            crop_left = 0
            crop_right = 0
            crop_top = 0
            crop_bottom = 0

            h_orig, w_orig = img.shape[:2]

            img = img[
                crop_top:h_orig - crop_bottom,
                crop_left:w_orig - crop_right
            ]

            img_start_position = -2.3
            img_end_position = 7.5

            h_new, w_new = img.shape[:2]

            ax_img.imshow(
                img,
                extent=[
                    img_start_position,
                    img_end_position,
                    0,
                    h_new
                ],
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

        ax_img.set_xlim(
            x_min,
            x_max
        )

        ax_img.axis('off')

    # ============================================================
    # SAVE AND DISPLAY
    # ============================================================

    plt.savefig(
        'WB_analysis.png',
        dpi=600,
        bbox_inches='tight'
    )

    plt.show()


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # COX-2 target rows
    # ------------------------------------------------------------

    t1_samples = {

        "C-": 3,
        "C+": 4,
        "KIT C": 5,
        "X24475": 6,
        "X24437": 7,
        "X24444": 8,
        "X24442": 9,
        "X24435": 10

    }

    # ------------------------------------------------------------
    # COX-2 α-Tubulin rows
    # ------------------------------------------------------------

    t1_loading = {

        "C-": 12,
        "C+": 13,
        "KIT C": 14,
        "X24475": 15,
        "X24437": 16,
        "X24444": 17,
        "X24442": 18,
        "X24435": 19

    }

    # ------------------------------------------------------------
    # mPGES-1 target rows
    # ------------------------------------------------------------

    t2_samples = {

        "C-": 21,
        "C+": 22,
        "KIT C": 23,
        "X24475": 24,
        "X24437": 25,
        "X24444": 26,
        "X24442": 27,
        "X24435": 28

    }

    # ------------------------------------------------------------
    # mPGES-1 α-Tubulin rows
    # ------------------------------------------------------------

    t2_loading = {

        "C-": 30,
        "C+": 31,
        "KIT C": 32,
        "X24475": 33,
        "X24437": 34,
        "X24444": 35,
        "X24442": 36,
        "X24435": 37

    }

    # ============================================================
    # RUN ANALYSIS
    # ============================================================

    analyze_western_blot(

        excel_file="WB_data_new.xlsx",

        target1_rows=t1_samples,
        lc1_rows=t1_loading,

        target2_rows=t2_samples,
        lc2_rows=t2_loading,

        target_names=(
            "COX-2",
            "mPGES-1"
        ),

        pos_control="C+",
        neg_control="C-",

        blot_image_path="bands.png",

        plot_left_boundary=0.2,
        plot_right_boundary=0.9,
        plot_top_boundary=0.9,
        plot_bottom_boundary=0.05

    )
