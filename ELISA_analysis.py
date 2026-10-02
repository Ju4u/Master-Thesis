import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
import re


# ==========================================
# 1. HELPER FUNCTIONS FOR EXCEL EXTRACTION
# ==========================================
def col2num(col_str):
    num = 0
    for c in col_str.upper():
        num = num * 26 + (ord(c) - ord('A')) + 1
    return num - 1


def extract_range(df, range_str):
    if ':' in range_str:
        start, end = range_str.split(':')
        start_col = col2num(re.findall(r'[A-Za-z]+', start)[0])
        start_row = int(re.findall(r'\d+', start)[0]) - 1
        end_col = col2num(re.findall(r'[A-Za-z]+', end)[0])
        end_row = int(re.findall(r'\d+', end)[0]) - 1

        vals = df.iloc[start_row:end_row + 1, start_col:end_col + 1].values.flatten()
    else:
        col = col2num(re.findall(r'[A-Za-z]+', range_str)[0])
        row = int(re.findall(r'\d+', range_str)[0]) - 1
        vals = [df.iloc[row, col]]

    cleaned_vals = [str(v).replace(',', '.') if isinstance(v, str) else v for v in vals]
    return pd.to_numeric(cleaned_vals, errors='coerce')


def get_val(df, cell_id):
    """Helper to extract a single cell value and gracefully handle empty/None."""
    if cell_id is None:
        return np.nan
    try:
        return extract_range(df, cell_id)[0]
    except Exception:
        return np.nan


# ==========================================
# 2. MAIN ANALYSIS AND PLOTTING
# ==========================================
def analyze_elisa():
    df_raw = pd.read_excel('ELISA_data.xlsx', sheet_name='ELISA_data', header=None)

    concentrations = [0.01, 0.1, 1, 5, 10, 20]

    MAPPING = {
        'C-': {'type': 'control', 'rep1': 'G36', 'rep2': 'H36'},
        'C+': {'type': 'control', 'rep1': 'G37', 'rep2': 'H37'},
        'KIT C': {'type': 'compound', 'rep1': 'G38:G43', 'rep2': 'H38:H43'},
        'X24475': {'type': 'compound', 'rep1': 'M38:M43', 'rep2': 'N38:N43'},
        'X24437': {'type': 'compound', 'rep1': 'I38:I43', 'rep2': 'J38:J43'},
        'X24444': {'type': 'compound', 'rep1': 'K38:K43', 'rep2': 'L38:L43'},
        'X24442': {'type': 'compound', 'rep1': 'I36:N36', 'rep2': 'I37:N37'},
        'X24435': {'type': 'compound', 'rep1': 'D38:D43', 'rep2': 'E38:E43'}
    }

    NEW_MAPPING = {
        'C-': {'rep3': 'G71', 'rep4': 'L76'},
        'C+': {'rep3': 'G72', 'rep4': 'L77'},
        'KIT C': {
            'rep3': ['G73', 'G74', 'G76', 'G77', 'H71', 'H73'],
            'rep4': [None, 'G75', None, 'G78', 'H72', None]
        },
        'X24475': {
            'rep3': ['H74', 'H75', 'H76', 'H77', 'H78', 'I71'],
            'rep4': [None, None, None, None, None, None]
        },
        'X24437': {
            'rep3': ['I72', 'I74', 'I75', 'I77', 'I78', 'J71'],
            'rep4': ['I73', None, 'I76', None, None, None]
        },
        'X24444': {
            'rep3': ['J72', 'J74', 'J75', 'J77', 'J78', 'K71'],
            'rep4': ['J73', None, 'J76', None, None, None]
        },
        'X24442': {
            'rep3': ['K72', 'K73', 'K74', 'K75', 'K76', 'K77'],
            'rep4': [None, None, None, None, None, None]
        },
        'X24435': {
            'rep3': ['K78', 'L71', 'L72', 'L73', 'L74', 'L75'],
            'rep4': [None, None, None, None, None, None]
        }
    }

    plot_order = ['C-', 'C+', 'KIT C', 'X24475', 'X24437', 'X24444', 'X24442', 'X24435']

    COLORS = {
        'C-': 'white', 'C+': 'black', 'KIT C': '#E0E0E0', 'X24475': '#C0C0C0',
        'X24437': '#A0A0A0', 'X24444': '#808080', 'X24442': '#606060', 'X24435': '#404040'
    }

    # --- DATA EXTRACTION ---
    data = []
    for comp in plot_order:
        info = MAPPING[comp]
        new_info = NEW_MAPPING[comp]

        r1_vals = extract_range(df_raw, info['rep1'])
        r2_vals = extract_range(df_raw, info['rep2'])

        if info['type'] == 'control':
            r3_val = get_val(df_raw, new_info['rep3'])
            r4_val = get_val(df_raw, new_info['rep4'])
            data.append({
                'Compound': comp, 'Conc': 'N/A',
                'Rep1': r1_vals[0], 'Rep2': r2_vals[0],
                'Rep3': r3_val, 'Rep4': r4_val
            })
        else:
            for i, conc in enumerate(concentrations):
                r3_val = get_val(df_raw, new_info['rep3'][i])
                r4_val = get_val(df_raw, new_info['rep4'][i])
                data.append({
                    'Compound': comp, 'Conc': conc,
                    'Rep1': r1_vals[i], 'Rep2': r2_vals[i],
                    'Rep3': r3_val, 'Rep4': r4_val
                })

    df = pd.DataFrame(data)

    # --- NORMALIZATION ---
    c_plus_mean = df[df['Compound'] == 'C+'][['Rep1', 'Rep2']].mean(axis=1).values[0]

    x24435_c_plus_rep1 = get_val(df_raw, 'D37')
    x24435_c_plus_rep2 = get_val(df_raw, 'E37')
    x24435_c_plus_mean = np.nanmean([x24435_c_plus_rep1, x24435_c_plus_rep2])

    df['Norm_Rep1'] = np.where(df['Compound'] == 'X24435',
                               df['Rep1'] / x24435_c_plus_mean,
                               df['Rep1'] / c_plus_mean)

    df['Norm_Rep2'] = np.where(df['Compound'] == 'X24435',
                               df['Rep2'] / x24435_c_plus_mean,
                               df['Rep2'] / c_plus_mean)

    c_plus_rep3 = df[df['Compound'] == 'C+']['Rep3'].values[0]
    c_plus_rep4 = df[df['Compound'] == 'C+']['Rep4'].values[0]
    c_plus_new_mean = np.nanmean([c_plus_rep3, c_plus_rep4])

    df['Norm_Rep3'] = df['Rep3'] / c_plus_new_mean
    df['Norm_Rep4'] = df['Rep4'] / c_plus_new_mean

    df['Mean'] = df[['Norm_Rep1', 'Norm_Rep2', 'Norm_Rep3', 'Norm_Rep4']].mean(axis=1)
    df['SEM'] = df[['Norm_Rep1', 'Norm_Rep2', 'Norm_Rep3', 'Norm_Rep4']].sem(axis=1)

    # --- STATISTICAL ANALYSIS (ANOVA with Dunnett's) ---
    df['p_val'] = np.nan

    control_vals = df[df['Compound'] == 'C+'][['Norm_Rep1', 'Norm_Rep2', 'Norm_Rep3', 'Norm_Rep4']].values.flatten()
    control_vals = control_vals[~pd.isna(control_vals)]

    treatment_arrays = []
    treatment_indices = []

    for idx, row in df.iterrows():
        if row['Compound'] == 'C+': continue
        vals = np.array([row['Norm_Rep1'], row['Norm_Rep2'], row['Norm_Rep3'], row['Norm_Rep4']])
        vals = vals[~pd.isna(vals)]
        if len(vals) >= 2:
            treatment_arrays.append(vals)
            treatment_indices.append(idx)

    if hasattr(stats, 'dunnett'):
        try:
            res = stats.dunnett(*treatment_arrays, control=control_vals, alternative='two-sided')
            for i, p in zip(treatment_indices, res.pvalue):
                df.at[i, 'p_val'] = p
        except Exception as e:
            print(f"Dunnett's test calculation failed: {e}")

    def get_sig_label(p):
        if pd.isna(p) or p >= 0.05:
            return 'ns'
        elif p < 0.0001:
            return '****'
        elif p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        else:
            return '*'

    df['sig'] = df['p_val'].apply(get_sig_label)

    # ==========================================
    # PLOT 1: BAR CHART
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(14, 5.8))
    sns.set_style("white")

    current_x = 0
    bar_width = 1.2
    bar_step = 1.7
    group_gap = 2.5

    all_bar_x = []
    all_comps = []
    all_concs = []
    group_centers = []
    group_labels = []

    max_y_global = (df['Mean'] + df['SEM']).max()
    sig_data = []

    for comp in plot_order:
        comp_data = df[df['Compound'] == comp]
        start_x = current_x

        for idx, row in comp_data.iterrows():
            ax1.bar(current_x, row['Mean'], width=bar_width, yerr=row['SEM'],
                    color=COLORS[comp], edgecolor='black', linewidth=1.5, capsize=5)

            all_bar_x.append(current_x)
            all_comps.append(comp)
            all_concs.append(row['Conc'])

            if MAPPING[comp]['type'] != 'control' or comp == 'C-':
                sig_data.append({'pos': current_x, 'sig': row['sig'], 'comp': comp, 'y_val': row['Mean'] + row['SEM']})

            current_x += bar_step

        group_centers.append((start_x + current_x - bar_step) / 2)
        group_labels.append(comp)
        current_x += group_gap

    current_group = []
    groups = []
    for i, item in enumerate(sig_data):
        if not current_group:
            current_group.append(item)
        else:
            if item['comp'] == current_group[-1]['comp'] and item['sig'] == current_group[-1]['sig']:
                current_group.append(item)
            else:
                groups.append(current_group)
                current_group = [item]
    if current_group: groups.append(current_group)

    for group in groups:
        if group[0]['sig'] == 'ns': continue
        start_x = group[0]['pos']
        end_x = group[-1]['pos']
        mid_x = (start_x + end_x) / 2
        local_max_y = max([item['y_val'] for item in group])

        if len(group) > 1:
            sig_height = local_max_y + (max_y_global * 0.05)
            ax1.plot([start_x, end_x], [sig_height, sig_height], color='black', lw=1.5)
            ax1.text(mid_x, sig_height + (max_y_global * 0.01), group[0]['sig'], ha='center', va='bottom', fontsize=13,
                     fontweight='bold')
        else:
            y_pos = local_max_y + (max_y_global * 0.03)
            ax1.text(mid_x, y_pos, group[0]['sig'], ha='center', va='bottom', fontsize=13, fontweight='bold')

    trans = ax1.get_xaxis_transform()
    row_names = ["1 ng/µL IL-1β", "Concentration (µM)"]
    x_label_pos = min(all_bar_x) - 1.8
    y_start = -0.05
    y_step = 0.06

    ax1.text(x_label_pos, y_start, row_names[0], transform=trans, ha='right', va='center', fontsize=12, clip_on=False)
    ax1.text(x_label_pos, y_start - y_step, row_names[1], transform=trans, ha='right', va='center', fontsize=12,
             clip_on=False)

    for bx, bcomp, bconc in zip(all_bar_x, all_comps, all_concs):
        il1b_val = "-" if bcomp == "C-" else "+"
        conc_val = str(bconc) if bconc != 'N/A' else "-"
        font_s = 10 if bconc == 0.01 else 12

        ax1.text(bx, y_start, il1b_val, transform=trans, ha='center', va='center', fontsize=12, clip_on=False)
        ax1.text(bx, y_start - y_step, conc_val, transform=trans, ha='center', va='center',
                 fontsize=font_s, fontstretch='condensed', clip_on=False)

    for x_pos, label in zip(group_centers, group_labels):
        ax1.text(x_pos, y_start - (2 * y_step) - 0.02, label, transform=trans,
                 ha='center', va='top', fontsize=14, fontweight='bold', clip_on=False)

    ax1.set_ylabel('PGE2 Level\n(time-fold of C+)', fontsize=14, fontweight='bold')
    ax1.set_title('PGE2 Release', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(all_bar_x)
    ax1.set_xticklabels([])
    ax1.set_ylim(0, max_y_global * 1.25)
    ax1.set_xlim(min(all_bar_x) - 4.5, max(all_bar_x) + 1.5)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)

    fig1.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.22)

    # ==========================================
    # PLOT 2: IC50 CURVES
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(8, 10))
    sns.set_style("white")

    print("\n" + "=" * 80)
    print(f"{'Compound':<15} | {'IC50 (µM)':<10} | {'Emax (Max Inhibition)':<20}")
    print("=" * 80)

    compounds_to_fit = ['KIT C', 'X24475', 'X24437', 'X24444', 'X24442', 'X24435']

    for comp in compounds_to_fit:
        comp_df = df[df['Compound'] == comp]

        x_vals_for_fit = []
        y_vals_for_fit = []
        x_vals_for_plot = []
        y_vals_for_plot = []

        for _, row in comp_df.iterrows():
            for rep_col in ['Norm_Rep1', 'Norm_Rep2', 'Norm_Rep3', 'Norm_Rep4']:
                if not pd.isna(row[rep_col]):
                    x_vals_for_fit.append(float(row['Conc']))
                    y_vals_for_fit.append(float(row[rep_col]))

            if not pd.isna(row['Mean']) and row['Conc'] != 'N/A':
                x_vals_for_plot.append(float(row['Conc']))
                y_vals_for_plot.append(float(row['Mean']))

        x_vals_for_fit = np.array(x_vals_for_fit)
        y_vals_for_fit = np.array(y_vals_for_fit)

        if len(x_vals_for_fit) > 3:
            def model_4pl(x_log, top, bottom, log_ic50, hill_slope):
                return bottom + (top - bottom) / (1 + 10 ** ((x_log - log_ic50) * hill_slope))

            x_log = np.log10(x_vals_for_fit)
            p0 = [1.0, 0.0, np.median(x_log), 1.0]
            bounds = ([0.5, -0.2, -4.0, 0.1], [1.5, 0.8, 4.0, 10.0])

            ax2.scatter(x_vals_for_plot, y_vals_for_plot, facecolor=COLORS[comp], edgecolor='black', s=50, zorder=3,
                        alpha=0.9)

            try:
                popt, _ = curve_fit(model_4pl, x_log, y_vals_for_fit, p0=p0, bounds=bounds, maxfev=10000)

                # Extract fit parameters
                bottom_asymptote = popt[1]
                ic50 = 10 ** popt[2]

                # Calculate Max Inhibition (Efficacy) and clamp extreme values for display
                max_inhib = (1.0 - bottom_asymptote) * 100
                max_inhib = max(0, min(100, max_inhib))

                x_fit_log = np.linspace(np.log10(min(concentrations)), np.log10(max(concentrations)), 100)
                y_fit = model_4pl(x_fit_log, *popt)
                x_fit = 10 ** x_fit_log

                if ic50 > max(concentrations):
                    print(f"{comp:<15} | > {max(concentrations):<8} | {max_inhib:>5.1f}%")
                    ax2.plot(x_fit, y_fit, color=COLORS[comp], lw=2.5, label=f"{comp} (Weak/No Inhibition)")
                else:
                    print(f"{comp:<15} | {ic50:<8.3f}   | {max_inhib:>5.1f}%")
                    ax2.plot(x_fit, y_fit, color=COLORS[comp], lw=2.5,
                             label=f"{comp} (IC50 = {ic50:.2f} µM, Emax = {max_inhib:.0f}%)")

            except Exception:
                print(f"{comp:<15} | Fit Failed | N/A")
                ax2.plot([], [], color=COLORS[comp], lw=2.5, label=f"{comp} (Fit Failed)")

    ax2.set_xscale('log')
    ax2.set_xlabel('Concentration (µM)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('PGE2 Level\n(time-fold of C+)', fontsize=14, fontweight='bold')
    ax2.set_title('IC50 Dose-Response Curves', fontsize=16, fontweight='bold', pad=20)
    ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)

    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=12)
    fig2.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.3)

    # --- DATA EXPORT & PRINTING ---
    print("\n" + "=" * 80)
    print("TIDY DATASET & STATISTICAL TEST RESULTS")
    print("=" * 80)

    df['n_reps'] = df[['Rep1', 'Rep2', 'Rep3', 'Rep4']].count(axis=1)
    print_df = df[['Compound', 'Conc', 'n_reps',
                   'Rep1', 'Rep2', 'Rep3', 'Rep4',
                   'Norm_Rep1', 'Norm_Rep2', 'Norm_Rep3', 'Norm_Rep4',
                   'Mean', 'SEM', 'p_val', 'sig']].copy()

    print(print_df.to_string(index=False))
    print("=" * 80 + "\n")

    fig1.savefig('ELISA_PGE2.png', dpi=600, bbox_inches='tight')
    fig2.savefig('ELISA_IC50.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    analyze_elisa()