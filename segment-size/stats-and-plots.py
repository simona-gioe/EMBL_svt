# Import libraries
import os
import sys
import warnings
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Ignore seaborn UserWarning and pandas SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=UserWarning)
pd.options.mode.chained_assignment = None


### ===================== DEFINE FUNCTIONS =========================== ###

# Function to return different dataframes
def processed_control_df_finder(directory, file_list):
    for file_name in file_list:
        if 'sorted-by-time' in file_name:
            if 'processed' in file_name:
                if 'control' in file_name:
                    control_dataframe = pd.read_csv(directory + '/' + file_name)
    return control_dataframe


def processed_treatment_df_finder(directory, file_list):
    for file_name in file_list:
        if 'sorted-by-time' in file_name:
            if 'processed' in file_name:
                if 'treatment' in file_name:
                    treatment_dataframe = pd.read_csv(directory + '/' + file_name)
    return treatment_dataframe


def normalized_control_df_finder(directory, file_list):
    for file_name in file_list:
        if 'sorted-by-time' in file_name:
            if 'normalized' in file_name:
                if 'control' in file_name:
                    control_dataframe = pd.read_csv(directory + '/' + file_name)
    return control_dataframe


def normalized_treatment_df_finder(directory, file_list):
    for file_name in file_list:
        if 'sorted-by-time' in file_name:
            if 'normalized' in file_name:
                if 'treatment' in file_name:
                    treatment_dataframe = pd.read_csv(directory + '/' + file_name)
    return treatment_dataframe


# Function to adjust color palette
def get_adjusted_palette(colormap_name, num_colors, min_shade=0.2, max_shade=0.8):
    colormap = sns.color_palette(colormap_name, as_cmap=True)
    return [colormap(i) for i in np.linspace(min_shade, max_shade, num_colors)]


### ===================== EXECUTE CODE =========================== ###


# Find files in tidy directory
cwd = os.getcwd()
tidy_dir = cwd + '/tidy'
files = os.listdir(tidy_dir)

# Check if 'tidy' files are .csv
new_files = []
for file in files:
    if ".csv" in file:
        new_files.append(file)
files = new_files

# Find files in temp directory
temp_dir = cwd + '/temp'
temp_files = os.listdir(temp_dir)

# Create new directories for plots
plots_dir = cwd + "/stat-plots"
isExist = os.path.exists(plots_dir)
if not isExist:
    os.makedirs(plots_dir)

boxplots_dir = plots_dir + "/boxplots"
isExist = os.path.exists(boxplots_dir)
if not isExist:
    os.makedirs(boxplots_dir)

pointplots_dir = plots_dir + "/pointplots"
isExist = os.path.exists(pointplots_dir)
if not isExist:
    os.makedirs(pointplots_dir)

timeplots_dir = plots_dir + "/time-plots"
isExist = os.path.exists(timeplots_dir)
if not isExist:
    os.makedirs(timeplots_dir)

statistics_dir = cwd + "/stat-files"
isExist = os.path.exists(statistics_dir)
if not isExist:
    os.makedirs(statistics_dir)

# Load your CSV file into a pandas DataFrame
for file in files:
    dataframe = pd.read_csv(tidy_dir + '/' + file)

    # Change name of somite columns into somite numbers
    all_columns = dataframe.columns
    new_columns = {}
    for column in all_columns:
        if 'somite-' in column:
            new_name = int(column.split('somite-')[1])
            new_columns[column] = new_name
        else:
            new_columns[column] = column
    dataframe.rename(columns=new_columns, inplace=True)

    # Identify the columns containing somite lengths
    somite_columns = dataframe.iloc[:, 4:]
    # Reorganise the dataframe by melting
    melted_dataframe = pd.melt(dataframe, id_vars=['experiment-date', 'sample-id', 'period', 'condition'],
                               value_vars=somite_columns, var_name="somite", value_name="length")

    # Replace labels in melted dataframe
    melted_dataframe["condition"] = melted_dataframe["condition"].replace({"C": "CTRL", "D": "DAPT"})

    # Generate .csv file with summary statistics
    summary_stats = melted_dataframe.groupby(["somite", "condition"])["length"].agg(
        mean=np.mean,
        std=np.std,
        sem=lambda x: np.std(x, ddof=1) / np.sqrt(len(x))  # for standard error of the mean
    ).reset_index()

    summary_table_path = f"{statistics_dir}/{file[5:-4]}-summary-stats.csv"
    summary_stats.to_csv(summary_table_path, index=False)

    ### Check for significance of segment size means difference between CTRL and DAPT samples
    if 'lengths' in file:
        # Check if the data is normally distributed and equally variant
        somites = melted_dataframe['somite'].unique()  # get unique somite numbers
        results = []

        for somite in somites:
            controls = \
            melted_dataframe[(melted_dataframe['somite'] == somite) & (melted_dataframe['condition'] == 'CTRL')][
                'length']
            controls = controls.dropna()
            entrained = \
            melted_dataframe[(melted_dataframe['somite'] == somite) & (melted_dataframe['condition'] == 'DAPT')][
                'length']
            entrained = entrained.dropna()

            # Skip somites with insufficient data
            if len(controls) < 3 or len(entrained) < 3:
                continue

            # Skip if variance is 0
            if np.var(controls, ddof=1) == 0 or np.var(entrained, ddof=1) == 0:
                continue
            
            test_name = "Welch's t-test"
            stat, p_val = stats.ttest_ind(controls, entrained, equal_var=False, alternative="two-sided")


            # Store the tests' results
            results.append({
                "somite": somite,
                "statistical_test": test_name,
                "p_value": p_val
            })

        # Convert statistical results to dataframe
        stats_dataframe = pd.DataFrame(results)

        # Save the stats_dataframe to a .csv file
        stats_dataframe.to_csv(statistics_dir + "/" + file[5:-4] + "_statistics.csv", index=False)

        ### Plot significant somites

        # Define options for plotting
        if "normalized" in file:
            low_limit = 0
            high_limit = 1.5
            y_label = 'segment length (normalized to somite 0)'
        else:
            low_limit = 10
            high_limit = 150
            y_label = 'segment length (microns)'

        # Define duble color palette
        div_palette = ["#984EA3", "#4DAF4A"]
        div_palette_r = ["#4DAF4A", "#984EA3"]

        if "p_value" in stats_dataframe.columns:
            # Create an error plot for all data separated by condition (mean and sem)
            fig, ax = plt.subplots()
            ax.set(ylim=(low_limit, high_limit))
            plt.ylabel(y_label)
            plt.xlabel('segment number')
            sns.pointplot(data=melted_dataframe, x="somite", y="length", hue="condition", palette=div_palette,
                          estimator="mean", errorbar="se")

            # Get significant somites for plotting
            significant_somites = stats_dataframe[stats_dataframe["p_value"] < 0.05]

            # Rate significance and assign asterisks
            for _, row in significant_somites.iterrows():
                somite = row["somite"]
                p_val = row["p_value"]

                if p_val <= 0.001:
                    significance = "***"
                elif p_val <= 0.01:
                    significance = "**"
                else:
                    significance = "*"

                # Get the max y-value for each somite
                max_y = melted_dataframe[melted_dataframe["somite"] == somite]["length"].max()
                ylim_low, ylim_high = ax.get_ylim()
                y_position = min(max_y + 0.005, ylim_high - 0.05)
                # Add significance asterisk
                ax.text(somite, y_position, significance, ha='center', va='bottom', fontsize=12, fontweight='bold',
                        color='black')

            # Save the plot
            mean_sem_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-conditionsHue-mean-sem_significance.png')
            mean_sem_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-conditionsHue-mean-sem_significance.pdf')
            # plt.show()
            plt.close(mean_sem_plot)

        if "p_value" in stats_dataframe.columns:
            # Create an error plot for all data separated by condition (mean and std)
            fig, ax = plt.subplots()
            ax.set(ylim=(low_limit, high_limit))
            plt.ylabel(y_label)
            plt.xlabel('segment number')
            sns.pointplot(data=melted_dataframe, x="somite", y="length", hue="condition", palette=div_palette,
                          estimator="mean", errorbar="sd")

            # Get significant somites for plotting
            significant_somites = stats_dataframe[stats_dataframe["p_value"] < 0.05]

            # Rate significance and assign asterisks
            for _, row in significant_somites.iterrows():
                somite = row["somite"]
                p_val = row["p_value"]

                if p_val <= 0.001:
                    significance = "***"
                elif p_val <= 0.01:
                    significance = "**"
                else:
                    significance = "*"

                # Get the max y-value for each somite
                max_y = melted_dataframe[melted_dataframe["somite"] == somite]["length"].max()
                ylim_low, ylim_high = ax.get_ylim()
                y_position = min(max_y + 0.005, ylim_high - 0.05)
                # Add significance asterisk
                ax.text(somite, y_position, significance, ha='center', va='bottom', fontsize=12, fontweight='bold',
                        color='black')

            # Save the plot
            mean_std_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-conditionsHue-mean-std_significance.png')
            mean_std_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-conditionsHue-mean-std_significance.pdf')
            # plt.show()
            plt.close(mean_std_plot)

    # Define options for plotting
    num_experiment_dates = melted_dataframe["experiment-date"].nunique()  # Get number of unique experiments
    if "control" in file:
        graph_color = '#984EA3'  # blue
        sample_title = 'CTRL samples'
        full_palette = get_adjusted_palette("Purples_r", num_experiment_dates, min_shade=0.3, max_shade=0.9)
    elif "treatment" in file:
        graph_color = '#4DAF4A'  # red (dapt)
        sample_title = 'entrained samples'
        full_palette = get_adjusted_palette("Greens_r", num_experiment_dates, min_shade=0.3, max_shade=0.9)

    if "normalized" in file:
        low_limit = 0
        high_limit = 1.5
        y_label = 'segment length (normalized to somite 0)'
    else:
        low_limit = 10
        high_limit = 150
        y_label = 'segment length (microns)'

    # Define duble color palette
    div_palette = ["#984EA3", "#4DAF4A"]
    div_palette_r = ["#4DAF4A", "#984EA3"]

    # Create a boxplot for the data (shows median and all data points
    fig, ax = plt.subplots()
    sns.set_theme(style='whitegrid')
    ax.set(ylim=(low_limit, high_limit))
    plt.ylabel(y_label)
    plt.xlabel('segment number')
    sns.boxplot(data=melted_dataframe, x=melted_dataframe['somite'], y="length", hue="condition", palette=div_palette,
                orient="v",
                saturation=0.5, dodge=1.0, gap=0.50)
    sns.stripplot(data=melted_dataframe, x="somite", y="length", hue="condition", palette=div_palette, orient="v",
                  dodge=1.0)
    boxplot = plt.savefig(boxplots_dir + '/' + file[5:-4] + '-conditionsHue-boxplot.png')
    boxplot = plt.savefig(boxplots_dir + '/' + file[5:-4] + '-conditionsHue-boxplot.pdf')
    # plt.show()
    plt.close(boxplot)

    fig, ax = plt.subplots()
    sns.set_theme(style='whitegrid')
    ax.set(ylim=(low_limit, high_limit))
    plt.ylabel(y_label)
    plt.xlabel('segment number')
    sns.boxplot(data=melted_dataframe, x=melted_dataframe['somite'], y="length", hue="condition", palette=div_palette_r,
                orient="v",
                saturation=0.5, dodge=1.0, gap=0.50)
    sns.stripplot(data=melted_dataframe, x="somite", y="length", hue="condition", palette=div_palette_r, orient="v",
                  dodge=1.0)
    boxplot = plt.savefig(boxplots_dir + '/' + file[5:-4] + '-conditionsHue-boxplot_r.png')
    boxplot = plt.savefig(boxplots_dir + '/' + file[5:-4] + '-conditionsHue-boxplot_r.pdf')
    # plt.show()
    plt.close(boxplot)

    # Create an error plot for all data separated by condition (mean and sem)
    fig, ax = plt.subplots()
    ax.set(ylim=(low_limit, high_limit))
    plt.ylabel(y_label)
    plt.xlabel('segment number')
    sns.pointplot(data=melted_dataframe, x="somite", y="length", hue="condition", palette=div_palette, estimator="mean",
                  errorbar="se")
    mean_sem_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-conditionsHue-mean-sem.png')
    mean_sem_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-conditionsHue-mean-sem.pdf')
    # plt.show()
    plt.close(mean_sem_plot)

    # Create an error plot for all data separated by condition (mean and std)
    fig, ax = plt.subplots()
    ax.set(ylim=(low_limit, high_limit))
    plt.ylabel(y_label)
    plt.xlabel('segment number')
    sns.pointplot(data=melted_dataframe, x="somite", y="length", hue="condition", palette=div_palette, estimator="mean",
                  errorbar="sd")
    mean_std_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-conditionsHue-mean-std.png')
    mean_std_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-conditionsHue-mean-std.pdf')
    # plt.show()
    plt.close(mean_std_plot)

    if 'control' in file:
        # Create an error plot for all data separated by experiment (mean and sem)
        fig, ax = plt.subplots()
        ax.set(ylim=(low_limit, high_limit))
        plt.ylabel(y_label)
        plt.xlabel('segment number')
        sns.pointplot(data=melted_dataframe, x="somite", y="length", hue="experiment-date", palette=full_palette,
                      estimator="mean",
                      errorbar="se", legend="full")
        mean_sem_d_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-experimentHue-mean-sem.png')
        mean_sem_d_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-experimentHue-mean-sem.pdf')
        # plt.show()
        plt.close(mean_sem_d_plot)

        # Create an error plot for all data separated by experiment (mean and std)
        fig, ax = plt.subplots()
        ax.set(ylim=(low_limit, high_limit))
        plt.ylabel(y_label)
        plt.xlabel('segment number')
        sns.pointplot(data=melted_dataframe, x="somite", y="length", hue="experiment-date", palette=full_palette,
                      estimator="mean",
                      errorbar="sd", legend="full")
        mean_std_d_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-experimentHue-mean-std.png')
        mean_std_d_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-experimentHue-mean-std.pdf')
        # plt.show()
        plt.close(mean_std_d_plot)

        # Create a boxplot for the data (shows median and all data points)
        fig, ax = plt.subplots()
        sns.set_theme(style='whitegrid')
        ax.set(ylim=(low_limit, high_limit))
        plt.ylabel(y_label)
        plt.xlabel('segment number')
        sns.boxplot(data=melted_dataframe, x=melted_dataframe['somite'], y="length", hue="experiment-date",
                    palette=full_palette, orient="v",
                    saturation=1.0, dodge=True, gap=0.50, legend="full")
        # sns.stripplot(data=melted_dataframe, x="somite", y="length", hue="experiment-date", orient="v")
        boxplot = plt.savefig(boxplots_dir + '/' + file[5:-4] + '-experimentHue-boxplot.png')
        boxplot = plt.savefig(boxplots_dir + '/' + file[5:-4] + '-experimentHue-boxplot.pdf')
        # plt.show()
        plt.close(boxplot)

    elif 'treatment' in file:
        # Create an error plot for all data separated by experiment (mean and sem)
        fig, ax = plt.subplots()
        ax.set(ylim=(low_limit, high_limit))
        plt.ylabel(y_label)
        plt.xlabel('segment number')
        sns.pointplot(data=melted_dataframe, x="somite", y="length", hue="experiment-date", palette=full_palette,
                      estimator="mean",
                      errorbar="se", legend="full")
        mean_sem_d_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-experimentHue-mean-sem.png')
        mean_sem_d_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-experimentHue-mean-sem.pdf')
        # plt.show()
        plt.close(mean_sem_d_plot)

        # Create an error plot for all data separated by experiment (mean and std)
        fig, ax = plt.subplots()
        ax.set(ylim=(low_limit, high_limit))
        plt.ylabel(y_label)
        plt.xlabel('segment number')
        sns.pointplot(data=melted_dataframe, x="somite", y="length", hue="experiment-date", palette=full_palette,
                      estimator="mean",
                      errorbar="sd", legend="full")
        mean_std_d_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-experimentHue-mean-std.png')
        mean_std_d_plot = plt.savefig(pointplots_dir + '/' + file[5:-4] + '-experimentHue-mean-std.pdf')
        # plt.show()
        plt.close(mean_std_d_plot)

        # Create a boxplot for the data (shows median and all data points)
        fig, ax = plt.subplots()
        sns.set_theme(style='whitegrid')
        ax.set(ylim=(low_limit, high_limit))
        plt.ylabel(y_label)
        plt.xlabel('segment number')
        sns.boxplot(data=melted_dataframe, x=melted_dataframe['somite'], y="length", hue="experiment-date",
                    palette=full_palette, orient="v",
                    saturation=1.0, dodge=True, gap=0.50, legend="full")
        # sns.stripplot(data=melted_dataframe, x="somite", y="length", hue="experiment-date", orient="v")
        boxplot = plt.savefig(boxplots_dir + '/' + file[5:-4] + '-experimentHue-boxplot.png')
        boxplot = plt.savefig(boxplots_dir + '/' + file[5:-4] + '-experimentHue-boxplot.pdf')
        # plt.show()
        plt.close(boxplot)

    # Reorganise dataframe for summary statistics and plotting
    if 'control' in file:
        somite_num = range(melted_dataframe['somite'].max() + 1)
        # Create an errorplot with matplotlib
        fig, ax = plt.subplots()
        plt.errorbar(x=somite_num, y=somite_columns.mean(), yerr=somite_columns.sem())
        # plt.show()
        plt.close()

    elif 'treatment' in file:
        somite_num = range(melted_dataframe['somite'].max() + 1)
        # Create an errorplot with matplotlib
        fig, ax = plt.subplots()
        plt.errorbar(x=somite_num, y=somite_columns.mean(), yerr=somite_columns.sem())
        plt.close()
        # plt.show()

# Load your CSV file into a pandas DataFrame

processed_controls_df = processed_control_df_finder(temp_dir, temp_files)
processed_treatment_df = processed_treatment_df_finder(temp_dir, temp_files)
normalized_controls_df = normalized_control_df_finder(temp_dir, temp_files)
normalized_treatment_df = normalized_treatment_df_finder(temp_dir, temp_files)

# Read values (processed controls)
segment_ids_pc = processed_controls_df.iloc[:, 0]
segment_lengths_pc = processed_controls_df.iloc[:, 1]
segment_sds_pc = processed_controls_df.iloc[:, 2]
measurement_time_pc = processed_controls_df.iloc[:, 3]

# Read values (processed treatment)
segment_ids_pt = processed_treatment_df.iloc[:, 0]
segment_lengths_pt = processed_treatment_df.iloc[:, 1]
segment_sds_pt = processed_treatment_df.iloc[:, 2]
measurement_time_pt = processed_treatment_df.iloc[:, 3]

# Read values (normalized controls)
segment_ids_nc = normalized_controls_df.iloc[:, 0]
segment_lengths_nc = normalized_controls_df.iloc[:, 1]
measurement_time_nc = normalized_controls_df.iloc[:, 2]

# Read values (normalized treatment)
segment_ids_nt = normalized_treatment_df.iloc[:, 0]
segment_lengths_nt = normalized_treatment_df.iloc[:, 1]
measurement_time_nt = normalized_treatment_df.iloc[:, 2]

# Plot processed data
fig, ax = plt.subplots()
plt.plot(measurement_time_pc, segment_lengths_pc, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(measurement_time_pt, segment_lengths_pt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlabel('time (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths over time')
processed_timeplot = plt.savefig(timeplots_dir + '/' + 'processed-lenghts-over-time.png')
processed_timeplot = plt.savefig(timeplots_dir + '/' + 'processed-lenghts-over-time.pdf')
# plt.show()
plt.close(processed_timeplot)

# Plot normalized data
fig, ax = plt.subplots()
plt.plot(measurement_time_nc, segment_lengths_nc, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(measurement_time_nt, segment_lengths_nt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.ylim(0.25, 1.35)
plt.xlabel('time (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths over time')
normalized_timeplot = plt.savefig(timeplots_dir + '/' + 'normalized-lengths-over-time.png')
normalized_timeplot = plt.savefig(timeplots_dir + '/' + 'normalized-lengths-over-time.pdf')
# plt.show()
plt.close(normalized_timeplot)

##### ENTRAINMENT DATA PLOTS

# Check if entrainment files are present
entrainment_dir = cwd + "/entrainment-files"
isExistEntrainment = os.path.exists(entrainment_dir)
if not isExistEntrainment:
    print("Entrainment data not found - stopping execution")
    sys.exit()

### Plot size against period

# Create directory for period plots
periodplots_dir = plots_dir + "/period-plots"
isExist = os.path.exists(periodplots_dir)
if not isExist:
    os.makedirs(periodplots_dir)

# Filter out segments and samples with missing global periods
global_period_processed_controls_df = processed_controls_df[processed_controls_df['GR-period'].notna()]
global_period_processed_treatment_df = processed_treatment_df[processed_treatment_df['GR-period'].notna()]
global_period_normalized_controls_df = normalized_controls_df[normalized_controls_df['GR-period'].notna()]
global_period_normalized_treatment_df = normalized_treatment_df[normalized_treatment_df['GR-period'].notna()]

# Filter out segments and samples with missing anterior periods
anterior_period_processed_controls_df = processed_controls_df[processed_controls_df['AR-period'].notna()]
anterior_period_processed_treatment_df = processed_treatment_df[processed_treatment_df['AR-period'].notna()]
anterior_period_normalized_controls_df = normalized_controls_df[normalized_controls_df['AR-period'].notna()]
anterior_period_normalized_treatment_df = normalized_treatment_df[normalized_treatment_df['AR-period'].notna()]

# Filter out segments and samples with missing posterior periods
posterior_period_processed_controls_df = processed_controls_df[processed_controls_df['PR-period'].notna()]
posterior_period_processed_treatment_df = processed_treatment_df[processed_treatment_df['PR-period'].notna()]
posterior_period_normalized_controls_df = normalized_controls_df[normalized_controls_df['PR-period'].notna()]
posterior_period_normalized_treatment_df = normalized_treatment_df[normalized_treatment_df['PR-period'].notna()]

# Isolate period values from dataframes
gr_period_pc = global_period_processed_controls_df.iloc[:, 4]
ar_period_pc = anterior_period_processed_controls_df.iloc[:, 5]
pr_period_pc = posterior_period_processed_controls_df.iloc[:, 6]
gr_period_pt = global_period_processed_treatment_df.iloc[:, 4]
ar_period_pt = anterior_period_processed_treatment_df.iloc[:, 5]
pr_period_pt = posterior_period_processed_treatment_df.iloc[:, 6]
gr_period_nc = global_period_normalized_controls_df.iloc[:, 3]
ar_period_nc = anterior_period_normalized_controls_df.iloc[:, 4]
pr_period_nc = posterior_period_normalized_controls_df.iloc[:, 5]
gr_period_nt = global_period_normalized_treatment_df.iloc[:, 3]
ar_period_nt = anterior_period_normalized_treatment_df.iloc[:, 4]
pr_period_nt = posterior_period_normalized_treatment_df.iloc[:, 5]

# Isolate length values from filtered dataframes
gr_segments_pc = global_period_processed_controls_df.iloc[:, 1]
ar_segments_pc = anterior_period_processed_controls_df.iloc[:, 1]
pr_segments_pc = posterior_period_processed_controls_df.iloc[:, 1]
gr_segments_pt = global_period_processed_treatment_df.iloc[:, 1]
ar_segments_pt = anterior_period_processed_treatment_df.iloc[:, 1]
pr_segments_pt = posterior_period_processed_treatment_df.iloc[:, 1]
gr_segments_nc = global_period_normalized_controls_df.iloc[:, 1]
ar_segments_nc = anterior_period_normalized_controls_df.iloc[:, 1]
pr_segments_nc = posterior_period_normalized_controls_df.iloc[:, 1]
gr_segments_nt = global_period_normalized_treatment_df.iloc[:, 1]
ar_segments_nt = anterior_period_normalized_treatment_df.iloc[:, 1]
pr_segments_nt = posterior_period_normalized_treatment_df.iloc[:, 1]

# Plot processed data (global ROI)
fig, ax = plt.subplots()
plt.plot(gr_period_pc, gr_segments_pc, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(gr_period_pt, gr_segments_pt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.xlabel('period (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths against global ROI period')
gr_processed_periodplot = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_GR.png')
gr_processed_periodplot = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_GR.pdf')
# plt.show()
plt.close(gr_processed_periodplot)

# Plot normalized data (global ROI)
fig, ax = plt.subplots()
plt.plot(gr_period_nc, gr_segments_nc, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(gr_period_nt, gr_segments_nt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.ylim(0.25, 1.35)
plt.xlabel('period (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths against global ROI period')
gr_normalized_periodplot = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_GR.png')
gr_normalized_periodplot = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_GR.pdf')
# plt.show()
plt.close(gr_normalized_periodplot)

# Plot processed data (anterior ROI)
fig, ax = plt.subplots()
plt.plot(ar_period_pc, ar_segments_pc, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(ar_period_pt, ar_segments_pt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.xlabel('period (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths against anterior moving ROI period')
ar_processed_periodplot = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_AR.png')
ar_processed_periodplot = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_AR.pdf')
# plt.show()
plt.close(ar_processed_periodplot)

# Plot normalized data (anterior ROI)
fig, ax = plt.subplots()
plt.plot(ar_period_nc, ar_segments_nc, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(ar_period_nt, ar_segments_nt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.ylim(0.25, 1.35)
plt.xlabel('period (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths against anterior moving ROI period')
ar_normalized_periodplot = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_AR.png')
ar_normalized_periodplot = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_AR.pdf')
# plt.show()
plt.close(ar_normalized_periodplot)

# Plot processed data (posterior ROI)
fig, ax = plt.subplots()
plt.plot(pr_period_pc, pr_segments_pc, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(pr_period_pt, pr_segments_pt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.xlabel('period (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths against posterior ROI period')
pr_processed_periodplot = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_PR.png')
pr_processed_periodplot = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_PR.pdf')
# plt.show()
plt.close(pr_processed_periodplot)

# Plot normalized data (posterior ROI)
fig, ax = plt.subplots()
plt.plot(pr_period_nc, pr_segments_nc, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(pr_period_nt, pr_segments_nt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.ylim(0.25, 1.35)
plt.xlabel('period (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths against posterior ROI period')
pr_normalized_periodplot = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_PR.png')
pr_normalized_periodplot = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_PR.pdf')
# plt.show()
plt.close(pr_normalized_periodplot)

## Plot size vs period plot subsets for segments 1-5 and 6-10

# Isolate segment numbers in new columns
global_period_processed_controls_df['segment-number'] = global_period_processed_controls_df['segment'].apply(lambda x: int(x.split('_')[-1]))
global_period_processed_treatment_df['segment-number'] = global_period_processed_treatment_df['segment'].apply(lambda x: int(x.split('_')[-1]))
global_period_normalized_controls_df['segment-number'] = global_period_normalized_controls_df['segment'].apply(lambda x: int(x.split('_')[-1]))
global_period_normalized_treatment_df['segment-number'] = global_period_normalized_treatment_df['segment'].apply(lambda x: int(x.split('_')[-1]))

anterior_period_processed_controls_df['segment-number'] = anterior_period_processed_controls_df['segment'].apply(lambda x: int(x.split('_')[-1]))
anterior_period_processed_treatment_df['segment-number'] = anterior_period_processed_treatment_df['segment'].apply(lambda x: int(x.split('_')[-1]))
anterior_period_normalized_controls_df['segment-number'] = anterior_period_normalized_controls_df['segment'].apply(lambda x: int(x.split('_')[-1]))
anterior_period_normalized_treatment_df['segment-number'] = anterior_period_normalized_treatment_df['segment'].apply(lambda x: int(x.split('_')[-1]))

posterior_period_processed_controls_df['segment-number'] = posterior_period_processed_controls_df['segment'].apply(lambda x: int(x.split('_')[-1]))
posterior_period_processed_treatment_df['segment-number'] = posterior_period_processed_treatment_df['segment'].apply(lambda x: int(x.split('_')[-1]))
posterior_period_normalized_controls_df['segment-number'] = posterior_period_normalized_controls_df['segment'].apply(lambda x: int(x.split('_')[-1]))
posterior_period_normalized_treatment_df['segment-number'] = posterior_period_normalized_treatment_df['segment'].apply(lambda x: int(x.split('_')[-1]))

# Filter datasets for segments
global_period_processed_controls_15 = global_period_processed_controls_df[global_period_processed_controls_df['segment-number'].between(1, 5)]
global_period_processed_controls_610 = global_period_processed_controls_df[global_period_processed_controls_df['segment-number'].between(6, 10)]
global_period_processed_treatment_15 = global_period_processed_treatment_df[global_period_processed_treatment_df['segment-number'].between(1, 5)]
global_period_processed_treatment_610 = global_period_processed_treatment_df[global_period_processed_treatment_df['segment-number'].between(6, 10)]
global_period_normalized_controls_15 = global_period_normalized_controls_df[global_period_normalized_controls_df['segment-number'].between(1, 5)]
global_period_normalized_controls_610 = global_period_normalized_controls_df[global_period_normalized_controls_df['segment-number'].between(6, 10)]
global_period_normalized_treatment_15 = global_period_normalized_treatment_df[global_period_normalized_treatment_df['segment-number'].between(1, 5)]
global_period_normalized_treatment_610 = global_period_normalized_treatment_df[global_period_normalized_treatment_df['segment-number'].between(6, 10)]

anterior_period_processed_controls_15 = anterior_period_processed_controls_df[anterior_period_processed_controls_df['segment-number'].between(1, 5)]
anterior_period_processed_controls_610 = anterior_period_processed_controls_df[anterior_period_processed_controls_df['segment-number'].between(6, 10)]
anterior_period_processed_treatment_15 = anterior_period_processed_treatment_df[anterior_period_processed_treatment_df['segment-number'].between(1, 5)]
anterior_period_processed_treatment_610 = anterior_period_processed_treatment_df[anterior_period_processed_treatment_df['segment-number'].between(6, 10)]
anterior_period_normalized_controls_15 = anterior_period_normalized_controls_df[anterior_period_normalized_controls_df['segment-number'].between(1, 5)]
anterior_period_normalized_controls_610 = anterior_period_normalized_controls_df[anterior_period_normalized_controls_df['segment-number'].between(6, 10)]
anterior_period_normalized_treatment_15 = anterior_period_normalized_treatment_df[anterior_period_normalized_treatment_df['segment-number'].between(1, 5)]
anterior_period_normalized_treatment_610 = anterior_period_normalized_treatment_df[anterior_period_normalized_treatment_df['segment-number'].between(6, 10)]

posterior_period_processed_controls_15 = posterior_period_processed_controls_df[posterior_period_processed_controls_df['segment-number'].between(1, 5)]
posterior_period_processed_controls_610 = posterior_period_processed_controls_df[posterior_period_processed_controls_df['segment-number'].between(6, 10)]
posterior_period_processed_treatment_15 = posterior_period_processed_treatment_df[posterior_period_processed_treatment_df['segment-number'].between(1, 5)]
posterior_period_processed_treatment_610 = posterior_period_processed_treatment_df[posterior_period_processed_treatment_df['segment-number'].between(6, 10)]
posterior_period_normalized_controls_15 = posterior_period_normalized_controls_df[posterior_period_normalized_controls_df['segment-number'].between(1, 5)]
posterior_period_normalized_controls_610 = posterior_period_normalized_controls_df[posterior_period_normalized_controls_df['segment-number'].between(6, 10)]
posterior_period_normalized_treatment_15 = posterior_period_normalized_treatment_df[posterior_period_normalized_treatment_df['segment-number'].between(1, 5)]
posterior_period_normalized_treatment_610 = posterior_period_normalized_treatment_df[posterior_period_normalized_treatment_df['segment-number'].between(6, 10)]


# Isolate period values from dataframes
gr_period_pc_15 = global_period_processed_controls_15.iloc[:, 4]
ar_period_pc_15 = anterior_period_processed_controls_15.iloc[:, 5]
pr_period_pc_15 = posterior_period_processed_controls_15.iloc[:, 6]
gr_period_pt_15 = global_period_processed_treatment_15.iloc[:, 4]
ar_period_pt_15 = anterior_period_processed_treatment_15.iloc[:, 5]
pr_period_pt_15 = posterior_period_processed_treatment_15.iloc[:, 6]
gr_period_nc_15 = global_period_normalized_controls_15.iloc[:, 3]
ar_period_nc_15 = anterior_period_normalized_controls_15.iloc[:, 4]
pr_period_nc_15 = posterior_period_normalized_controls_15.iloc[:, 5]
gr_period_nt_15 = global_period_normalized_treatment_15.iloc[:, 3]
ar_period_nt_15 = anterior_period_normalized_treatment_15.iloc[:, 4]
pr_period_nt_15 = posterior_period_normalized_treatment_15.iloc[:, 5]
gr_period_pc_610 = global_period_processed_controls_610.iloc[:, 4]
ar_period_pc_610 = anterior_period_processed_controls_610.iloc[:, 5]
pr_period_pc_610 = posterior_period_processed_controls_610.iloc[:, 6]
gr_period_pt_610 = global_period_processed_treatment_610.iloc[:, 4]
ar_period_pt_610 = anterior_period_processed_treatment_610.iloc[:, 5]
pr_period_pt_610 = posterior_period_processed_treatment_610.iloc[:, 6]
gr_period_nc_610 = global_period_normalized_controls_610.iloc[:, 3]
ar_period_nc_610 = anterior_period_normalized_controls_610.iloc[:, 4]
pr_period_nc_610 = posterior_period_normalized_controls_610.iloc[:, 5]
gr_period_nt_610 = global_period_normalized_treatment_610.iloc[:, 3]
ar_period_nt_610 = anterior_period_normalized_treatment_610.iloc[:, 4]
pr_period_nt_610 = posterior_period_normalized_treatment_610.iloc[:, 5]

# Isolate length values from filtered dataframes
gr_segments_pc_15 = global_period_processed_controls_15.iloc[:, 1]
ar_segments_pc_15 = anterior_period_processed_controls_15.iloc[:, 1]
pr_segments_pc_15 = posterior_period_processed_controls_15.iloc[:, 1]
gr_segments_pt_15 = global_period_processed_treatment_15.iloc[:, 1]
ar_segments_pt_15 = anterior_period_processed_treatment_15.iloc[:, 1]
pr_segments_pt_15 = posterior_period_processed_treatment_15.iloc[:, 1]
gr_segments_nc_15 = global_period_normalized_controls_15.iloc[:, 1]
ar_segments_nc_15 = anterior_period_normalized_controls_15.iloc[:, 1]
pr_segments_nc_15 = posterior_period_normalized_controls_15.iloc[:, 1]
gr_segments_nt_15 = global_period_normalized_treatment_15.iloc[:, 1]
ar_segments_nt_15 = anterior_period_normalized_treatment_15.iloc[:, 1]
pr_segments_nt_15 = posterior_period_normalized_treatment_15.iloc[:, 1]
gr_segments_pc_610 = global_period_processed_controls_610.iloc[:, 1]
ar_segments_pc_610 = anterior_period_processed_controls_610.iloc[:, 1]
pr_segments_pc_610 = posterior_period_processed_controls_610.iloc[:, 1]
gr_segments_pt_610 = global_period_processed_treatment_610.iloc[:, 1]
ar_segments_pt_610 = anterior_period_processed_treatment_610.iloc[:, 1]
pr_segments_pt_610 = posterior_period_processed_treatment_610.iloc[:, 1]
gr_segments_nc_610 = global_period_normalized_controls_610.iloc[:, 1]
ar_segments_nc_610 = anterior_period_normalized_controls_610.iloc[:, 1]
pr_segments_nc_610 = posterior_period_normalized_controls_610.iloc[:, 1]
gr_segments_nt_610 = global_period_normalized_treatment_610.iloc[:, 1]
ar_segments_nt_610 = anterior_period_normalized_treatment_610.iloc[:, 1]
pr_segments_nt_610 = posterior_period_normalized_treatment_610.iloc[:, 1]

# Plot processed data (global ROI), segments 1-5
fig, ax = plt.subplots()
plt.plot(gr_period_pc_15, gr_segments_pc_15, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(gr_period_pt_15, gr_segments_pt_15, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.xlabel('period (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths (1-5) against global ROI period')
gr_processed_periodplot_15 = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_GR_1-5.png')
gr_processed_periodplot_15 = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_GR_1-5.pdf')
# plt.show()
plt.close(gr_processed_periodplot_15)

# Plot processed data (global ROI), segments 6-10
fig, ax = plt.subplots()
plt.plot(gr_period_pc_610, gr_segments_pc_610, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(gr_period_pt_610, gr_segments_pt_610, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.xlabel('period (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths (6-10) against global ROI period')
gr_processed_periodplot_610= plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_GR_6-10.png')
gr_processed_periodplot_610 = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_GR_6-10.pdf')
# plt.show()
plt.close(gr_processed_periodplot_610)

# Plot normalized data (global ROI), segments 1-5
fig, ax = plt.subplots()
plt.plot(gr_period_nc_15, gr_segments_nc_15, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(gr_period_nt_15, gr_segments_nt_15, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.ylim(0.25, 1.35)
plt.xlabel('period (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths (1-5) against global ROI period')
gr_normalized_periodplot_15 = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_GR_1-5.png')
gr_normalized_periodplot_15 = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_GR_1-5.pdf')
# plt.show()
plt.close(gr_normalized_periodplot_15)

# Plot normalized data (global ROI), segments 6-10
fig, ax = plt.subplots()
plt.plot(gr_period_nc_610, gr_segments_nc_610, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(gr_period_nt_610, gr_segments_nt_610, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.ylim(0.25, 1.35)
plt.xlabel('period (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths (6-10) against global ROI period')
gr_normalized_periodplot_610 = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_GR_6-10.png')
gr_normalized_periodplot_610 = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_GR_6-10.pdf')
# plt.show()
plt.close(gr_normalized_periodplot_610)

# Plot processed data (anterior ROI), segments 1-5
fig, ax = plt.subplots()
plt.plot(ar_period_pc_15, ar_segments_pc_15, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(ar_period_pt_15, ar_segments_pt_15, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.xlabel('period (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths (1-5) against anterior moving ROI period')
ar_processed_periodplot_15 = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_AR_1-5.png')
ar_processed_periodplot_15 = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_AR_1-5.pdf')
# plt.show()
plt.close(ar_processed_periodplot_15)

# Plot processed data (anterior ROI), segments 6-10
fig, ax = plt.subplots()
plt.plot(ar_period_pc_610, ar_segments_pc_610, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(ar_period_pt_610, ar_segments_pt_610, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.xlabel('period (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths (6-10) against anterior moving ROI period')
ar_processed_periodplot_610 = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_AR_6-10.png')
ar_processed_periodplot_610 = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_AR_6-10.pdf')
# plt.show()
plt.close(ar_processed_periodplot_610)

# Plot normalized data (anterior ROI), segments 1-5
fig, ax = plt.subplots()
plt.plot(ar_period_nc_15, ar_segments_nc_15, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(ar_period_nt_15, ar_segments_nt_15, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.ylim(0.25, 1.35)
plt.xlabel('period (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths (1-5) against anterior moving ROI period')
ar_normalized_periodplot_15 = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_AR_1-5.png')
ar_normalized_periodplot_15 = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_AR_1-5.pdf')
# plt.show()
plt.close(ar_normalized_periodplot_15)

# Plot normalized data (anterior ROI), segments 6-10
fig, ax = plt.subplots()
plt.plot(ar_period_nc_610, ar_segments_nc_610, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(ar_period_nt_610, ar_segments_nt_610, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.ylim(0.25, 1.35)
plt.xlabel('period (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths (6-10) against anterior moving ROI period')
ar_normalized_periodplot_610 = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_AR_6-10.png')
ar_normalized_periodplot_610 = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_AR_6-10.pdf')
# plt.show()
plt.close(ar_normalized_periodplot_610)

# Plot processed data (posterior ROI), segments 1-5
fig, ax = plt.subplots()
plt.plot(pr_period_pc_15, pr_segments_pc_15, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(pr_period_pt_15, pr_segments_pt_15, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.xlabel('period (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths (1-5) against posterior ROI period')
pr_processed_periodplot_15 = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_PR_1-5.png')
pr_processed_periodplot_15 = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_PR_1-5.pdf')
# plt.show()
plt.close(pr_processed_periodplot_15)

# Plot processed data (posterior ROI), segments 6-10
fig, ax = plt.subplots()
plt.plot(pr_period_pc_610, pr_segments_pc_610, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(pr_period_pt_610, pr_segments_pt_610, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.xlabel('period (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths (6-10) against posterior ROI period')
pr_processed_periodplot_610 = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_PR_6-10.png')
pr_processed_periodplot_610 = plt.savefig(periodplots_dir + '/' + 'processed-lenghts-against-period_PR_6-10.pdf')
# plt.show()
plt.close(pr_processed_periodplot_610)

# Plot normalized data (posterior ROI), segments 1-5
fig, ax = plt.subplots()
plt.plot(pr_period_nc_15, pr_segments_nc_15, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(pr_period_nt_15, pr_segments_nt_15, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.ylim(0.25, 1.35)
plt.xlabel('period (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths (1-5) against posterior ROI period')
pr_normalized_periodplot_15 = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_PR_1-5.png')
pr_normalized_periodplot_15 = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_PR_1-5.pdf')
# plt.show()
plt.close(pr_normalized_periodplot_15)

# Plot normalized data (posterior ROI), segments 6-10
fig, ax = plt.subplots()
plt.plot(pr_period_nc_610, pr_segments_nc_610, color='#984EA3', marker='o', linestyle='None', alpha=0.7, label='CTRL')
plt.plot(pr_period_nt_610, pr_segments_nt_610, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlim(95, 210)
plt.ylim(0.25, 1.35)
plt.xlabel('period (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths (6-10) against posterior ROI period')
pr_normalized_periodplot_610 = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_PR_6-10.png')
pr_normalized_periodplot_610 = plt.savefig(periodplots_dir + '/' + 'normalized-lengths-against-period_PR_6-10.pdf')
# plt.show()
plt.close(pr_normalized_periodplot_610)


### Plot segment size against time of phase-locking

# Filter out samples and segments with missing phase-locking times
global_locking_processed_treatment_df = processed_treatment_df[processed_treatment_df['GR-locking-time'].notna()]
anterior_locking_processed_treatment_df = processed_treatment_df[processed_treatment_df['AR-locking-time'].notna()]
posterior_locking_processed_treatment_df = processed_treatment_df[processed_treatment_df['PR-locking-time'].notna()]
global_locking_normalized_treatment_df = normalized_treatment_df[normalized_treatment_df['GR-locking-time'].notna()]
anterior_locking_normalized_treatment_df = normalized_treatment_df[normalized_treatment_df['AR-locking-time'].notna()]
posterior_locking_normalized_treatment_df = normalized_treatment_df[normalized_treatment_df['PR-locking-time'].notna()]

# Isolate time values from filtered dataframes
gr_locking_pt = global_locking_processed_treatment_df.iloc[:, 7]
ar_locking_pt = anterior_locking_processed_treatment_df.iloc[:, 8]
pr_locking_pt = posterior_locking_processed_treatment_df.iloc[:, 9]
gr_locking_nt = global_locking_normalized_treatment_df.iloc[:, 6]
ar_locking_nt = anterior_locking_normalized_treatment_df.iloc[:, 7]
pr_locking_nt = posterior_locking_normalized_treatment_df.iloc[:, 8]

# Isolate segment size values from filtered dataframes
gr_segments_pt = global_locking_processed_treatment_df.iloc[:, 1]
ar_segments_pt = anterior_locking_processed_treatment_df.iloc[:, 1]
pr_segments_pt = posterior_locking_processed_treatment_df.iloc[:, 1]
gr_segments_nt = global_locking_normalized_treatment_df.iloc[:, 1]
ar_segments_nt = anterior_locking_normalized_treatment_df.iloc[:, 1]
pr_segments_nt = posterior_locking_normalized_treatment_df.iloc[:, 1]

# Plot processed data (global ROI)
fig, ax = plt.subplots()
plt.plot(gr_locking_pt, gr_segments_pt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlabel('time (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths over time (t = 0 is global ROI phase-locking time')
gr_processed_timeplot = plt.savefig(timeplots_dir + '/' + 'processed-lenghts-over-phase-locking-time_GR.png')
gr_processed_timeplot = plt.savefig(timeplots_dir + '/' + 'processed-lenghts-over-phase-locking-time_GR.pdf')
# plt.show()
plt.close(gr_processed_timeplot)

# Plot normalized data (global ROI)
fig, ax = plt.subplots()
plt.plot(gr_locking_nt, gr_segments_nt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.ylim(0.25, 1.35)
plt.xlabel('time (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths over time (t = 0 is global ROI phase-locking time')
gr_normalized_timeplot = plt.savefig(timeplots_dir + '/' + 'normalized-lengths-over-phase-locking-time_GR.png')
gr_normalized_timeplot = plt.savefig(timeplots_dir + '/' + 'normalized-lengths-over-phase-locking-time_GR.pdf')
# plt.show()
plt.close(gr_normalized_timeplot)

# Plot processed data (anterior ROI)
fig, ax = plt.subplots()
plt.plot(ar_locking_pt, ar_segments_pt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlabel('time (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths over time (t = 0 is anterior moving ROI phase-locking time')
ar_processed_timeplot = plt.savefig(timeplots_dir + '/' + 'processed-lenghts-over-phase-locking-time_AR.png')
ar_processed_timeplot = plt.savefig(timeplots_dir + '/' + 'processed-lenghts-over-phase-locking-time_AR.pdf')
# plt.show()
plt.close(ar_processed_timeplot)

# Plot normalized data (anterior ROI)
fig, ax = plt.subplots()
plt.plot(ar_locking_nt, ar_segments_nt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.ylim(0.25, 1.35)
plt.xlabel('time (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths over time (t = 0 is anterior moving ROI phase-locking time')
ar_normalized_timeplot = plt.savefig(timeplots_dir + '/' + 'normalized-lengths-over-phase-locking-time_AR.png')
ar_normalized_timeplot = plt.savefig(timeplots_dir + '/' + 'normalized-lengths-over-phase-locking-time_AR.pdf')
# plt.show()
plt.close(ar_normalized_timeplot)

# Plot processed data (posterior ROI)
fig, ax = plt.subplots()
plt.plot(pr_locking_pt, pr_segments_pt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.xlabel('time (min)')
plt.ylabel('segment length (microns)')
plt.legend(loc='best')
plt.title('segment lengths over time (t = 0 is posterior ROI phase-locking time')
pr_processed_timeplot = plt.savefig(timeplots_dir + '/' + 'processed-lenghts-over-phase-locking-time_PR.png')
pr_processed_timeplot = plt.savefig(timeplots_dir + '/' + 'processed-lenghts-over-phase-locking-time_PR.pdf')
# plt.show()
plt.close(pr_processed_timeplot)

# Plot normalized data (posterior ROI)
fig, ax = plt.subplots()
plt.plot(pr_locking_nt, pr_segments_nt, color='#4DAF4A', marker='o', linestyle='None', alpha=0.7, label='DAPT')
plt.ylim(0.25, 1.35)
plt.xlabel('time (min)')
plt.ylabel('segment length (normalized)')
plt.legend(loc='best')
plt.title('segment lengths over time (t = 0 is posterior ROI phase-locking time')
pr_normalized_timeplot = plt.savefig(timeplots_dir + '/' + 'normalized-lengths-over-phase-locking-time_PR.png')
pr_normalized_timeplot = plt.savefig(timeplots_dir + '/' + 'normalized-lengths-over-phase-locking-time_PR.pdf')
# plt.show()
plt.close(pr_normalized_timeplot)
