# Import libraries
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore pandas FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

### ===================== DEFINE FUNCTIONS =========================== ###

# Function to find the index of segment 0 in a list
def segment_index_finder(segment_list, segment_num):
    for value in segment_list:
        if value == segment_num:
            index_num = segment_numbers.index(value)
    return index_num

### ===================== EXECUTE CODE =========================== ###

# Find files in directory
cwd = os.getcwd()
files = os.listdir(cwd)

# Read files into dataframes
file_names = []
dataframes = []

for file in files:
    if "processed.csv" in file:
        data = pd.read_csv(file)
        dataframes.append(data)
        file_names.append(file[:-4])

# Create directory for results
plots_dir = cwd + "/size_plots"
isExist = os.path.exists(plots_dir)
if not isExist:
    os.makedirs(plots_dir)

# Create directory for temp files
temp_dir = cwd + "/temp"
isExist = os.path.exists(temp_dir)
if not isExist:
    os.makedirs(temp_dir)

# Check if entrainment files are present
entrainment_dir = cwd + "/entrainment-files"
isExistEntrainment = os.path.exists(entrainment_dir)
# Empty lists to collect all period and phase-locking files
period_files = []
phase_files = []
# Read and append period and phase-locking files
if isExistEntrainment:
    entrainment_files = os.listdir(entrainment_dir)
    for entrainment_file in entrainment_files:
        if "periods" in entrainment_file:
            period_files.append(entrainment_file)
        else:
            phase_files.append(entrainment_file)

## Start loop for processing the data
file_name_index = 0 #for file identification

for dataframe in dataframes:
    # Read values
    segment_ids = dataframe.iloc[:,0]
    segment_lengths = dataframe.iloc[:,1]
    segment_sds = dataframe.iloc[:,2]
    measurement_time = dataframe.iloc[:,3]
    # Turn measurement time into a list
    measurement_time = list(measurement_time.values)

    # Define empty lists for storage
    new_segment_ids = []
    exp_date = []
    samples = []
    sample_types = []
    segment_numbers = []
    # Collect identifying information for each entry
    for segment_id in segment_ids:
        segment_id = str(segment_id)
        segment_id = segment_id[:10] + "0" + segment_id[10:]
        new_segment_ids.append(segment_id)
        date = segment_id[:8]
        exp_date.append(date)
        sample = segment_id[9:14]
        samples.append(sample)
        sample_type = segment_id[15:16]
        sample_types.append(sample_type)
        segment_number = int(segment_id[-2:])
        segment_numbers.append(segment_number)


    # Normalize segment sizes to the size of segment 0
    normalized = [] #empty list for value storage
    # find index of segment 0 in list
    position = segment_index_finder(segment_numbers, 0)
    for length in segment_lengths:
        normalized_to0 = length / segment_lengths[position]
        normalized.append(normalized_to0)


    # Save normalized measurements in a new .csv file
    normalized_dataframe = pd.DataFrame(list(zip(new_segment_ids, normalized, measurement_time)),
                                        columns=['segment', 'normalized-length', 'measurement-time(min)'])
    normalized_dataframe.to_csv(cwd + "/" + file_names[file_name_index][:-9] + "normalized.csv", encoding='utf-8', index=False)

    # Save new processed .csv file with correct sample-id
    new_processed_dataframe = pd.DataFrame(list(zip(new_segment_ids, segment_lengths, segment_sds, measurement_time)),
                                           columns=['segment', 'average-length', 'standard-deviation',
                                                    'measurement-time(min)'])
    new_processed_dataframe.to_csv(cwd + "/" + file_names[file_name_index][:-9] + "processed_new.csv", encoding='utf-8',
                                   index=False)

    ### Add period information (if available)
    # Read period files into dataframes
    if isExistEntrainment:
        for period_file in period_files:
            if "DAPT" in period_file and "GR" in period_file:
                dapt_gr_data = pd.read_csv(entrainment_dir + "/" + period_file)
            elif "DMSO" in period_file and "GR" in period_file:
                ctrl_gr_data = pd.read_csv(entrainment_dir + "/" + period_file)
            elif "DAPT" in period_file and "AR" in period_file:
                dapt_ar_data = pd.read_csv(entrainment_dir + "/" + period_file)
            elif "DMSO" in period_file and "AR" in period_file:
                ctrl_ar_data = pd.read_csv(entrainment_dir + "/" + period_file)
            elif "DAPT" in period_file and "PR" in period_file:
                dapt_pr_data = pd.read_csv(entrainment_dir + "/" + period_file)
            elif "DMSO" in period_file and "PR" in period_file:
                ctrl_pr_data = pd.read_csv(entrainment_dir + "/" + period_file)

        # Create empty lists to store period information
        gr_periods = []
        ar_periods = []
        pr_periods = []

        # Locate period information and append to the column
        for new_segment_id, timepoint in zip(new_segment_ids, measurement_time):
            entrainment_id = new_segment_id[9:14] + "_" + new_segment_id[:8]
            # Assign NaN period value
            period_value = float('NaN')

            # Check global ROI dataframe first
            if entrainment_id in dapt_gr_data.columns:
                gr_period_dataframe = dapt_gr_data
                # Assign period to timepoints
                match = gr_period_dataframe[gr_period_dataframe['Time']*10 == timepoint]
                if not match.empty:
                    period_value = match[entrainment_id].values[0]
            elif entrainment_id in ctrl_gr_data.columns:
                gr_period_dataframe = ctrl_gr_data
                # Assign period to timepoints
                match = gr_period_dataframe[gr_period_dataframe['Time']*10 == timepoint]
                if not match.empty:
                    period_value = match[entrainment_id].values[0]
            
            gr_periods.append(period_value)

            # Assign NaN to anterior and posterior period values
            ar_period_value = float('NaN')
            pr_period_value = float('NaN')
            # Check anterior and posterior ROI dataframes next
            if entrainment_id in dapt_pr_data.columns:
                pr_period_dataframe = dapt_pr_data
                ar_period_dataframe = dapt_ar_data
                # Assign anterior periods to timepoints
                match_ar = ar_period_dataframe[ar_period_dataframe['Time'] * 10 == timepoint]
                if not match_ar.empty:
                    ar_period_value = match_ar[entrainment_id].values[0]
                # Assign posterior period to timepoints
                match_pr = pr_period_dataframe[pr_period_dataframe['Time'] * 10 == timepoint]
                if not match_pr.empty:
                    pr_period_value = match_pr[entrainment_id].values[0]
                pr_periods.append(period_value)
            elif entrainment_id in ctrl_pr_data.columns:
                pr_period_dataframe = ctrl_pr_data
                ar_period_dataframe = ctrl_ar_data
                # Assign anterior periods to timepoints
                match_ar = ar_period_dataframe[ar_period_dataframe['Time'] * 10 == timepoint]
                if not match_ar.empty:
                    ar_period_value = match_ar[entrainment_id].values[0]
                # Assign posterior period to timepoints
                match_pr = pr_period_dataframe[pr_period_dataframe['Time'] * 10 == timepoint]
                if not match_pr.empty:
                    pr_period_value = match_pr[entrainment_id].values[0]

            ar_periods.append(ar_period_value)
            pr_periods.append(pr_period_value)

        ### Add phase-locking information (if available)
        for phase_file in phase_files:
            if "GR" in phase_file:
                gr_locking = pd.read_csv(entrainment_dir + "/" + phase_file)
            elif "AR" in phase_file:
                ar_locking = pd.read_csv(entrainment_dir + "/" + phase_file)
            elif "PR" in phase_file:
                pr_locking = pd.read_csv(entrainment_dir + "/" + phase_file)

        # Define sample-id to look for
        locking_id = "W0" + file_names[file_name_index][10:13] + "_" + file_names[file_name_index][:8]

        # Check if sample is in the global ROI dataframe
        match_gr = gr_locking[gr_locking['sample-id'] == locking_id]
        if not match_gr.empty:
            locking_time = match_gr['timepoints'].values[0]
            gr_relative_time = measurement_time - locking_time
            gr_relative_time = list(gr_relative_time)
        else:
            gr_relative_time = []
            for i in range(len(measurement_time)):
                gr_relative_time.append(float('Nan'))

        # Check if sample is in anterior roi dataframe
        match_ar = ar_locking[ar_locking['sample-id'] == locking_id]
        if not match_ar.empty:
            ar_locking_time = match_ar['timepoints'].values[0]
            ar_relative_time = measurement_time - ar_locking_time
            ar_relative_time = list(ar_relative_time)
        else:
            ar_relative_time = []
            for i in range(len(measurement_time)):
                ar_relative_time.append(float('NaN'))

        # Check if sample is in posterior roi dataframe
        match_pr = pr_locking[pr_locking['sample-id'] == locking_id]
        if not match_pr.empty:
            pr_locking_time = match_pr['timepoints'].values[0]
            pr_relative_time = measurement_time - pr_locking_time
            pr_relative_time = list(pr_relative_time)
        else:
            pr_relative_time = []
            for i in range(len(measurement_time)):
                pr_relative_time.append(float('NaN'))

        # Save normalized measurements with attached period information in a new .csv file
        normalized_dataframe = pd.DataFrame(list(zip(new_segment_ids, normalized, measurement_time, gr_periods, ar_periods, pr_periods, gr_relative_time, ar_relative_time, pr_relative_time)),
                                            columns=['segment', 'normalized-length', 'measurement-time(min)', 'GR-period', 'AR-period', 'PR-period', 'GR-locking-time', 'AR-locking-time', 'PR-locking-time'])
        normalized_dataframe.to_csv(cwd + "/" + file_names[file_name_index][:-9] + "normalized_periods_phase.csv",
                                    encoding='utf-8', index=False)

        # Save new processed .csv file with correct sample-id and period information
        new_processed_dataframe = pd.DataFrame(
            list(zip(new_segment_ids, segment_lengths, segment_sds, measurement_time, gr_periods, ar_periods, pr_periods, gr_relative_time, ar_relative_time, pr_relative_time)),
            columns=['segment', 'average-length', 'standard-deviation', 'measurement-time(min)', 'GR-period', 'AR-period', 'PR-period', 'GR-locking-time', 'AR-locking-time', 'PR-locking-time'])
        new_processed_dataframe.to_csv(cwd + "/" + file_names[file_name_index][:-9] + "processed_new_periods_phase.csv",
                                       encoding='utf-8', index=False)


    file_name_index = file_name_index + 1 #continue name iteration

    # Define color code for plotting
    if sample_types[0] == 'D':
        graph_color = '#4DAF4A' # green (dapt)
    else:
        graph_color = '#984EA3' # purple (controls)

    # Plot individual samples and save figures
    plt.errorbar(segment_numbers, segment_lengths, segment_sds, color=graph_color, fmt='-o', linewidth=2, capsize=6)
    plt.xlabel('segment number')
    plt.ylim(10,150)
    plt.ylabel('segment length (microns)')
    plt.title(exp_date[0] + '_' + samples[0] + '_' + sample_types[0])
    segment_length_figure = plt.savefig(plots_dir + "/" + exp_date[0] + "_" + samples[0] + "_" + sample_types[0] + "_lengths-fig.png")
    segment_length_figure = plt.savefig(plots_dir + "/" + exp_date[0] + "_" + samples[0] + "_" + sample_types[0] + "_lengths-fig.pdf")
    plt.close(segment_length_figure)

    # Plot individual samples and save figures
    plt.errorbar(segment_numbers, normalized, color=graph_color, fmt='-o', linewidth=2, capsize=6)
    plt.xlabel('segment number')
    plt.ylim(0,1.5)
    plt.ylabel('segment length (normalized)')
    plt.title(exp_date[0] + '_' + samples[0] + '_' + sample_types[0])
    normalized_segment_length_figure = plt.savefig(
        plots_dir + "/" + exp_date[0] + "_" + samples[0] + "_" + sample_types[0] + "_normalized-lengths-fig.png")
    normalized_segment_length_figure = plt.savefig(
        plots_dir + "/" + exp_date[0] + "_" + samples[0] + "_" + sample_types[0] + "_normalized-lengths-fig.pdf")
    plt.close(normalized_segment_length_figure)

### Global plots

# Read files again so "processed_new.csv"s are included
files = os.listdir(cwd)

# Create empty dataframe to store combined data
combined_data = pd.DataFrame()

# Iterate through each .csv file and concatenate its contents
if isExistEntrainment:
    for file in files:
        if "processed_new_periods_phase.csv" in file:
            file_path = os.path.join(cwd, file)
            df = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, df], ignore_index=True)
else:
    for file in files:
        if "processed_new.csv" in file:
            file_path = os.path.join(cwd, file)
            df = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, df], ignore_index=True)

# Save concatenated dataframe to a new .csv file
combined_data.to_csv(temp_dir + "/" + "all_processed_lengths.csv", encoding='utf-8', index=False)


# Create empty dataframes to store samples by condition
combined_controls = pd.DataFrame()
combined_treatments = pd.DataFrame()

# Isolate segment ids in the combined data and sort measurements based on condition
segment_ids = combined_data.iloc[:,0]
for segment_id in segment_ids:
    if segment_id[15:16] == "D":
        row_index = int(combined_data[combined_data['segment'] == segment_id].index.values)
        combined_treatments = combined_treatments.append(combined_data.iloc[row_index,:], ignore_index=True)
    if segment_id[15:16] == "C":
        row_index = int(combined_data[combined_data['segment'] == segment_id].index.values)
        combined_controls = combined_controls.append(combined_data.iloc[row_index,:], ignore_index=True)
# Save to new .csv files
combined_controls.to_csv(temp_dir + "/" + "all_processed_control-samples.csv", encoding='utf-8', index=False)
combined_treatments.to_csv(temp_dir + "/" + "all_processed_treatment-samples.csv", encoding='utf-8', index=False)

# Concatenate normalized lengths dataframes
files = os.listdir(cwd) #find files

all_normalized = pd.DataFrame() #empty dataframe to store data

if isExistEntrainment:
    for file in files:
        if "normalized_periods_phase.csv" in file:
            file_path = os.path.join(cwd, file)
            df = pd.read_csv(file_path)
            all_normalized = pd.concat([all_normalized, df], ignore_index=True)
else:
    for file in files:
        if "normalized.csv" in file:
            file_path = os.path.join(cwd, file)
            df = pd.read_csv(file_path)
            all_normalized = pd.concat([all_normalized, df], ignore_index=True)

# Save concatenated dataframe to a new .csv file
all_normalized.to_csv(temp_dir + "/" + "all_normalized_lengths.csv", encoding='utf-8', index=False)

# Create empty dataframes to store normalized-length samples by condition
normalized_controls = pd.DataFrame()
normalized_treatments = pd.DataFrame()

# Isolate segment ids in the combined data and sort measurements based on condition
segment_ids = all_normalized.iloc[:,0]
for segment_id in segment_ids:
    if segment_id[15:16] == "D":
        row_index = int(all_normalized[all_normalized['segment'] == segment_id].index.values)
        normalized_treatments = normalized_treatments.append(all_normalized.iloc[row_index,:], ignore_index=True)
    if segment_id[15:16] == "C":
        row_index = int(all_normalized[all_normalized['segment'] == segment_id].index.values)
        normalized_controls = normalized_controls.append(all_normalized.iloc[row_index,:], ignore_index=True)
# Save to new .csv files
normalized_controls.to_csv(temp_dir + "/" + "all_normalized_control-samples.csv", encoding='utf-8', index=False)
normalized_treatments.to_csv(temp_dir + "/" + "all_normalized_treatment-samples.csv", encoding='utf-8', index=False)


# Read combined .csv files into dataframes
temp_files = os.listdir(temp_dir)

file_names = []
dataframes = []

for temp_file in temp_files:
    if "samples.csv" in temp_file and 'sorted' not in temp_file:
        data = pd.read_csv(temp_dir + "/" + temp_file)
        dataframes.append(data)
        file_names.append(temp_file[:-4])

file_names_index = 0

for dataframe in dataframes:
    # Read values
    segment_ids = dataframe.iloc[:,0]
    segment_lengths = dataframe.iloc[:,1]
    measurement_time = dataframe.iloc[:,2]

    # Define empty lists for storage
    exp_date = []
    samples = []
    unique_samples = []
    sample_types = []
    segment_numbers = []
    # Collect identifying information for each entry
    for segment_id in segment_ids:
        date = str(segment_id)[:8]
        exp_date.append(date)
        sample = str(segment_id)[9:14]
        unique_sample = str(segment_id)[:14]
        unique_samples.append(unique_sample)
        samples.append(sample)
        sample_type = str(segment_id)[15:16]
        sample_types.append(sample_type)
        segment_number = int(str(segment_id)[-2:])
        segment_numbers.append(segment_number)

    #Count N (number of experiments) and n (number of samples)
    different_dates = []
    for i in range(len(exp_date)):
        if len(different_dates) == 0:
            different_dates.append(exp_date[i])
        else:
            if exp_date[i] not in different_dates:
                different_dates.append(exp_date[i])
    N = len(different_dates)

    different_samples = []
    for i in range(len(unique_samples)):
        if len(different_samples) == 0:
            different_samples.append(unique_samples[i])
        else:
            if unique_samples[i] not in different_samples:
                different_samples.append(unique_samples[i])
    n = len(different_samples)

    if "control" in file_names[file_names_index]:
        graph_color = '#984EA3' # purple
        sample_title = 'CTRL samples'
    else:
        graph_color = '#4DAF4A' # green
        sample_title = 'entrained samples'

    if "normalized" in file_names[file_names_index]:
        low_limit = 0
        high_limit = 1.5
        y_label = 'segment length (normalized to somite 0)'
    else:
        low_limit = 10
        high_limit = 150
        y_label = 'segment length (microns)'

    # Create violin plot
    fig, ax = plt.subplots()
    sns.set_theme(style='whitegrid')
    ax.set(ylim=(low_limit, high_limit))
    plt.ylabel(y_label)
    plt.xlabel('segment number')
    sns.violinplot(x=segment_numbers, y=segment_lengths, data=dataframe, orient='v', native_scale=True,
                   color=graph_color, inner='point')
    plt.title('Segment size measurements for ' + sample_title + ': n = ' + str(n) + ', N = ' + str(N))
    save_violin_plot = plt.savefig(plots_dir + "/" + file_names[file_names_index][4:] + "-violinplot.png")
    save_violin_plot = plt.savefig(plots_dir + "/" + file_names[file_names_index][4:] + "-violinplot.pdf")
    #plt.show()
    plt.close(save_violin_plot)

    # Create box plot
    fig, ax = plt.subplots()
    sns.boxplot(x=segment_numbers, y=segment_lengths, data=dataframe, orient='v', color=graph_color, saturation=0.5, linecolor='black')
    sns.stripplot(x=segment_numbers, y=segment_lengths, data=dataframe, orient='v', color=graph_color, alpha=0.7, linewidth=0.1)
    sns.set_theme(style='whitegrid')
    ax.set(ylim=(low_limit, high_limit))
    plt.ylabel(y_label)
    plt.xlabel('segment number')
    plt.title('Segment size measurements for ' + sample_title + ': n = ' + str(n) + ', N = ' + str(N))
    save_boxplot = plt.savefig(plots_dir + "/" + file_names[file_names_index][4:] + "-boxplot.png")
    save_boxplot = plt.savefig(plots_dir + "/" + file_names[file_names_index][4:] + "-boxplot.pdf")
    #plt.show()
    plt.close(save_boxplot)

    file_names_index = file_names_index + 1

### ===================== MAKE DATA TIDY =========================== ###

# Read files in temp directory into dataframes
cwd = os.getcwd()
temp_dir = cwd + '/temp'
files = os.listdir(temp_dir)

# Create directory for sorted and tidy files
tidy_dir = cwd + "/tidy"
isExist = os.path.exists(tidy_dir)
if not isExist:
    os.makedirs(tidy_dir)

file_names = []
dataframes = []

for file in files:
    if 'all' in file:
        data = pd.read_csv(temp_dir + '/' + file)
        dataframes.append(data)
        file_names.append(file[:-4])

# Find and read entrainment file
cwd_files = os.listdir(cwd)
period_info = []
for doc in cwd_files:
    if 'entrainment_period' in doc:
        data = pd.read_csv(cwd + '/' + doc)
        period_info.append(data)

period_info = period_info[0]  # make period info into dataframe

file_name_index = 0

for dataframe in dataframes:
    # Sort dataframe based on segment-id and save to a new file
    segment_sorted_dataframe = dataframe.sort_values('segment')
    segment_sorted_dataframe.to_csv(temp_dir + "/" + "sorted-by-segment" + file_names[file_name_index][3:] + ".csv",
                                    encoding='utf-8', index=False)

    # Sort dataframe based on measurement time and save to a new file
    time_sorted_dataframe = dataframe.sort_values('measurement-time(min)')
    time_sorted_dataframe.to_csv(temp_dir + "/" + "sorted-by-time" + file_names[file_name_index][3:] + ".csv",
                                 encoding='utf-8', index=False)

    file_name_index = file_name_index + 1

# List files again and find sorted dataframes
files = os.listdir(temp_dir)

file_names = []
dataframes = []

for file in files:
    if 'sorted-by-segment' in file:
        data = pd.read_csv(temp_dir + '/' + file)
        dataframes.append(data)
        file_names.append(file[:-4])

file_name_index = 0

for dataframe in dataframes:
    #Find segment ids and segment lengths
    segment_ids = dataframe.iloc[:, 0]
    segment_lengths = dataframe.iloc[:, 1]

    # Find the maximum number of segments in the table
    segment_number_list = []
    for segment_id in segment_ids:
        segment_number = int(segment_id[-2:])
        segment_number_list.append(segment_number)
    max_num = max(segment_number_list)

    # Create dataframe with tidy structure
    tidy = pd.DataFrame(columns=['experiment-date',
                                 'period',
                                 'condition',
                                 'sample-id'])
    for i in range(max_num + 1):
        tidy.insert(len(tidy.axes[1]),'somite-'+ str(i), [])

    # Find all unique samples and load data on tidy spreadsheet
    for i in range(len(segment_ids)):
        if i == 0:
            tidy.loc[i, 'experiment-date'] = segment_ids[i][:8]
            tidy.loc[i, 'sample-id'] = segment_ids[i][9:14]
            tidy.loc[i, 'condition'] = segment_ids[i][15:16]
            tidy.loc[i, 'period'] = int(period_info.loc[period_info[period_info['date'] == int(segment_ids[i][:8])].index.values]['period'])
            for num in range(max_num + 1):
                if num == int(segment_ids[i][-2:]):
                    tidy.loc[i, 'somite-' + str(num)] = dataframe.iloc[i, 1]
        else:
            if segment_ids[i][:14] == segment_ids[i -1][:14]:
                for num in range(max_num + 1):
                    if num == int(segment_ids[i][-2:]):
                        tidy.loc[len(tidy) - 1, 'somite-' + str(num)] = dataframe.iloc[i, 1]
            else:
                tidy.loc[len(tidy), 'experiment-date'] = segment_ids[i][:8]
                tidy.loc[len(tidy) - 1, 'sample-id'] = segment_ids[i][9:14]
                tidy.loc[len(tidy) - 1, 'condition'] = segment_ids[i][15:16]
                tidy.loc[len(tidy) - 1, 'period'] = int(period_info.loc[period_info[period_info['date'] == int(segment_ids[i][:8])].index.values]['period'])
                for num in range(max_num + 1):
                    if num == int(segment_ids[i][-2:]):
                        tidy.loc[len(tidy) - 1, 'somite-' + str(num)] = dataframe.iloc[i, 1]

    # Save tidy dataframe to a new .csv file
    tidy.to_csv(tidy_dir + "/" + "tidy" + file_names[file_name_index][17:] + ".csv", encoding='utf-8', index=False)

    file_name_index = file_name_index + 1
