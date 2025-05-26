# Import libraries
import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

### ============================== DEFINE FUNCTIONS ========================= ###

# Define smoothing function (Savitzky-Golay filter)
window_global = []
order_global = []

def smooth(a):
    window = 31
    order = 6
    window_global.append(window)
    order_global.append(order)
    savgol = savgol_filter(a, window, order)
    return savgol

# Define function to invert curve, find peaks, and return as troughs
def find_troughs(function):
    inverted_function = function * (-1)
    troughs = find_peaks(inverted_function)
    return troughs

# Define function to calculate segment length
def segment_length(distance_1, distance_2):
    if pd.isna(distance_1) == True:
        length = float('NaN')
    else:
        if pd.isna(distance_2) == True:
            length = float('NaN')
        else:
            if distance_2 > distance_1:
                length = distance_2 - distance_1
            else:
                length = distance_1 - distance_2
    return length

### ===================== EXECUTE CODE =========================== ###

# Find files in directory
cwd = os.getcwd()
files = os.listdir(cwd)

# Read files into dataframes
file_names = []
dataframes = []

for file in files:
    if ".csv" in file:
        data = pd.read_csv(file)
        dataframes.append(data)
        file_names.append(file[:-4])



# Create directory for results
plots_dir = cwd + "/plots"
isExist = os.path.exists(plots_dir)
if not isExist:
    os.makedirs(plots_dir)

half_plots_dir = cwd + "/half_plots"
isExist = os.path.exists(half_plots_dir)
if not isExist:
    os.makedirs(half_plots_dir)

sg_half_plots_dir = cwd + "/sg_half_plots"
isExist = os.path.exists(sg_half_plots_dir)
if not isExist:
    os.makedirs(sg_half_plots_dir)

csv_results_dir = cwd + "/results"
isExist = os.path.exists(csv_results_dir)
if not isExist:
    os.makedirs(csv_results_dir)

# Index storage for iteration
file_name_index = 0
# Values storage
raw_troughs_1 = []
raw_troughs_2 = []
raw_troughs_half1 = []
raw_troughs_half2 = []
savgol_troughs_half1 = []
savgol_troughs_half2 = []
raw_distances_1 = []
raw_distances_2 = []
raw_distances_half1 = []
raw_distances_half2 = []
savgol_distances_half1 = []
savgol_distances_half2 = []
raw_lengths = []
raw_half_lengths = []
savgol_half_lengths = []

# Iterate across each dataframe
for dataframe in dataframes:
    # Read distance and intensity values
    distance = dataframe.iloc[:,0]
    intensity = dataframe.iloc[:,1]
    # Smooth intensity curve with Savitzky-Golay filter
    savgol_smoothened = smooth(intensity)
    ## Divide curves in two halves
    # divide raw curve
    half_point_index = len(distance) / 2
    if type(half_point_index) is int:
        half_point_index = half_point_index
    else:
        half_point_index = int(half_point_index)
    half1_distance = dataframe.iloc[:half_point_index, 0]
    half2_distance = dataframe.iloc[half_point_index:, 0]
    half1_intensity = dataframe.iloc[:half_point_index, 1]
    half2_intensity = dataframe.iloc[half_point_index:, 1]
    # divide smoothened curve
    half1_savgol_intensity = savgol_smoothened[:half_point_index]
    half2_savgol_intensity = savgol_smoothened[half_point_index:]

    ## Find troughs
    # troughs for the raw, undivided curve
    raw_troughs, _ = find_troughs(intensity)
    # Find troughs for each half in the unsmoothed curve
    raw1_troughs, _ = find_troughs(half1_intensity)
    raw2_troughs, _ = find_troughs(half2_intensity)
    # Find troughs on each half of the SG-smoothed curve
    savgol1_troughs, _ = find_troughs(half1_savgol_intensity)
    savgol2_troughs, _ = find_troughs(half2_savgol_intensity)
    # Sort raw troughs and locate minima
    raw_troughs_list = []
    for raw_trough in intensity[raw_troughs]:
        raw_troughs_list.append(raw_trough)
    sorted_raw_troughs = sorted(raw_troughs_list)
    # Check how many troughs were found
    raw_troughs_num = len(sorted_raw_troughs)
    # If there is at least two troughs in the raw, undivided curve, proceed normally
    if raw_troughs_num > 1:
        raw_trough_1 = sorted_raw_troughs[0]
        raw_trough_2 = sorted_raw_troughs[1]
        raw_troughs_1.append(raw_trough_1)
        raw_troughs_2.append(raw_trough_2)
        # Find troughs position in dataframe
        # Find all instances of the trough values
        raw_trough_1_series = intensity[intensity == raw_trough_1]
        raw_trough_2_series = intensity[intensity == raw_trough_2]
        # Check how many elements exist with the same value as the first trough for the raw curve
        if len(raw_trough_1_series) == 1:
            raw_trough_1_row = int(dataframe[dataframe['Gray_Value'] == raw_trough_1].index.values)
        else:  # escape mechanism for more than one trough with the same value
            print('More elements with first trough value in ' + file_names[file_name_index])
            print(raw_trough_1_series)
            raw_trough_1_row = input('Enter correct index value: ')
            raw_trough_1_row = int(raw_trough_1_row)
        # Check how many elements exist with the same value as the second trough for the raw curve
        if len(raw_trough_2_series) == 1:
            raw_trough_2_row = int(dataframe[dataframe['Gray_Value'] == raw_trough_2].index.values)
        else:  # escape mechanism for more than one trough with the same value
            print('More elements with second trough value in ' + file_names[file_name_index])
            print(raw_trough_2_series)
            raw_trough_2_row = input('Enter correct index value: ')
            raw_trough_2_row = int(raw_trough_2_row)
        # Find corresponding distance (in microns)
        raw_distance_1 = dataframe.iloc[raw_trough_1_row, 0]
        raw_distance_2 = dataframe.iloc[raw_trough_2_row, 0]
        raw_distances_1.append(raw_distance_1)
        raw_distances_2.append(raw_distance_2)
    # If there is only one trough in the raw, undivided curve
    else:
        raw_trough_1 = sorted_raw_troughs[0]
        raw_trough_2 = float('NaN')
        raw_troughs_1.append(raw_trough_1)
        raw_troughs_2.append(raw_trough_2)
        # Find troughs position in dataframe
        raw_trough_1_series = intensity[intensity == raw_trough_1]
        # Check how many elements exist with the same value as the first trough for the raw curve
        if len(raw_trough_1_series) == 1:
            raw_trough_1_row = int(dataframe[dataframe['Gray_Value'] == raw_trough_1].index.values)
        else:  # escape mechanism for more than one trough with the same value
            print('More elements with first trough value in ' + file_names[file_name_index])
            print(raw_trough_1_series)
            raw_trough_1_row = input('Enter correct index value: ')
            raw_trough_1_row = int(raw_trough_1_row)
        # Find corresponding distance (in microns)
        raw_distance_1 = dataframe.iloc[raw_trough_1_row, 0]
        raw_distance_2 = float('NaN')
        raw_distances_1.append(raw_distance_1)
        raw_distances_2.append(raw_distance_2)
        # Print message
        print('Only one trough found in the raw curve for ' + file_names[file_name_index])
        # Add row to log
        ############ CODE FOR LOG ##############

    # Calculate segment length using troughs found in the raw, undivided curve
    raw_length = segment_length(raw_distance_1, raw_distance_2)
    raw_lengths.append(raw_length)

    ## Sort troughs for each half and locate the minima
    # troughs for each half of the raw curve
    raw1_troughs_list = []
    for raw1_trough in half1_intensity[raw1_troughs]:
        raw1_troughs_list.append(raw1_trough)
    sorted_raw1_troughs = sorted(raw1_troughs_list)
    # Check if troughs have been found in the first half of the raw curve
    if len(raw1_troughs_list) > 0:
        raw1_trough_1 = sorted_raw1_troughs[0]
        raw_troughs_half1.append(raw1_trough_1)
    # Escape mechanism if no trough is found in the first half of the raw curve
    else:
        print('No trough found in the first half of the raw curve for ' + file_names[file_name_index])
        # Find minimum with numpy instead
        raw1_trough_1_index = np.argmin(half1_intensity)  # np.argmin returns the index
        raw1_trough_1 = half1_intensity[raw1_trough_1_index]
        raw_troughs_half1.append(raw1_trough_1)
        # Add row to log
        # ('No trough found in the first half of the raw curve for ' + file_names[file_name_index] + '. Minimum calculated with numpy.')
    raw2_troughs_list = []
    for raw2_trough in half2_intensity[raw2_troughs + half_point_index]:
        raw2_troughs_list.append(raw2_trough)
    sorted_raw2_troughs = sorted(raw2_troughs_list)
    # Check if troughs have been found in the second half of the raw curve
    if len(raw2_troughs_list) > 0:
        raw2_trough_1 = sorted_raw2_troughs[0]
        raw_troughs_half2.append(raw2_trough_1)
    # Escape mechanism if no trough is found in the second half of the raw curve
    else:
        print('No trough found in the second half of the raw curve for ' + file_names[file_name_index])
        # Find minimum with numpy instead
        raw2_trough_1_index = np.argmin(half2_intensity)  # np.argmin returns the index
        raw2_trough_1 = half2_intensity[raw2_trough_1_index]
        raw_troughs_half2.append(raw2_trough_1)
        # Add row to log
        # ('No trough found in the second half of the raw curve for ' + file_names[file_name_index] + '. Minimum calculated with numpy.')

    # troughs for each half of the smoothened curve
    savgol1_troughs_list = []
    for savgol1_trough in half1_savgol_intensity[savgol1_troughs]:
        savgol1_troughs_list.append(savgol1_trough)
    sorted_savgol1_troughs = sorted(savgol1_troughs_list)
    # Check if troughs have been found in the first half of the smoothed curve
    if len(savgol1_troughs_list) > 0:
        savgol1_trough_1 = sorted_savgol1_troughs[0]
        savgol_troughs_half1.append(savgol1_trough_1)
    # Escape mechanism if no trough is found in the first half of the smoothed curve
    else:
        print('No trough found in the first half of the smoothed curve for ' + file_names[file_name_index])
        # Find minimum with numpy instead
        savgol1_trough_1_index = np.argmin(half1_savgol_intensity) # np.argmin returns the index
        savgol1_trough_1 = half1_savgol_intensity[savgol1_trough_1_index]
        savgol_troughs_half1.append(savgol1_trough_1)
        # Add row to log
        # ('No trough found in the first half of the smoothed curve for ' + file_names[file_name_index] + '. Minimum calculated with numpy.')

    savgol2_troughs_list = []
    for savgol2_trough in half2_savgol_intensity[savgol2_troughs]:
        savgol2_troughs_list.append(savgol2_trough)
    sorted_savgol2_troughs = sorted(savgol2_troughs_list)
    # Check if troughs have been found in the second half of the smoothed curve
    if len(savgol2_troughs_list) > 0:
        savgol2_trough_1 = sorted_savgol2_troughs[0]
        savgol_troughs_half2.append(savgol2_trough_1)
    # Escape mechanism if no trough is found in the second half of the smoothed curve
    else:
        print('No trough found in the second half of the smoothed curve for ' + file_names[file_name_index])
        # Find minimum with numpy instead
        savgol2_trough_1_index = np.argmin(half2_savgol_intensity) # np.argmin returns the index
        savgol2_trough_1 = half2_savgol_intensity[savgol2_trough_1_index]
        savgol_troughs_half2.append(savgol2_trough_1)
        # Add row to log
        # ('No trough found in the second half of the smoothed curve for ' + file_names[file_name_index] + '. Minimum calculated with numpy.')

    # Map troughs on original curves
    # Find all instances of the trough values
    raw1_trough_1_series = intensity[intensity == raw1_trough_1]
    raw2_trough_1_series = intensity[intensity == raw2_trough_1]
    # Check how many elements exist with the same value as the trough for the first half of the curve
    if len(raw1_trough_1_series) == 1:
        raw1_trough_1_row = int(dataframe[dataframe['Gray_Value'] == raw1_trough_1].index.values)
    else: #escape mechanism for more than one trough with the same value
        print('More elements with trough value in the first half of ' + file_names[file_name_index])
        print(raw1_trough_1_series)
        raw1_trough_1_row = input('Enter correct index value: ')
        raw1_trough_1_row = int(raw1_trough_1_row)
    # Check how many elements exist with the same value as the trough for the second half of the curve
    if len(raw2_trough_1_series) == 1:
        raw2_trough_1_row = int(dataframe[dataframe['Gray_Value'] == raw2_trough_1].index.values)
    else: #escape mechanism for more than one trough with the same value
        print('More elements with trough value in the second half of ' + file_names[file_name_index])
        print(raw2_trough_1_series)
        raw2_trough_1_row = input('Enter correct index value: ')
        raw2_trough_1_row = int(raw2_trough_1_row)
    # Find corresponding distance (in microns)
    raw1_distance_1 = dataframe.iloc[raw1_trough_1_row, 0]
    raw2_distance_1 = dataframe.iloc[raw2_trough_1_row, 0]
    raw_distances_half1.append(raw1_distance_1)
    raw_distances_half2.append(raw2_distance_1)
    # Calculate segment length for troughs found in each half of the raw curve
    half_length = segment_length(raw1_distance_1, raw2_distance_1)
    raw_half_lengths.append(half_length)
    # Map and find corresponding distance for smoothened curve
    savgol1_trough_1_index = []
    savgol2_trough_1_index = []
    for i in range(len(savgol_smoothened)):
        if savgol_smoothened[i] == savgol1_trough_1:
            savgol1_trough_1_index.append(i)
        if savgol_smoothened[i] == savgol2_trough_1:
            savgol2_trough_1_index.append(i)
    savgol1_trough_1_x = distance[savgol1_trough_1_index].values[0]
    savgol2_trough_1_x = distance[savgol2_trough_1_index].values[0]
    savgol_distances_half1.append(savgol1_trough_1_x)
    savgol_distances_half2.append(savgol2_trough_1_x)
    # Calculate segment length for troughs found in each half of the savgol smoothed curve
    sg_half_length = segment_length(savgol1_trough_1_x, savgol2_trough_1_x)
    savgol_half_lengths.append(sg_half_length)

    # Plots
    # Plot raw curve with somite boundaries
    plt.plot(distance, intensity, label='intensity profile')
    plt.plot(raw_distance_1, raw_trough_1, '*', label='anterior boundary')
    plt.plot(raw_distance_2, raw_trough_2, '*', label='posterior boundary')
    plt.legend(loc='best')
    plt.xlabel('distance (microns)')
    plt.ylabel('intensity')
    plt.title(file_names[file_name_index])
    raw_boundary_figure = plt.savefig(plots_dir + "/" + file_names[file_name_index] + "_raw-boundaries.png")
    plt.close(raw_boundary_figure)
    ## Plot half-mapped points on original curves and save
    # raw curve (halves)
    plt.plot(distance, intensity, label='intensity profile')
    plt.plot(raw1_distance_1, raw1_trough_1, '*', label='anterior boundary')
    plt.plot(raw2_distance_1, raw2_trough_1, '*', label='posterior boundary')
    plt.legend(loc='best')
    plt.xlabel('distance (microns)')
    plt.ylabel('intensity')
    half_raw_boundary_figure = plt.savefig(half_plots_dir + "/" + file_names[file_name_index] + "_raw-boundaries.png")
    plt.close(half_raw_boundary_figure)
    # smoothened curve (halves)
    plt.plot(distance, savgol_smoothened, label='smoothed intensity')
    plt.plot(savgol1_trough_1_x, savgol1_trough_1, '*', label='anterior boundary (SG)')
    plt.plot(savgol2_trough_1_x, savgol2_trough_1, '*', label='posterior boundary (SG)')
    plt.xlabel('distance (microns)')
    plt.ylabel('intensity')
    plt.legend(loc='best')
    sg_half_boundaries_figure = plt.savefig(sg_half_plots_dir + "/" + file_names[file_name_index] + "_SG-boundary.png")
    plt.close(sg_half_boundaries_figure)


    file_name_index = file_name_index + 1

# Get file name with date and sample number
file_name_example = file_names[0]
date_sample = file_name_example[:15]

### Create new comprehensive dataframe for all lengths results
lengths_results = pd.DataFrame(list(zip(file_names, raw_lengths, raw_half_lengths, savgol_half_lengths)),
               columns =['segment', 'length-raw', 'length-raw-half', 'length-sg-half'])

sorted_lengths_results = lengths_results.sort_values('segment')


# Save dataframe to .csv
sorted_lengths_results.to_csv(csv_results_dir + "/" + date_sample[:-2] + "_segment-lengths.csv", encoding='utf-8', index=False)
