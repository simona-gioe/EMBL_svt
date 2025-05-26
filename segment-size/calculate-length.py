# Import libraries
import os
import pandas as pd
import statistics

### ===================== EXECUTE CODE =========================== ###

# Find files in directory
cwd = os.getcwd()
files = os.listdir(cwd)

# Read files into dataframes
file_names = []
dataframes = []

for file in files:
    if "_lengths.csv" in file:
        data = pd.read_csv(file)
        dataframes.append(data)
        file_names.append(file[:-4])

# print(file_names)
# print(dataframes)

# Index storage for iteration
file_name_index = 0

# Iterate across each file
for dataframe in dataframes:
    # Sort dataframe
    dataframe = dataframe.sort_values('segment')
    # print(file_names[file_name_index])
    # print(dataframe)
    # Values storage
    segment = []
    average = []
    standard_deviation = []
    # Read segment ID and length values
    segment_ids = dataframe.iloc[:, 0]
    length_values = dataframe.iloc[:, 1]
    # Store length values in a list in the new sorted order
    length_values_list = []
    for length_value in length_values:
        length_values_list.append(length_value)
    # print(length_values_list)
    # print(segment_ids)

    # Iterate through segment ID names and find compatible ones
    segment_length_temp = []  # updated list with segment length values relating to same segment
    counter = 0
    for segment_id in segment_ids:
        segment_name = segment_id[:18]
        # print(segment_name)
        # Store unique segment ID for results column
        if len(segment) == 0:
            segment.append(segment_name)
            segment_length_temp.append(length_values_list[counter])
        else:
            if segment_name not in segment:
                segment.append(segment_name)
                segment_length_temp = []
                segment_length_temp.append(length_values_list[counter])
            else:
                segment_length_temp.append(length_values_list[counter])
        if len(segment_length_temp) == 3:
            # Calculate average and collect result
            average_length = sum(segment_length_temp) / len(segment_length_temp)
            average.append(average_length)
            # Calculate standard deviation and collect result
            st_dev = statistics.pstdev(segment_length_temp)
            standard_deviation.append(st_dev)

        counter = counter + 1

    # Identify timepoint filename
    timepoint_filename = file_names[file_name_index][:-7] + "ROIs-timepoints.csv"
    # Create empty list to store timepoints
    timepoints = []
    # Create list of timepoints for the appropriate file
    for file in files:
        if timepoint_filename in file:
            timepoint_dataframe = pd.read_csv(file)
            timepoint_values = timepoint_dataframe.iloc[:, 1]
            for timepoint_value in timepoint_values:
                timepoints.append(timepoint_value)

    # Check whether the number of segments, lengths, standard deviations and timepoints is the same
    while len(segment) == len(average) and len(average) == len(timepoints) and len(timepoints) == len(segment):
        break
    else:
        print('In file ' + file_names[file_name_index][:-8] + ', there is a discrepancy in the amount of segments, '
                                                              'measurements, and timepoints')
        print('number of segments: ' + str(len(segment)))
        print('number of measurements: ' + str(len(average)))
        print('number of timepoints: ' + str(len(timepoints)))
        print('!!! DOUBLE CHECK !!!')

    ### Create new comprehensive dataframe for all lengths results
    results = pd.DataFrame(list(zip(segment, average, standard_deviation, timepoints)),
                           columns=['segment', 'average-length', 'standard-deviation', 'measurement-time(min)'])
    # Sort results dataframe by segment number
    sorted_results = results.sort_values('segment')

    # Save dataframe to .csv
    sorted_results.to_csv(cwd + "/" + file_names[file_name_index] + "_processed.csv", encoding='utf-8', index=False)

    file_name_index = file_name_index + 1
