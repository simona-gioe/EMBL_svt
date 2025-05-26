# Import libraries
import os
import zipfile
import pandas as pd


### =========================== DEFINE FUNCTIONS ========================= ###

def extract_rois(zip_file_path, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Open the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all files with the .roi extension
        roi_files = [name for name in zip_ref.namelist() if name.endswith('.roi')]

        for roi_file in roi_files:
            # Extract the .roi files from the ZIP archive
            zip_ref.extract(roi_file, output_directory)

    return roi_files


### =========================== EXTRACT AND ANALYSE ROIs ========================= ###

# Create directory for results
cwd = os.getcwd()

results_dir = cwd + "/results"
isExist = os.path.exists(results_dir)
if not isExist:
    os.makedirs(results_dir)


z_stack_num = 7  # number of z-stacks in the imaging data

if __name__ == "__main__":
    files_zip = [file for file in os.listdir(cwd) if file.endswith('.zip')]
    for file_zip in files_zip:
        zip_file_path = cwd + '/' + file_zip
        output_directory = file_zip[:-4]
        roi_files = extract_rois(zip_file_path, output_directory)

        # List all (and only the) slices in the rois
        slices = []
        for roi in roi_files:
            slice = roi[:4]
            slices.append(int(slice))

        # Extract time slices
        time_slices = []
        for slice in slices:
            quotient = slice // z_stack_num
            if slice % z_stack_num == 0:
                time_slices.append(quotient)
            else:
                time_slices.append(quotient + 1)

        # Check number of time slices
        while len(time_slices) % 3 == 0: # this means that there are three ROIs for each measurement
            break
        else:
            print('!!! The number of ROIs for file ' + output_directory + ' is: ' + str(len(time_slices)) +
                  '. Double check for missing time information !!!')

        # Extract timepoint from timeslice
        counter = []
        timepoints = []
        for time_slice in time_slices:
            counter.append(time_slice)
            while len(counter) < 3:
                break
            if len(counter) == 3:
                if counter[0] == counter[1] and counter[1] == counter[2]:
                    timepoints.append((counter[0] - 1) * 10)
                else:
                    if counter[0] == counter[1]:
                        timepoints.append((counter[0] - 1) * 10)
                    elif counter[1] == counter[2]:
                        timepoints.append((counter[1] - 1) * 10)
                    elif counter[0] == counter[2]:
                        timepoints.append((counter[0] - 1) * 10)
                    else:
                        print('In file ' + output_directory + ':')
                        print('These three time slices are not the same:')
                        print(counter)
                        manual_input_time = input('Please enter the correct time slice manually: ')
                        manual_input_time = int(manual_input_time)
                        timepoints.append((manual_input_time - 1) * 10)
                counter = []

        # Count how many somites the timepoints correspond to
        somite_num = []
        for i in range(len(timepoints)):
            somite_num.append(i)

        # Create a dataframe with these results
        measurement_times = pd.DataFrame(list(zip(somite_num, timepoints)),
                                             columns=['somite-number', 'time(min)'])
        # Save the dataframe to a .csv file
        measurement_times.to_csv(results_dir + '/' + file_zip[:14] + 'ROIs-timepoints.csv', encoding='utf-8', index=False)