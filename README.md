# EMBL_svt (documentation in progress)

This repository contains a collection of scripts supporting image and data analysis for my PhD work in the Aulehla Group at EMBL Heidelberg. Aside from the [image concatenation script](https://github.com/simona-gioe/EMBL_svt/blob/main/concatenate/Concatenate_Images.py), which was written and developed by Aliaksandr Halavatyi as part of a previous project, all other scripts in this repository were developed by me, and used to analyse data extracted from imaging datasets of mouse intact presomitic mesoderm. The results of this work will be available [here](https://archiv.ub.uni-heidelberg.de/volltextserver/37071/) from July 30th 2026.

## Timeseries concatenation

The [Concatenate_Images.py](https://github.com/simona-gioe/EMBL_svt/blob/main/concatenate/Concatenate_Images.py) script was written by Aliaksandr Halavatyi to concatenate multi-position live imaging time series. Information on dependencies (requirements), input and output formats, and instructions for running this script are documented at the beginning of the file.

## Period evolution analysis

This script was developed to align imaging samples based on their oscillations period evolution dynamics along the anteroposterior axis of the presomitic mesoderm. Period evolution data was extracted by [wavelet analysis](https://github.com/PGLSanchez/EMBL_OscillationsAnalysis/blob/master/EntrainmentAnalysis/SCRIPT-EntrainmentAnalysis.ipynb) performed on raw intensity data extracted from [intensity kymographs](https://git.embl.org/grp-cba/tail-analysis/-/blob/main/analysis_workflow.md?ref_type=heads).

**Input files**

The expected input for this script is period evolution data stored in CSV time series datasets. These can be generated as output files from the [EntrainmnetAnalysis script](https://github.com/PGLSanchez/EMBL_OscillationsAnalysis/blob/master/EntrainmentAnalysis/SCRIPT-EntrainmentAnalysis.ipynb) linked above.
Files are only processed if their filename contains "periods". The script then categorizes the data automatically based on two keywords in the filename:
* the condition is labeled as "DAPT" if "DAPT" is in the filename, and "CTRL" in any other case
* the region of interest (ROI) is labeled as "arrival" if "AR" is in the filename, and "origin" in any other case

The first column of each CSV file is expected to be labeled "Time", and it should contain integer values representing imaging frames (time points).
Each additional column should then correspond to a separate experimental sample: column names should be unique sample identifiers, and the data values should be numeric (floating point values) and represent the oscillation period of that sample at each timepoint.

*CSV structure example*

| Time | W0001_YYYYMMDD | W0002_YYYYMMDD | W0003_YYYYMMDD |
| ---- | -------------- | -------------- | -------------- |
| 0    | 140.1          | 143.01         | 144.20         |
| 1    | 140.12         | 142.01         | 140.50         |
| 2    | 140.2          | 141.01         | 144.20         |

<br />

**Output files**

Two output files will be stored in a newly-created `/results` folder inside the working directory:
* `AP_period_comparison.csv` contains the full summary of arrival/origin period comparisons between control and treated samples, grouped by 200-minute time bins
* `AP_period_comparison_outliers.csv` contains a filtered subset of the first file, highlighting treated samples with unusually high ratios relative to the control distribution

<br />

**Instructions for execution**
1. Download [ap-period-stats.py](https://github.com/simona-gioe/EMBL_svt/blob/main/period-evolution/ap-period-stats.py) and place it in the same directory as your input files
2. From the terminal, navigate to the correct directory and activate a suitable conda environment
    * You can install a virtual environment for the execution of this script by downloading [period-environment.yml](https://github.com/simona-gioe/EMBL_svt/blob/main/environments/period-environment.yml) to your home directory and running the snippet below on the terminal

      ```
      conda env create -f period-environment.yml
      ```
      After creating the environment, you can activate it by running

      ```
      conda activate period-environment
      ```
    
    * If you want to create your own environment, you can consult [period-requirements.txt](https://github.com/simona-gioe/EMBL_svt/blob/main/period-requirements.txt) for a list of dependencies
    * The full conda environment used to run this code to generate outputs for my PhD thesis is stored [here](https://github.com/simona-gioe/EMBL_svt/blob/main/full-environment.yml) for archival purposes
3. After activating the virtual environment, you can run the script by copy-pasting and executing this command on the terminal:
   ```
   python ap-period-stats.py
   ```


## Segment size analysis

This series of scripts was developed to quantify and analyse nascent somite length measurements acquired from brightfield imaging timeseries.

Detailed information on how the length measurements need to be acquired is available in section *5.5.4 Acquisition of somite length measurements* in the [published thesis](https://archiv.ub.uni-heidelberg.de/volltextserver/37071/). In brief, nascent somites can be measured by tracing three 1-pixel-wide ROI lines across the somite width, on the clearest Z-slice of the brightfield channel. For every ROI line, the intensity profile should then be saved as a CSV file. All ROI lines for a sample should also be saved together by adding them to the Fiji ROI manager as they are traced and then exporting the entire ROI set as a ZIP file.
It is recommended to lightly smooth the image prior to ROI line tracing by applying a Gaussian filter (σ= 0.5-1.7). The workflow assumes timelapse datasets acquired every 10 minutes with 7 Z-slices per time point. To use the workflow with data acquired with different settings, these values should be changed in [extract_roi.py](https://github.com/simona-gioe/EMBL_svt/blob/main/segment-size/extract-roi.py). ROI lines are expected to be traced on files containing the brightfield channel only.


**Conda environment**

All scripts for segment size analysis run within the same virtual environment.

* You can install a virtual environment for the execution of this script by downloading [segment-size-environment.yml](https://github.com/simona-gioe/EMBL_svt/blob/main/environments/segment-size-environment.yml) to your home directory and running the snippet below on the terminal

   ```
   conda env create -f segment-size-environment.yml
   ```
   After creating the environment, you can activate it by running
   ```
   conda activate segment-size-environment
   ```
    
* If you want to create your own environment, you can consult [segment-size-requirements.txt](https://github.com/simona-gioe/EMBL_svt/blob/main/segment-size-requirements.txt) for a list of dependencies
  
* The full conda environment used to run this code to generate outputs for my PhD thesis is stored [here](https://github.com/simona-gioe/EMBL_svt/blob/main/full-environment.yml) for archival purposes

**Execution order**

The segment size analysis scripts are inteded to be executed sequentially, as each script generates the input required for the next. The order of execution is as follows:

1. [segment-size.py](https://github.com/simona-gioe/EMBL_svt/blob/main/segment-size/segment-size.py)
2. [extract_roi.py](https://github.com/simona-gioe/EMBL_svt/blob/main/segment-size/extract-roi.py)
3. [calculate-length.py](https://github.com/simona-gioe/EMBL_svt/blob/main/segment-size/calculate-length.py)
4. [sort-and-tidy.py](https://github.com/simona-gioe/EMBL_svt/blob/main/segment-size/sort-and-tidy.py)
5. [stats-and-plots.py](https://github.com/simona-gioe/EMBL_svt/blob/main/segment-size/stats-and-plots.py)


### Detection of somite boundaries

The first step of the workflow is to detect the anterior and posterior boundaries of nascent somites from brightfield intensity profiles extracted from manually traced ROI lines. The distance between the two boundaries is calculated as the somite length.



**Input files**

The expected input files are CSV files containing intensity profiles exported from Fiji. Each CSV file should correspond to one manually traced ROI line across the length of a nascent somite.
Each CSV file is expected to contain two columns: the first column should contain the distance along the ROI (in microns), and the second (labelled "Gray_Value") should contain the corresponding pixel intensity values.


*CSV structure example*

| Distance_(microns) | Gray_Value | 
| ------------------ | ---------- |
| 0.000              | 1234.000   |
| 1.384              | 9876.999   |
| 2.768              | 8877.665   |

<br />

For each nascent somite, three intensity profile CSV files should be available, corresponding to three ROI lines traced at different positions across the lenght of the somite during image analysis. These filenames are expected to follow the convention "YYYYMMDD_W00XX_C_ZZ_R.csv", where:
* YYYYMMDD is the experiment date,
* W00XX is the unique sample identifier,
* C is the experimental condition (C for control or D for DAPT-treated samples),
* ZZ is the nascent somite number,
* R is the ROI line number (1, 2, or 3).

For example, the three intensity profiles acquired for somite 8 of DAPT-treated sample W0003 imaged on January 1st 2021 are expected to be named:

20210101_W0003_D_08_1.csv

20210101_W0003_D_08_2.csv

20210101_W0003_D_08_3.csv

The script uses this naming convention to group the three independent measurements belonging to the same somite.


**Output files**

The script creates four output folders in the working directory:
* ```/plots``` contains quality control plots showing the boundaries detected on the complete raw intensity profiles
* ```/half_plots``` contains quality control plots showing boundary detection performed independently on the two halves of each raw intensity profile
* ```/sg_half_plots``` contains quality control plots showing boundary detection performed independently on the two halves of each Savitzky-Golay smoothed intensity profile
* ```/results``` contains a CSV file for each experimental sample. Each file is named ```YYYYMMDD_W00XX_segment-lengths.csv``` and it reports the calculated segment length according to the three boundary detection strategies

The quality control plots should be visually inspected to verify that the automatically detected boundaries correspond to the anatomical somite boundaries. Following visual inspection, the correct segment length should be selected for each intensity profile. Where necessary, automatically detected boundary position can be detected manually by comparison to the original brightfiled image. The curated measurements should be saved in a separate file for each sample, named ```YYYYMMDD_W00XX_lengths.csv```, which will serve as input for a subsequent step.

*CSV structure example*

| segment               | length | 
| --------------------- | ------ |
| YYYYMMDD_W00XX_C_01_1 | 123.45 |
| YYYYMMDD_W00XX_C_01_2 | 678.90 |
| YYYYMMDD_W00XX_C_01_3 | 56.78  |

<br />


**Instructions for execution**

1. Download [segment-size.py](https://github.com/simona-gioe/EMBL_svt/blob/main/segment-size/segment-size.py) and place it in the directory containing the intensity profile CSV files.
2. From the terminal, navigate to the correct directory and activate the segment size analysis conda environment (see **"Conda environment"** above)
3. Run the script by executing
   ```
   python segment-size.py
   ```
   
4. If the script encounters ambiguous boundary positions, it will pause and request manual input from the user to identify the correct boundary coordinates. In this event, follow the prompts diplayed in the terminal before allowing the analysis to continue.
5. After execution, inspect the generated quality control plots and manually correct the calculated segment lengths where necessary.
6. Save the curated segment length measurements as a CSV file (see **"Output files"**)


### Correlation of length measurements with acquisition timepoint

This script extracts the acquisition time corresponding to each nascent somite measurement from the Fiji ROI ZIP files saved during image analysis.

**Input files**

The expected input for this script consists of ZIP files exported from the Fiji ROI Manager. Each ZIP file should contain the complete set of ROI lines traced for a single sample. The file names are expected to follow the convention ```YYYYMMDD_W00XX_ROIs.zip```. Each nascent somite should be represented by three ROI lines, corresponding to the three independednt length measurements. The script assumes that the imaging data consists of 7 Z-slices acquired every 10 minutes.

**Output files**

The script creates a ```/results``` folder containing onw CSV file per analysed sample. The file names follow the convention ```YYYYMMDD_W00XX_ROIs-timepoints.csv```. Each output file contains two colums:
* ```somite-number``` stores the sequential index of the measured nascent somite,
* ```time(min)``` stores the acquisition timepoint of the measurement.

These CSV files serve as input for the next step of the analysis.

**Instructions for execution**

1. Download [extract-roi.py](https://github.com/simona-gioe/EMBL_svt/blob/main/segment-size/extract-roi.py) and place it in the directory containing the Fiji ROI ZIP files.
2. From the terminal, navigate to the correct directory and activate the segment size analysis conda environment (see **"Conda environment"** above)
3. Run the script by executing
   ```
   python extract-roi.py
   ```
   
4. If the script detects inconsistent acquisition times among the three ROI lines corresponding to the same somite, it will pause and request manual input from the user to specify the correct time point before continuing the analysis. In this event, follow the prompts diplayed in the terminal before allowing the analysis to continue.


### Calculation of somite lengths

This script combines the manually curated segment length measurements with their corresponding acquisition time points. For each nascent somite, the three independent ROI line measurements are averaged to obtain a single representative somite length.

**Input files**

The script expects two input files per analysed sample:
* A manually curated ```YYYYMMDD_W00XX_lengths.csv```` file containing the segment length measurements for all ROI lines acquired from the corresponding experimental sample
* A corresponding ```YYYYMMDD_W00XX_ROIs-timepoints.csv``` file containing the acquisition time for each mesured somite

Both these files are generated at previous steps of this workflow. They are automatically matched based on experiment date and sample identifier. For example:

20250418_W0005_lengths.csv
20250418_W0005_ROIs-timepoints.csv

**Output files**

The script generates one CSV file per analysed sample and stores it in the working directory. The output files follow the naming convention ```YYYYMMDD_W00XX_lengths_processed.csv```. For example:

20250418_W0005_lengths_processed.csv

Each output file contains the following columns:
* ```segment```: segment identifier
* ```average-length```: mean somite length calculated from the three idependent ROI line measurements
* ```standard-deviation```: standard deviation of the three independent measurements
* ```measurement-time(min)```: acquisition time of the corresponding somite measurements, in minutes

*CSV structure example*

| segment             | average-length | standard-deviation | measurement-time(min) | 
| ------------------- | -------------- | ------------------ | --------------------- |
| 20250418_W0005_D_00 | 123.45         | 1.1234567          | 0                     |
| 20250418_W0005_D_01 | 678.90         | 2.2345678          | 140                   |
| 20250418_W0005_D_02 | 56.78          | 3.3456789          | 280                   |

<br />

**Instructions for execution**



### Statistical analysis



**Input files**



**Output files**



**Instructions for execution**


### (Optional) Correlation of length measurements to oscillation period and phase



**Input files**



**Output files**



**Instructions for execution**


