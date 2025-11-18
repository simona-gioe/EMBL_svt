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

This series of scripts was developed to analyse nascent somite length measurements acquired from brightfield imaging timeseries.
Detailed information on how the length measurements need to be acquired is available in section *5.5.4 Acquisition of somite length measurements* in the [published thesis](https://archiv.ub.uni-heidelberg.de/volltextserver/37071/). In brief, nascent somites can be measured by tracing three 1-pixel-wide ROI lines across the somite width, on the clearest Z-slice of the brightfield channel. For every ROI line, the intensity profile should then be saved as a CSV file. All ROI lines for a sample should also be saved together by adding them to the Fiji ROI manager as they are traced and then exporting the entire ROI set as a ZIP file.
It is recommended to lightly smooth the image prior to ROI line tracing by applying a Gaussian filter (σ= 0.5-1.7). The code is configured for timelapse datasets acquired every 10 minutes with 7 Z-slices. ROI lines should be traced on files containing the brightfield channel only.

**Conda environment**
All scripts for segment size analysis run within the same virtual environment.


### Detection of somite boundaries



**Input files**



**Output files**



**Instructions for execution**


### Correlation of length measurements with acquisition timepoint



**Input files**



**Output files**



**Instructions for execution**


### Calculation of somite lengths



**Input files**



**Output files**



**Instructions for execution**


### Statistical analysis



**Input files**



**Output files**



**Instructions for execution**


### (Optional) Correlation of length measurements to oscillation period and phase



**Input files**



**Output files**



**Instructions for execution**


