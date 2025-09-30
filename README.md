# EMBL_svt

This repository contains a collection of scripts supporting image and data analysis for my PhD work in the Aulehla Group at EMBL Heidelberg. Aside from the [image concatenation script](https://github.com/simona-gioe/EMBL_svt/blob/main/concatenate/Concatenate_Images.py), which was written and developed by Aliaksandr Halavatyi as part of a previous project, all other scripts in this repository were developed by me, and used to analyse data extracted from imaging datasets of mouse intact presomitic mesoderm. The results of this work will be available [here](https://archiv.ub.uni-heidelberg.de/volltextserver/37071/) from July 30th 2026.

## Timeseries concatenation

The [Concatenate_Images.py](https://github.com/simona-gioe/EMBL_svt/blob/main/concatenate/Concatenate_Images.py) script was written by Aliaksandr Halavatyi to concatenate multi-position live imaging time series. Information on dependencies (requirements), input and output formats, and instructions for running this script are documented at the beginning of the file.

## Period evolution analysis

This script was developed to align imaging samples based on their oscillations period evolution dynamics along the anteroposterior axis of the presomitic mesoderm. Period evolution data was extracted by [wavelet analysis](https://github.com/PGLSanchez/EMBL_OscillationsAnalysis/blob/master/EntrainmentAnalysis/SCRIPT-EntrainmentAnalysis.ipynb) performed on raw intensity data extracted from [intensity kymographs](https://git.embl.org/grp-cba/tail-analysis/-/blob/main/analysis_workflow.md?ref_type=heads).

**Input files**

**Output files**

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
    * If you want to create your a new environment, you can consult [period-requirements.txt](https://github.com/simona-gioe/EMBL_svt/blob/main/period-requirements.txt) for a list of dependencies
    * The full conda environment used to run this code to generate outputs for my PhD thesis is stored [here](https://github.com/simona-gioe/EMBL_svt/blob/main/full-environment.yml) for archival purposes
3. After activating the virtual environment, you can run the script by copy-pasting and executing this command on the terminal:
```
python ap-period-stats.py
```


## Segment size analysis
