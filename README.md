# EMBL_svt

This repository contains a collection of scripts supporting image and data analysis for my PhD work in the Aulehla Group at EMBL Heidelberg. Aside from the [image concatenation script](https://github.com/simona-gioe/EMBL_svt/blob/main/concatenate/Concatenate_Images.py), which was written and developed by Aliaksandr Halavatyi as part of a previous project, all other scripts in this repository were developed by me, and used to analyse data extracted from imaging datasets of mouse intact presomitic mesoderm. The results of this work will be available [here](https://archiv.ub.uni-heidelberg.de/volltextserver/37071/) from July 30th 2026.

## Timeseries concatenation

The [Concatenate_Images.py](https://github.com/simona-gioe/EMBL_svt/blob/main/concatenate/Concatenate_Images.py) script was written by Aliaksandr Halavatyi to concatenate multi-position live imaging time series. Information on dependencies (requirements), input and output formats, and instructions for running this script are documented at the beginning of the file.

## Period evolution analysis

This script was developed to align imaging samples based on their oscillations period evolution dynamics along the anteroposterior axis of the presomitic mesoderm. Period evolution data was extracted by [wavelet analysis](https://github.com/PGLSanchez/EMBL_OscillationsAnalysis/blob/master/EntrainmentAnalysis/SCRIPT-EntrainmentAnalysis.ipynb) performed on raw intensity data extracted from [intensity kymographs](https://git.embl.org/grp-cba/tail-analysis/-/blob/main/analysis_workflow.md?ref_type=heads).

To execute the [ap-period-stats.py](https://github.com/simona-gioe/EMBL_svt/blob/main/period-evolution/ap-period-stats.py) script, download it and save it in the same folder as your input files. The full conda environment used to run this code to generate outputs for my PhD thesis is stored [here](https://github.com/simona-gioe/EMBL_svt/blob/main/full-environment.yml) for archival purposes. For a list of dependencies, please consult the [period-requirements.txt]() file.

**Input files**

**Output files**


## Segment size analysis
