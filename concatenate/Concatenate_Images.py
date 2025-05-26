'''

Concatenates timelapses for multiposition experiments. Input raw image data for individual timepoints are saved in separate files.
Raw image files can contain multiple channels or multiple z-slices, but can not contain data from multiple positions or multiple timepoints.
Raw data from different positions are saved in different subfolders.

 Repository (original): https://git.embl.de/grp-almf/fiji-scripts/


 Creation:
 Date: November 2017
 
 Last modified: May 2025
 
 By: aliaksandr.halavatyi@embl.de

 Requirements
 - ImageJ
 - Bioformats plugin

 inputs:
 - Path to the input folder with image data to be processed (selected interactively with opened dialog).
 - namePattern - regular expression to fiter image files.

 outputs:
 - Saved tif (ImageJ) image files with concatenated timelapses. Image files are saved in the selected (input) folder. Names of saved timelapses correspond to the names of subfolders containing raw data for individual positions. 
 

'''

import re, os, os.path
from ij.io import DirectoryChooser
from ij.plugin import Concatenator,HyperStackConverter
from ij import IJ
from loci.plugins import BF

from java.io import File


inputFileExtention='lsm'

namePattern = re.compile('(.*).'+inputFileExtention) 

concatenator=Concatenator()



def concatenateWellPosition(inputDirectory, saveDirectory):

    filenames = sorted(os.listdir(inputDirectory))

    imageList=[]
    fileCount=0
    for filename in filenames:
        match = re.search(namePattern, filename)
        if (match == None):
            continue
        #print filename
        fileCount = fileCount + 1
        rawImage=BF.openImagePlus(os.path.join(inputDirectory,filename))[0]
        imageList.append(rawImage)

    result=concatenator.concatenateHyperstacks(imageList,'Concatenated hyperstack',False)
    print result.getNSlices()
    if result.getNSlices()>1:
        result2 = HyperStackConverter.toHyperStack(result, result.getNChannels(), result.getNSlices()/fileCount, fileCount, "czt", "Color")
        result=result2

    inputDirectoryFile=File(inputDirectory)
    parentDirName=inputDirectoryFile.getParent()
    outFileName=inputDirectoryFile.getName()

    IJ.saveAsTiff(result,os.path.join(saveDirectory,outFileName+'.tif'))

IJ.log('Concatenate script start')
inputDirectoryChooser = DirectoryChooser("Select Input Directory for many positions")
inputDirectoryGlobal=inputDirectoryChooser.getDirectory()
if inputDirectoryGlobal is None:
    sys.exit("No folder selected!")

for root, subdirectories, files in os.walk(inputDirectoryGlobal):
    #print 'root', root
    #print 'sub ', subdirectories
    #print 'files', files
    if len(subdirectories)>0:
        continue
    concatenateWellPosition(root,inputDirectoryGlobal)
IJ.log('Concatenate script end')
