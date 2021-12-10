# -*- coding: utf-8 -*-
"""

Samuel S. Streeter
Thayer School of Engineering at Dartmouth
Hanover, NH USA

Code for the following manuscript:
Streeter, S.S., Hunt, B., Zuurbier, R.A. et al. Developing diagnostic
assessment of breast lumpectomy tissues using radiomic and optical signatures.
Sci Rep 11, 21832 (2021). https://doi.org/10.1038/s41598-021-01414-z 

Radiomic/optomic feature quantification using PyRadiomics.

Updated 2021-12-08

"""

# Import packages
import logging
import radiomics
from radiomics import featureextractor, getFeatureClasses
import six
import SimpleITK as sitk
from scipy.io import loadmat
import numpy as np
import os
import pandas as pd

#------------------------------------------------------------------------------

# Need these variables for accessing the proper local sub-directory

# Minimum percentage of sub-image that must be tissue (i.e., non-NaN) pixels
MIN_PCT = 90

# Sampling ratio determines the # of sub-image samples to be extracted
# For a given specimen: (# of sub-image samples) = (# tissue pixels)/(# sub-
# image pixels) * RATIO
RATIO = 5

# Micro-CT and optical image data are co-registerd with the same pixel size
px_per_mm = 8 # pixels per mm
   
# Sampling window size
WIN_SZ_ALL    = np.array([2,3,4,5]) # Target window sz (square sub-image, mm)

# Local path to data
home_dir = './data/'

#------------------------------------------------------------------------------

# Get the PyRadiomics logger (default log-level = INFO)
logger = radiomics.logger
logger.setLevel(logging.DEBUG) # Set to DEBUG so debug messages are in log file

# Set up the handler to write out all log entries to a file
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

#------------------------------------------------------------------------------

# Grab local settings file
paramsFile_OP = os.path.abspath('Params_Optical.yaml')
paramsFile_CT = os.path.abspath('Params_MicroCT.yaml')

# Initialize feature extractor using the settings file
extractor_OP = featureextractor.RadiomicsFeatureExtractor(paramsFile_OP)
extractor_CT = featureextractor.RadiomicsFeatureExtractor(paramsFile_CT)

# For the Scientific Reports manuscript, a constant bin width of 0.005 was used
# for both the micro-CT and optical data, but the code here enables one to 
# iteratively adjust the bin widths and repeatedily extract radiomic features
# binWidth_OP = extractor_OP.settings['binWidth']
# binWidth_CT = extractor_CT.settings['binWidth']
binWidth_OP_all = np.array([0.005])
binWidth_CT_all = np.array([0.005])

# Loop through these PyRadiomics "bin widths" for optical data
for binWidth_OP in binWidth_OP_all: 
    
    # Update optical data bin width
    extractor_OP.settings['binWidth'] = binWidth_OP
    
    # Loop through these PyRadiomics "bin widths" for CT data
    for binWidth_CT in binWidth_CT_all:
        
        # Update CT data bin width
        extractor_CT.settings['binWidth'] = binWidth_CT
        
        # Count the # of features that will be quantified, print to console
        M = 0
        feature_names = []
        featureClasses = getFeatureClasses()
        for cls, features in six.iteritems(extractor_OP.enabledFeatures):
            if features is None or len(features) == 0:
                features = [f for f, deprecated in six.iteritems(featureClasses[cls].getFeatureNames()) if not deprecated]
            for f in features:
              
                print('%s %s'%(featureClasses[cls],f))
                print(getattr(featureClasses[cls], 'get%sFeatureValue' % f).__doc__)
                
                # Grab feature name
                temp_feature_class = cls.upper()
                
                # Finally, save this name
                feature_names.append('%s-%s'%(temp_feature_class,f))
                M = M + 1
                
        print('')
                 
        #------------------------------------------------------------------------------
        # Loop through all sub-image sample (i.e., sample window) sizes
        for i_ws in range(len(WIN_SZ_ALL)):
            
            # Current window size
            WIN_SZ = WIN_SZ_ALL[i_ws]
            print('Sampling window size: %.1f mm x %.1f mm...\n'%(WIN_SZ, WIN_SZ))
            WIN_DIM = px_per_mm * WIN_SZ # Num of pxls along side of sq window
            if WIN_DIM%2 == 0:
                WIN_DIM = WIN_DIM + 1    # WIN_DIM must be odd
            WIN_DIM_HALF = (WIN_DIM - 1)/2
            
            # Input path
            pth_in = "%sstep1/Sampling_Ratio%.2f_MinPct%i_Win%iPxls/"%(home_dir,RATIO,MIN_PCT,WIN_DIM)
        
            # Gather sample MAT file metadata
            Meta    = pd.read_table("%s%s"%(pth_in, 'Image_MAT_Data.csv'),delimiter=',',header=[0])
            MAT     = Meta.mat_file_id
            N       = len(MAT)
            
            # Gather image types that are in each MAT file 3D matrix sample
            IT = pd.read_table("%s%s"%(pth_in, 'Image_MAT_Matrix_Types.csv'),delimiter=',',header=[0])
            IMAGE_TYPES = IT.image_type
            CHANNELS    = IT.number_of_channels
            T           = np.sum(CHANNELS)
            
            # Create feature name array, including image type prefix
            FEATURE_LIST = []
            for image_type in IMAGE_TYPES:
                for feat in feature_names:
                    FEATURE_LIST.append("%s-%s"%(image_type,feat))
        
            # Preallocate radiomics data matrix for current sub-image size
            DATA = np.zeros([N, T*M])
            
            # Output path for current window size
            pth_out_main = "%sstep2/Sampling_Ratio%.2f_MinPct%i_Win%iPxls/"%(home_dir,RATIO,MIN_PCT,WIN_DIM)
            if not os.path.exists(pth_out_main):
                os.makedirs(pth_out_main)
            
            # Loop through all samples
            print('Progress: 000\u0025',end = '')
            for i in range(N):
            
                # Update command line
                print('\b\b\b\b\b%03i'%round(100*i/N),end = '')
                print('\u0025',end = '')
                
                # Load next MAT sample
                sample = loadmat("%sData/%s"%(pth_in,MAT[i]))
                sample = sample['current_sample']
            
                # Create mask for sample
                mask = np.copy(sample[:,:,0])
                mask[np.isnan(mask)] = 0
                mask[mask > 0] = 1
        
                # It seems like PyRadiomics NEEDS a mask that is not all 
                # inclusion pxls. If mask indicates that all pxls are tissue 
                # pxls, then simply zero pad the mask and sample image
                FLAG_PAD = 0
                if len(np.unique(mask)) == 1:
                    mask = np.pad(mask,1)
                    FLAG_PAD = 1
        
                # Convert mask to Simple ITK object
                mask_ITK = sitk.GetImageFromArray(mask)
                
                # Preallocate overall feature array
                output_all = []
                
                # Loop through all channels in spatially co-registered sample
                start = 0
                for j in range(T):
        
                    # Current sample (and current channel)
                    temp_sample     = np.copy(sample[:,:,j])
                    
                    # Zero pad if all inclusion pixels
                    if FLAG_PAD == 1:
                        temp_sample = np.pad(temp_sample,1)
                    
                    # Convert to Simple ITK object
                    temp_sample_ITK = sitk.GetImageFromArray(temp_sample)
        
                    # Calculate features - NOTE: Assuming CT is first here
                    if j == 0:
                        result = extractor_CT.execute(temp_sample_ITK, mask_ITK)
                    else:
                        result = extractor_OP.execute(temp_sample_ITK, mask_ITK)
        
                    # Save quantified features
                    output = []
                    for key, val in six.iteritems(result):
                        if 'original' in key and 'diagnostics' not in key:
                            # print("\t%s: %s" %(key, val))
                            output = np.append(output,val)
        
                    # Append to overall feature array
                    output_all = np.append(output_all,output)
                    
                # Append to overall sample feature array
                DATA[i,:] = output_all
        
            # Done extracting radiomic features, now save output in CSV
            df = pd.DataFrame(data=DATA, columns=FEATURE_LIST)
            df.to_csv("%sRadiomic_Features_binWidthsOP%.3fCT%.3f.csv"%(pth_out_main,binWidth_OP,binWidth_CT), index=False)
            print('')
            print('DONE!')
            print('')
