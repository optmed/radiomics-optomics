# -*- coding: utf-8 -*-
"""

Samuel S. Streeter
Thayer School of Engineering at Dartmouth
Hanover, NH USA

Code for the following manuscript:
Streeter, S.S., Hunt, B., Zuurbier, R.A. et al. Developing diagnostic
assessment of breast lumpectomy tissues using radiomic and optical signatures.
Sci Rep 11, 21832 (2021). https://doi.org/10.1038/s41598-021-01414-z 

Standardize radiomic/optomic features to z-scores/

Updated 2021-12-08

"""

# Import packages
import numpy as np
import os
import pandas as pd

#------------------------------------------------------------------------------

# Need these variables for accessing the proper local sub-directory

# Set PyRadiomic bin widths
binWidth_OP = 0.005
binWidth_CT = 0.005

# Standardization method
standardization_choice = 'Zscore'

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

# Loop through all sub-image sample (i.e., sample window) sizes
for i_ws in range(len(WIN_SZ_ALL)):
    
    # Current window size
    WIN_SZ = WIN_SZ_ALL[i_ws]
    print('Sampling window size: %.1f mm x %.1f mm...'%(WIN_SZ, WIN_SZ), end='')
    WIN_DIM = px_per_mm * WIN_SZ # Num of pxls along side of sq window
    if WIN_DIM%2 == 0:
        WIN_DIM = WIN_DIM + 1    # WIN_DIM must be odd
    WIN_DIM_HALF = (WIN_DIM - 1)/2
    
    # Input paths
    pth_in1 = "%sstep1/Sampling_Ratio%.2f_MinPct%i_Win%iPxls/"%(home_dir,RATIO,MIN_PCT,WIN_DIM)
    pth_in2 = "%sstep2/Sampling_Ratio%.2f_MinPct%i_Win%iPxls/"%(home_dir,RATIO,MIN_PCT,WIN_DIM)

    # Gather sample MAT file metadata
    Meta    = pd.read_table("%s%s"%(pth_in1, 'Image_MAT_Data.csv'),delimiter=',',header=[0])
    MAT     = Meta.mat_file_id
    N       = len(MAT)
    
    # Gather image types that are in each MAT file 3D matrix sample
    IT = pd.read_table("%s%s"%(pth_in1, 'Image_MAT_Matrix_Types.csv'),delimiter=',',header=[0])
    IMAGE_TYPES = IT.image_type
    CHANNELS    = IT.number_of_channels
    T           = np.sum(CHANNELS)
    
    # Read in radiomic data
    df = pd.read_table("%sRadiomic_Features_binWidthsOP%.3fCT%.3f.csv"%(pth_in2,binWidth_OP,binWidth_CT),delimiter=',',header=[0])

    # Standardize each column (feature) to z-scores
    df_scaled = df.copy()
    df_scaled = (df_scaled - df_scaled.mean()) / df_scaled.std()
        
    # Output path for current window size
    pth_out_main = "%sstep3/Sampling_Ratio%.2f_MinPct%i_Win%iPxls/"%(home_dir,RATIO,MIN_PCT,WIN_DIM)
    if not os.path.exists(pth_out_main):
        os.makedirs(pth_out_main)

    # Save z-score data in CSV
    df_scaled.to_csv("%sRadiomic_Features_binWidthsOP%.3fCT%.3f_Standardized_%s.csv"%(pth_out_main,binWidth_OP,binWidth_CT,standardization_choice), index=False)
    print('DONE!')
    print('')