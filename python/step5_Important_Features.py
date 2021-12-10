# -*- coding: utf-8 -*-
"""

Samuel S. Streeter
Thayer School of Engineering at Dartmouth
Hanover, NH USA

Code for the following manuscript:
Streeter, S.S., Hunt, B., Zuurbier, R.A. et al. Developing diagnostic
assessment of breast lumpectomy tissues using radiomic and optical signatures.
Sci Rep 11, 21832 (2021). https://doi.org/10.1038/s41598-021-01414-z 

Identify the most important radiomic/optomic features based on MRMR feature 
selection and 1000 Monte Carlo cross-validation (i.e., GroupShuffleSplit) splits.

Updated 2021-12-08

"""

import pandas as pd
import sys
import pickle
import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg') # <-- prevents figures from displaying on screen
from matplotlib import cm
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
import logging
import collections
logging.getLogger().setLevel(logging.CRITICAL) # Avoids "No handles with labels found to put in legend." error


#------------------------------------------------------------------------------
# Analyze or remove adipose tissue samples
ANALYZE_ADIPOSE = "Yes" # "Yes" or "No"

#------------------------------------------------------------------------------
# Repeat this script for each of these three:
Feature_Sets = ["CT","Optical","Combined"]
colors = cm.get_cmap('viridis',3)

#------------------------------------------------------------------------------

# Classifier options
model_str = "RF"

# Number of features
FNUM_ALL = np.array([6]) #np.linspace(5,10,6) # np.linspace(1,30,30)

# Number of MRMR ensemble solutions
MRMReMethod = 'classic' # or 'bootstrap' for ensemble, or 'classic' for just MRMR
MRMRe = 1

# Cross validation
N_SPLITS  = 1000
TEST_FRAC = 0.2
CV        = GroupShuffleSplit(n_splits=N_SPLITS,test_size=TEST_FRAC,random_state=0)

# ROC curve predictive confidence band
confidence = 0.95

# Nested cross validation - for feature selection in training set only
N_SPLITS2  = 10
TEST_FRAC2 = 0.2
CV2        = GroupShuffleSplit(n_splits=N_SPLITS2,test_size=TEST_FRAC2,random_state=0)

# Random under sampler
rus = RandomUnderSampler(random_state=0)

# Create standard scaler (Z score)
stdscaler = StandardScaler()

#------------------------------------------------------------------------------

# Need these variables for accessing the proper local sub-directory

# Set PyRadiomic bin widths
binWidth_OP_all = np.array([0.005])
binWidth_CT_all = np.array([0.005])

# Minimum percentage of sub-image that must be non-NaN tissue pixels
MIN_PCT = 90

# Sampling ratio determines the # of sub-image samples to be extracted
# For a given specimen: (# of sub-image samples) = (# tissue pixels)/(# sub-
# image pixels) * RATIO
RATIO = 5

# Micro-CT and optical image data are co-registerd with the same pixel size
px_per_mm = 8 # pixels per mm

# Sampling window size
WIN_SZ_ALL    = np.array([2,3,4,5])

# Local path to data
home_dir = './data/'

# Customize font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
        
#------------------------------------------------------------------------------

# Clustering options
METHOD = 'ward'
METRIC = 'euclidean'

#------------------------------------------------------------------------------

# Loop through PyRadiomics bin widths - OPTICAL
cnt_bw_op = 0
for binWidth_OP in binWidth_OP_all:
               
    cnt_bw_op = cnt_bw_op + 1
    print('PyRadiomics OPTICAL bin width (%i/%i): %.3f'%(cnt_bw_op,len(binWidth_OP_all),binWidth_OP))
    
    #--------------------------------------------------------------------------
    
    # Loop through PyRadiomics bin widths - CT
    cnt_bw_ct = 0
    for binWidth_CT in binWidth_CT_all:
    
        cnt_bw_ct = cnt_bw_ct + 1
        print('  PyRadiomics CT bin width (%i/%i): %.3f'%(cnt_bw_ct,len(binWidth_CT_all),binWidth_CT))
    
        #----------------------------------------------------------------------
        
        # Loop through each feature set - CT alone, optical alone, then combined
        cnt_fs = 0
        for Feature_Set in Feature_Sets:
          
            cnt_fs = cnt_fs + 1
            print('    FEATURE SET (%i/%i): %s'%(cnt_fs,len(Feature_Sets),Feature_Set))
                
            #------------------------------------------------------------------
            
            # Loop through all sub-image sizes, load unstandardized radiomic features
            cnt_ws = 0
            for i_ws in range(len(WIN_SZ_ALL)):
        
                # Current window size
                cnt_ws = cnt_ws + 1
                WIN_SZ = WIN_SZ_ALL[i_ws]
                WIN_DIM = px_per_mm * WIN_SZ # Number of pixels along each side of sliding window
                if WIN_DIM%2 == 0:
                    WIN_DIM = WIN_DIM + 1    # WIN_DIM must be odd
                WIN_DIM_HALF = (WIN_DIM - 1)/2
            
                print('      Sampling window size (%i/%i): %.1f mm x %.1f mm'%(cnt_ws, len(WIN_SZ_ALL),WIN_SZ, WIN_SZ))

                # Set input path
                pth_in = "%sstep5/Win%iPxls_binOP%.3fCT%.3f_%s_%isplitCV_%.2fTest/FeatSet%s_%iMRMRe%s_Adi%s/"%(home_dir,WIN_DIM,binWidth_OP,binWidth_CT,model_str,N_SPLITS,TEST_FRAC,Feature_Set,MRMRe,MRMReMethod[0],ANALYZE_ADIPOSE[0])
                
                # Create output path
                pth_out = "%sstep6/Win%iPxls_binOP%.3fCT%.3f_%s_%isplitCV_%.2fTest/FeatSet%s_%iMRMRe%s_Adi%s/"%(home_dir,WIN_DIM,binWidth_OP,binWidth_CT,model_str,N_SPLITS,TEST_FRAC,Feature_Set,MRMRe,MRMReMethod[0],ANALYZE_ADIPOSE[0])
                if not os.path.exists(pth_out):
                    os.makedirs(pth_out)
                    
                #----------------------------------------------------------------------
                
                # Loop through array of feature numbers
                cnt_fs = 0
                for FNUM in FNUM_ALL:
                
                    cnt_fs = cnt_fs + 1
                    print('        Features (%i/%i): %i'%(cnt_fs,len(FNUM_ALL),FNUM))
                    
                    # Preallocate feature tallying list
                    feature_list_original = []
                    feature_list = []
                
                    # Load classification summary data
                    Data = pd.read_table("%s%s"%(pth_in, '%iFeatures_Summary.csv'%FNUM),delimiter=',',header=[0])  
        
                    # Grab selected features
                    t = Data['mrmr_features_selected']
                    
                    # Clean up feature names
                    for ii in range(len(t)):
                        temp = t[ii].split("'")

                        for jj in range(len(temp)):
                            if any(c.isalpha() for c in temp[jj]) == True:
                        
                                # Here, we know that we have a useful feature name
                                # However, let's "clean up" these feature names for final figure display
                                # Convention: Only capital letters with preceeding "feature set" or modality
                                # with an underscore in between
                                                                
                                # Record original name
                                feature_list_original.append(temp[jj])
                        
                                # First, parse feature name
                                tt = temp[jj].split('-')
                                if len(tt) != 3:
                                    print('WARNING: CHECK FEATURE NAME!')
                                temp_fset   = tt[0]
                                temp_fclass = tt[1]
                                temp_feat   = tt[2]
                                
                                # Second, determine the class of feature
                                # The options are: FIRSTORDER, GLCM, GLRLM, GLSZM, GLDM, NGTDM
                                # Really, just abbreviate "FIRSTORDER" as "FO"
                                if temp_fclass == 'FIRSTORDER':
                                    temp_fclass = 'FO'
                                
                                # GO HERE IF CT
                                if temp_fset == 'CT_50kVp':
                                    grab_cap_letters = [char for char in temp_feat if char.isupper()]
                                    if len(grab_cap_letters) == 1:
                                        temp_feat_abbv = temp_feat[:4]
                                    else:
                                        temp_feat_abbv = "".join(grab_cap_letters)
                                    final = "".join(['CT-',temp_fclass,'-',temp_feat_abbv])
                                
                                    # A final correction - this convention works for every
                                    # feature EXCEPT GLCM Joint Energy and GLCM Joint Entropy
                                    if temp_feat == 'JointEnergy':
                                        final = "".join(['CT-',temp_fclass,'-JEner'])
                                    if temp_feat == 'JointEntropy':
                                        final = "".join(['CT-',temp_fclass,'-JEntr'])
                                    
                                # GO HERE IF OPTICAL
                                else:
                                    grab_cap_letters = [char for char in temp_feat if char.isupper()]
                                    if len(grab_cap_letters) == 1:
                                        temp_feat_abbv = temp_feat[:4]
                                    else:
                                        temp_feat_abbv = "".join(grab_cap_letters)
                                    temp_fset = temp_fset.split('_')
                                    
                                    if len(temp_feat_abbv) == 1:
                                        sys.exit()
                                    final = "".join([temp_fset[0],'-',temp_fset[1],'-',temp_fclass,'-',temp_feat_abbv])
                                    
                                    # A final correction - this convention works for every
                                    # feature EXCEPT GLCM Joint Energy and GLCM Joint Entropy
                                    if temp_feat == 'JointEnergy':
                                        final = "".join([temp_fset[0],'-',temp_fset[1],'-',temp_fclass,'-JEner'])
                                    if temp_feat == 'JointEntropy':
                                        final = "".join([temp_fset[0],'-',temp_fset[1],'-',temp_fclass,'-JEntr'])
                                    
                                # Add cleaned up feature name to list
                                feature_list.append(final)
                                        
                    # Create a histogram of selected features
                    tallies = collections.Counter(feature_list)
                    tallies = collections.OrderedDict(tallies.most_common())
                    
                    # Create a histogram of selected features - SAME FOR ORIGINAL FEATURE NAMES
                    tallies2 = collections.Counter(feature_list_original)
                    tallies2 = collections.OrderedDict(tallies2.most_common())
                    
                    # Next, if feature dictionary is longer than 10, truncate it
                    if len(tallies) > 25:
                        temp = list(tallies.items())[:25]
                        tallies = collections.OrderedDict(temp)
                        
                    # Plot result
                    fig, ax = plt.subplots()
                    barlist = ax.bar(tallies.keys(), np.array(list(tallies.values()))/N_SPLITS, width=.75)
                                        
                    # Color each bar depending on the modality
                    for ii in range(len(tallies)):
                        if 'CT-' in list(tallies.keys())[ii]:
                            barlist[ii].set_color(colors(0))
                            barlist[ii].set_alpha(0.5)
                        else:
                            barlist[ii].set_color(colors(1))
                            barlist[ii].set_alpha(0.5)
                    ax.set_xlabel('Features Selected By MRMR')
                    ax.set_ylabel('Fraction Of All CV Splits')
                    plt.setp(ax.get_xticklabels(), ha="center", rotation=90,fontsize=SMALL_SIZE)
                    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
                    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
                    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
                    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
                    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
                    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
                    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
                    
                    # Plot cutoff for six optimal features
                    ax.plot([5.5, 5.5],[0,1],linestyle=':',color='red',linewidth=1)

                    # Adjust the aspect ratio of the axes
                    ax.set_aspect(8)
                        
                    # Set y limits
                    ax.set_ylim([0, 1])
                    
                    # Save it
                    fname = "%s%iFeatures.png"%(pth_out,FNUM)
                    plt.savefig(fname,bbox_inches='tight',dpi=400)
                    
                    #----------------------------------------------------------------------
                    # Keep the original feature names
                    temp2 = list(tallies2.items())[:25] # Keep only six features, because that's the cutoff needed for 1% change in accuracy....
                    tallies2 = collections.OrderedDict(temp2)
                    
                    # Save these most important features
                    fname = "%s%ifeatures_MostFrequentlyUsed25Features.pickle"%(pth_out,FNUM)
                    with open(fname,'wb') as f:
                        pickle.dump(tallies2, f)