# -*- coding: utf-8 -*-
"""

Samuel S. Streeter
Thayer School of Engineering at Dartmouth
Hanover, NH USA

Code for the following manuscript:
Streeter, S.S., Hunt, B., Zuurbier, R.A. et al. Developing diagnostic
assessment of breast lumpectomy tissues using radiomic and optical signatures.
Sci Rep 11, 21832 (2021). https://doi.org/10.1038/s41598-021-01414-z 

Create hierarchically clustered heatmap of breast cancer radiomic/optomic data.

Updated 2021-12-08

"""

import seaborn as sns; sns.set(color_codes=True); sns.set(font="Arial")
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import pandas as pd
import os
import numpy as np
import scipy.cluster.hierarchy as sch
import warnings
warnings.filterwarnings("ignore", message="Clustering large matrix with scipy. Installing `fastcluster` may give better performance.")

#------------------------------------------------------------------------------

# Select the feature set desired:
Feature_Sets = ["Combined"] # ["CT","Optical","Combined"]

#------------------------------------------------------------------------------

# Clustering options
METHOD = 'ward'
METRIC = 'euclidean'

# Clustering settings - iteratively call:
# CLUSTERS = sch.fcluster(L, X.XX*d.max(), 'distance') where you adjust X.XX
# until np.unique(CLUSTERS) returns the desired number of clusters, then hard-
# code the desired cutoff here
TARGET_N_CLUSTERS = 3;

# Clustermap font size for text
FSZ = 1.5
sns.set(font_scale=FSZ)

# Leave this =1 to filter features
FILTER_FEATURES = 1

#------------------------------------------------------------------------------

# Need these variables for accessing the proper local sub-directory

# Set PyRadiomic bin widths
binWidth_OP = 0.005
binWidth_CT = 0.005

# Standardization method
standardization_choice = 'Zscore'

# Minimum percentage of sub-image that must be non-NaN tissue pixels
MIN_PCT = 90

# Sampling ratio determines the # of sub-image samples to be extracted
# For a given specimen: (# of sub-image samples) = (# tissue pixels)/(# sub-
# image pixels) * RATIO
RATIO = 5

# Micro-CT and optical image data are co-registerd with the same pixel size
px_per_mm = 8 # pixels per mm
   
# Sampling window size
WIN_SZ_ALL    = np.array([5]) # Target window size (square sub-image, mm)

# Local path to data
home_dir = './data/'
    
#------------------------------------------------------------------------------
# Loop through each feature set - CT alone, optical alone, then combined
for Feature_Set in Feature_Sets:

    print('FEATURE SET: %s'%Feature_Set)

    #--------------------------------------------------------------------------
    # Loop through all sub-image sizes, load STANDARDIZED radiomic features
    for i_ws in range(len(WIN_SZ_ALL)):
        
        # Current window size
        WIN_SZ = WIN_SZ_ALL[i_ws]
        print('  Sampling window size: %.1f mm x %.1f mm...\n'%(WIN_SZ, WIN_SZ))
        WIN_DIM = px_per_mm * WIN_SZ # Number of pixels along each side of sliding window
        if WIN_DIM%2 == 0:
            WIN_DIM = WIN_DIM + 1    # WIN_DIM must be odd
        WIN_DIM_HALF = (WIN_DIM - 1)/2
        
        # Input path for metadata
        pth_in1 = "%sstep1/Sampling_Ratio%.2f_MinPct%i_Win%iPxls/"%(home_dir,RATIO,MIN_PCT,WIN_DIM)
    
        # Create output path
        # Output path for current window size
        pth_out = "%sstep4/Sampling_Ratio%.2f_MinPct%i_Win%iPxls/"%(home_dir,RATIO,MIN_PCT,WIN_DIM)
        if not os.path.exists(pth_out):
            os.makedirs(pth_out)
        
        # Gather sample MAT file metadata
        Meta = pd.read_table("%s%s"%(pth_in1, 'Image_MAT_Data.csv'),delimiter=',',header=[0])  
        CLASSES  = Meta.tissue_type_id
        SUBTYPES = Meta.tissue_subtype_abbrev
        PATIENTS = Meta.patient_id
        
        #----------------------------------------------------------------------
        # ROWS (samples)
        
        # Create binary malignancy array - malignant (=2) versus non-malignant (=1)
        MALIGNANCY_NUMS = pd.DataFrame.to_numpy(CLASSES)
        MALIGNANCY_NUMS[MALIGNANCY_NUMS==2]=1
        MALIGNANCY_NUMS[MALIGNANCY_NUMS==3]=2
        MALIGNANCY = np.copy(SUBTYPES)
        for i in range(len(SUBTYPES)):
            if MALIGNANCY_NUMS[i]==1:
                MALIGNANCY[i] = 'Benign'
            else:
                MALIGNANCY[i] = 'Malignant'
                
        # Gather image types that are in each MAT file 3D matrix sample
        IT = pd.read_table("%s%s"%(pth_in1, 'Image_MAT_Matrix_Types.csv'),delimiter=',',header=[0])
        IMAGE_TYPES = IT.image_type
        CHANNELS    = IT.number_of_channels
        T           = np.sum(CHANNELS)
    
        # Input path for standardized radiomic features
        pth_in2 = "%sstep3/Sampling_Ratio%.2f_MinPct%i_Win%iPxls/"%(home_dir,RATIO,MIN_PCT,WIN_DIM)
    
        # Read in data
        Data = pd.read_table("%sRadiomic_Features_binWidthsOP%.3fCT%.3f_Standardized_%s.csv"%(pth_in2, binWidth_OP, binWidth_CT, standardization_choice),delimiter=',',header=[0])
    
        # Create row color labeling - malignant versus non-malignant   
        lut1 = dict(zip(np.unique(MALIGNANCY), ['#000000','#d3d3d3']))
        row_colors1 = pd.Series(MALIGNANCY,name='Malignancy').map(lut1)
        lut2 = dict(zip(np.unique(SUBTYPES), ['#FDFFC9','#FBC9FF','#C9FFFE','#4B0000','#990000','#FF0000','#9D0053']))
        row_colors2 = pd.Series(SUBTYPES,name='Subtype').map(lut2)
                
        # Create row color labeling - patient ID
        lut3 = dict(zip(np.unique(PATIENTS), sns.color_palette("plasma",len(np.unique(PATIENTS)))))
        row_colors3 = pd.Series(PATIENTS,name='Patient').map(lut3)
        
        #----------------------------------------------------------------------
        # COLUMNS (features or variables)
        
        # Filter feature set
        if FILTER_FEATURES == 1:
            FEATURE_NAMES = Data.columns
            FEATURE_NAMES_KEEP = np.copy(FEATURE_NAMES)
            if Feature_Set == "CT":
                for i in range(len(FEATURE_NAMES)):
                    if "CT"  in FEATURE_NAMES[i]:
                        # Do nothing
                        pass
                    else:
                        # Delete feature
                        del Data[FEATURE_NAMES[i]]
            elif Feature_Set == "Optical":
                for i in range(len(FEATURE_NAMES)):
                    if "CT"  not in FEATURE_NAMES[i]:
                        # Do nothing
                        pass
                    else:
                        # Delete feature
                        del Data[FEATURE_NAMES[i]]
            elif Feature_Set == "Combined":
                # Do nothing, keep all features
                pass
            
        #----------------------------------------------------------------------
        
        # Row linkage
        d_row = sch.distance.pdist(Data,metric=METRIC)
        L_row = sch.linkage(d_row, method=METHOD)
        
        # Column linkage
        d_col = sch.distance.pdist(Data.T,metric=METRIC)
        L_col = sch.linkage(d_col, method=METHOD)
        
        # Row (sample) clustering
        CLUSTERS = sch.fcluster(L_row, TARGET_N_CLUSTERS, criterion='maxclust')
        N_clusters = np.unique(CLUSTERS) # <-- redundant now, but leaving for the time being....
        
        # And create row labels for these clusters
        indexes = np.unique(CLUSTERS, return_index=True)[1]
        lut4 = dict(zip(CLUSTERS[indexes], sns.color_palette("Blues",len(N_clusters))))
        row_colors4 = pd.Series(CLUSTERS,name='Clusters').map(lut4)
        
        #----------------------------------------------------------------------
        
        # Create column coloring label - image feature modality
        FEATURE_NAMES = Data.columns
        MODALITY      = np.copy(FEATURE_NAMES)
        for i in range(len(FEATURE_NAMES)):
            if "CT" in FEATURE_NAMES[i]:
                MODALITY[i] = "CT"
            else:
                MODALITY[i] = "Optical"    
        Data.columns = MODALITY # <-- NEED TO REASSIGN COLUMN LABELS HERE
        indexes = np.unique(MODALITY, return_index=True)[1]
        indexes = np.flip(indexes)
        if len(np.unique(MODALITY)) == 1 and np.unique(MODALITY)=='CT':
            lut5 = dict(zip(MODALITY[indexes], ['#616161']))
        elif len(np.unique(MODALITY)) == 1 and np.unique(MODALITY)=='Optical':
            lut5 = dict(zip(MODALITY[indexes], ['#040C60']))
        else:
            lut5 = dict(zip(MODALITY[indexes], ['#040C60','#616161']))
        col_colors1 = pd.Series(MODALITY,name='Modality').map(lut5) 
        
        # Combine row color labeling - need to do this in order to keep 
        # labeling of all row labels
        row_colors_all = pd.concat([row_colors4,row_colors3,row_colors1,row_colors2],axis=1)
    
        # Do the same for column color labeling - IFF you have more than one set of column color labels!
        col_colors_all = pd.concat([col_colors1],axis=1)
        
        # Optional coloring of dendrogram
        # colmap = {(0.5,0.5,0.5),(0.6,0.6,0.6),(0.7,0.7,0.7),(0.8,0.8,0.8),(0.9,0.9,0.9)}
        # tree_kws=colmap <-- this argument in call to clustermap.... not working...
        
        #----------------------------------------------------------------------
        
        # Generate clustermap
        cg = sns.clustermap(Data, method=METHOD, col_cluster=True, row_cluster=True,
                            row_linkage=L_row, col_linkage=L_col, 
                            linewidths=0,
                            row_colors=row_colors_all, col_colors=[col_colors1], 
                            cmap="icefire", vmin=-2, vmax=2, 
                            xticklabels=False, yticklabels=False, dendrogram_ratio=(.125,0.001),#(.5,0.001),
                            cbar_kws={'label': standardization_choice})
                            #annot=True, annot_kws={"fontsize":FSZ})
        cg.ax_row_dendrogram.set_visible(False)
        cg.ax_col_dendrogram.set_visible(False)
        cg.ax_heatmap.set_xlabel('Features')
        cg.ax_heatmap.set_ylabel('Samples')
        plt.savefig("%sClustermap_%s_Features.png"%(pth_out,Feature_Set),bbox_inches='tight',dpi=1200)
        plt.close('all')
        
        #----------------------------------------------------------------------

        # Preallocate malignancy totals
        MALIGNANCY_TOTALS = np.zeros([len(N_clusters),2])
        
        # Preallocate tissue subtype totals
        temp_subtypes = np.unique(SUBTYPES)
        my_order = [0,1,2,5,4,3,6]
        temp_subtypes = [temp_subtypes[i] for i in my_order]
        SUBTYPES_IN_CLUSTER = np.zeros([len(N_clusters),len(temp_subtypes)])
        
        SUBTYPE_TOTALS = np.zeros(len(temp_subtypes))
        #SUBTYPE_FRACS  = np.zeros(len(temp_subtypes))
        for i in range(len(temp_subtypes)):
            t = SUBTYPES==temp_subtypes[i]
            SUBTYPE_TOTALS[i] = sum(bool(x) for x in t)
           # SUBTYPE_FRACS[i]  = sum(bool(x) for x in t)/Data.shape[0]
            
        # Loop through each cluster
        for i in np.unique(CLUSTERS):
    
            # Update
            print('Cluster %i/%i>>>'%(i,len(np.unique(CLUSTERS))))
            idx = CLUSTERS==i
            
            # First, patients
            PatientsInCluster = len(np.unique(PATIENTS[idx]))
            print('  %i patients'%PatientsInCluster)
    
            # Second, samples
            SamplesInCluster = len(idx[idx==True])
            print('  %i samples'%SamplesInCluster)
            
            # Third, malignancy
            MalignInCluster = MALIGNANCY_NUMS[idx]
            temp_num_benign  = len(MalignInCluster[MalignInCluster==1])
            temp_num_malign  = len(MalignInCluster[MalignInCluster==2])
            MALIGNANCY_TOTALS[i-1][0] = temp_num_benign
            MALIGNANCY_TOTALS[i-1][1] = temp_num_malign
            print('  %i%% malignant'%(np.round(100*temp_num_malign/(temp_num_benign+temp_num_malign))))
            
            # Fourth, tissue subtype
            SubtypesInCluster = SUBTYPES[idx]
            for j in range(len(temp_subtypes)):
                t = SubtypesInCluster==temp_subtypes[j]
                temp_SubtypeInClusterTotal = sum(bool(x) for x in t)
                SUBTYPES_IN_CLUSTER[i-1][j] = temp_SubtypeInClusterTotal
                print('    %i%% (%i/%i) %s'%(np.round(100*temp_SubtypeInClusterTotal/(SUBTYPE_TOTALS[j])),temp_SubtypeInClusterTotal,SUBTYPE_TOTALS[j],temp_subtypes[j]))