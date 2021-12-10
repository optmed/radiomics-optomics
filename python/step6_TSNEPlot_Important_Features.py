# -*- coding: utf-8 -*-
"""

Samuel S. Streeter
Thayer School of Engineering at Dartmouth
Hanover, NH USA

Code for the following manuscript:
Streeter, S.S., Hunt, B., Zuurbier, R.A. et al. Developing diagnostic
assessment of breast lumpectomy tissues using radiomic and optical signatures.
Sci Rep 11, 21832 (2021). https://doi.org/10.1038/s41598-021-01414-z 

Two-dimensional (or three-dimensional) t-distributed Stochastic Neighbor 
Embedding (t-SNE) of the most important radiomic/optomic features based on 1000
Monte Carlo (i.e., GroupShuffleSplit) cross-validation folds.

Updated 2021-12-08

"""

import pandas as pd
import pickle
from sklearn.manifold import TSNE
import matplotlib; matplotlib.use('Qt5Agg') # <-- enables figures to be displayed on screen
import matplotlib.pyplot as plt
import os

#------------------------------------------------------------------------------

# Set the number of features used in classification task
featPerSplit = 6

# Set the target number of most important features to include to create t-SNE plot
target_num_features = 6

# Analyze or remove adipose tissue samples
ANALYZE_ADIPOSE = "No" # "Yes" or "No"

# Feature set
Feature_Set = "Combined" # "CT" or "Optical" or "Combined"

#------------------------------------------------------------------------------

# Set PyRadiomic bin widths
binWidth_OP = 0.005
binWidth_CT = 0.005

# Minimum percentage of sub-image that must be non-NaN tissue pixels
MIN_PCT = 90

# Sampling ratio determines the # of sub-image samples to be extracted
# For a given specimen: (# of sub-image samples) = (# tissue pixels)/(# sub-
# image pixels) * RATIO
RATIO = 5

# From step 4, we know that the processed tissue image pixel size for the CT 
# slices and optical data are all the same
px_per_mm = 8
   
# Select sub-image sample (i.e., window) size
WIN_SZ = 5 # mm
WIN_DIM = px_per_mm * WIN_SZ # Number of pixels along each side of sliding window
if WIN_DIM%2 == 0:
    WIN_DIM = WIN_DIM + 1    # WIN_DIM must be odd
WIN_DIM_HALF = (WIN_DIM - 1)/2

#------------------------------------------------------------------------------

# Classifier options
model_str = "RF"

# Number of MRMR ensemble solutions
MRMReMethod = 'classic'
MRMRe = 1

# Cross validation
N_SPLITS  = 1000
TEST_FRAC = 0.2

# ROC curve predictive confidence band
confidence = 0.95

# Nested cross validation - for feature selection in training set only
N_SPLITS2  = 10
TEST_FRAC2 = 0.2

# Local path to data
home_dir = './data/'

#------------------------------------------------------------------------------

# Input path for metadata
pth_in0 = "%sstep1/Sampling_Ratio%.2f_MinPct%i_Win%iPxls/"%(home_dir,RATIO,MIN_PCT,WIN_DIM)
    
# Gather sample MAT file metadata
Meta = pd.read_table("%s%s"%(pth_in0, 'Image_MAT_Data.csv'),delimiter=',',header=[0])  
CLASSES  = Meta.tissue_type_id
SUBTYPES = Meta.tissue_subtype_abbrev
PATIENTS = Meta.patient_id
MAT      = Meta.mat_file_id

#------------------------------------------------------------------------------

# Input path for unstandardized radiomic features
pth_in1 = "%sstep3/Sampling_Ratio%.2f_MinPct%i_Win%iPxls/"%(home_dir,RATIO,MIN_PCT,WIN_DIM)   

# Read in data
Data = pd.read_table("%sRadiomic_Features_binWidthsOP%.3fCT%.3f_Standardized_Zscore.csv"%(pth_in1, binWidth_OP, binWidth_CT),delimiter=',',header=[0])

#------------------------------------------------------------------------------

# Input path for most important features
pth_in2 = "%sstep6/Win%iPxls_binOP%.3fCT%.3f_%s_%isplitCV_%.2fTest/FeatSet%s_%iMRMRe%s_Adi%s/"%(home_dir,WIN_DIM,binWidth_OP,binWidth_CT,model_str,N_SPLITS,TEST_FRAC,Feature_Set,MRMRe,MRMReMethod[0],ANALYZE_ADIPOSE[0])

# Read in the most important features
content = pickle.load(open('%s%ifeatures_MostFrequentlyUsed25Features.pickle'%(pth_in2,featPerSplit), 'rb'))
important_features = list(content.keys()) # Already in order from most to least important
important_features = important_features[:target_num_features]

# Isolate subset of data containing only the most important features
subset = Data[important_features]

#------------------------------------------------------------------------------

# Set up colors for tissue subtypes
lut = {'Adi': '#FDFFC9',
 'Conn': '#FBC9FF',
 'FCD': '#C9FFFE',
 'IDCaHG': '#4B0000',
 'IDCaIG': '#990000',
 'IDCaLG': '#FF0000',
 'ILCa': '#9D0053'}
subtypes = list(lut.keys())
colors   = list(lut.values())

# Customize font sizes
SMALL_SIZE  = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

#------------------------------------------------------------------------------

# Now use t-distributed Stochastic Neighbor Embedding (t-SNE)
fig, ax = plt.subplots()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
tsne = TSNE(n_components=2, random_state=0,perplexity=30,n_iter=1000)
subset_transformed = tsne.fit_transform(subset)
ax.grid(True)
ax.set_axisbelow(True)
sc = []
for i in range(len(lut)):
    idx = SUBTYPES == subtypes[i]
    ax.scatter(subset_transformed[idx,0],subset_transformed[idx,1],#subset_transformed[idx,2],
               linewidth=0.5, marker='o',edgecolor='black',color=colors[i],
               alpha=0.5,s=25,label="%s"%subtypes[i])

# Now repeat the scatter plotting for ALL POINTS but make the points invisible
sc = ax.scatter(subset_transformed[:,0],subset_transformed[:,1],alpha = 0)
           
ax.set_box_aspect(1)
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
plt.rc('font', size=BIGGER_SIZE)       # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE) # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE) # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE) # legend fontsize
    
# OR turn off all axis details
#ax.axis('off')

# Create output path
pth_out = "%sstep7/Win%iPxls_binOP%.3fCT%.3f_%s_%isplitCV_%.2fTest/FeatSet%s_%iMRMRe%s_Adi%s/"%(home_dir,WIN_DIM,binWidth_OP,binWidth_CT,model_str,N_SPLITS,TEST_FRAC,Feature_Set,MRMRe,MRMReMethod[0],ANALYZE_ADIPOSE[0])
if not os.path.exists(pth_out):
    os.makedirs(pth_out)
                    
# Save the UNLABELED figure
fname = "%stSNE_%iFeatPerSplit_%iTargetFeat.svg"%(pth_out,featPerSplit,target_num_features)
plt.savefig(fname,bbox_inches='tight')

#------------------------------------------------------------------------------

# Then add labels for creating annotated inset examples
# CREDIT: https://stackoverflow.com/questions/7908636/is-it-possible-to-make-labels-appear-when-hovering-mouse-over-a-point-in-matplot
annot = ax.annotate("", xy=(0,0), xytext=(30,-50),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "\n ".join('{},{}'.format(*t) for t in zip(MAT[ind["ind"]],SUBTYPES[ind["ind"]]))
    annot.set_text(text)
    annot.set_fontsize = 0.05
    annot.get_bbox_patch().set_facecolor('white')
    annot.get_bbox_patch().set_alpha(0.5)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
plt.show()

#------------------------------------------------------------------------------
    
