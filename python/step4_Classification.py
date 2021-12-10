# -*- coding: utf-8 -*-
"""

Samuel S. Streeter
Thayer School of Engineering at Dartmouth
Hanover, NH USA

Code for the following manuscript:
Streeter, S.S., Hunt, B., Zuurbier, R.A. et al. Developing diagnostic
assessment of breast lumpectomy tissues using radiomic and optical signatures.
Sci Rep 11, 21832 (2021). https://doi.org/10.1038/s41598-021-01414-z 

Perform classification.

Updated 2021-12-08

"""

import pandas as pd
import sys
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
import matplotlib; matplotlib.use('Agg') # <-- prevents figures from displaying on screen
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from pymrmre import mrmr
from sklearn.metrics import auc, plot_roc_curve, accuracy_score,recall_score,roc_auc_score,f1_score,precision_score
import logging
import collections
logging.getLogger().setLevel(logging.CRITICAL) # Avoids "No handles with labels found to put in legend." error

#------------------------------------------------------------------------------

# Analyze or remove adipose tissue samples
ANALYZE_ADIPOSE = "Yes" # "Yes" or "No"

#------------------------------------------------------------------------------

# Repeat this script for each of these three:
Feature_Sets = ["CT","Optical","Combined"] # All options: ["CT","Optical","Combined"]
CI_colors    = ['k','b','g'] # Confidence interval fill colors based on feature set

# Keep =1 to filter features
FILTER_FEATURES = 1

#------------------------------------------------------------------------------

# Classifier options
model = RandomForestClassifier(random_state=0,n_jobs=-1); model_str = "RF"

# Number of features
FNUM_ALL = np.array([6]) #np.linspace(1,30,30) 

# Number of MRMR ensemble solutions
MRMReMethod = 'classic' # 'bootstrap' for ensemble or 'classic' for just MRMR
MRMRe = 1

# Cross validation - training/testing sets
N_SPLITS  = 5
TEST_FRAC = 0.2
CV        = GroupShuffleSplit(n_splits=N_SPLITS,test_size=TEST_FRAC,random_state=1)

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
WIN_SZ_ALL    = np.array([5]) # np.array([2,3,4,5])

# Local path to data
home_dir = './data/'

# Make sure that classic MRMR uses solution length of 1
if MRMReMethod == 'classic' and MRMRe != 1:
    print('WARNING: If using CLASSIC MRMR, set MRMRe to 1! Proceeding with MRMRe=1...')
    MRMRe = 1

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
                print('      Sampling window size (%i/%i): %.1f mm x %.1f mm'%(cnt_ws, len(WIN_SZ_ALL),WIN_SZ, WIN_SZ))
                WIN_DIM = px_per_mm * WIN_SZ # Number of pixels along each side of sliding window
                if WIN_DIM%2 == 0:
                    WIN_DIM = WIN_DIM + 1    # WIN_DIM must be odd
                WIN_DIM_HALF = (WIN_DIM - 1)/2
            
                # Input for original sample metadata
                pth_in0 = "%sstep1/Sampling_Ratio%.2f_MinPct%i_Win%iPxls/"%(home_dir,RATIO,MIN_PCT,WIN_DIM)
                
                # Input path for unstandardized radiomic features
                pth_in1 = "%sstep2/Sampling_Ratio%.2f_MinPct%i_Win%iPxls/"%(home_dir,RATIO,MIN_PCT,WIN_DIM)   
                
                # Create output path
                pth_out = "%sstep5/Win%iPxls_binOP%.3fCT%.3f_%s_%isplitCV_%.2fTest/FeatSet%s_%iMRMRe%s_Adi%s/"%(home_dir,WIN_DIM,binWidth_OP,binWidth_CT,model_str,N_SPLITS,TEST_FRAC,Feature_Set,MRMRe,MRMReMethod[0],ANALYZE_ADIPOSE[0])
                if not os.path.exists(pth_out):
                    os.makedirs(pth_out)
                
                #--------------------------------------------------------------
                
                # Load meta data
                Meta = pd.read_table("%s%s"%(pth_in0, 'Image_MAT_Data.csv'),delimiter=',',header=[0])  
                CLASSES = pd.Series.to_numpy(Meta.tissue_type_id)
                CLASSES[CLASSES==2]=1
                CLASSES[CLASSES==3]=2  
                PATIENTS = Meta.patient_id
                SUBTYPES = Meta.tissue_subtype_abbrev
                
                # Read in feature data
                Data = pd.read_table("%sRadiomic_Features_binWidthsOP%.3fCT%.3f.csv"%(pth_in1, binWidth_OP, binWidth_CT),delimiter=',',header=[0])
                
                # Remove unnamed column IF it is present
                try:
                    del Data['Unnamed: 0']
                except KeyError:
                    pass
                
                #--------------------------------------------------------------
               
                # Remove all adipose samples if desired
                if ANALYZE_ADIPOSE == "No":
                    idx     = SUBTYPES != 'Adi'
                    CLASSES = CLASSES[idx]
                    PATIENTS = PATIENTS[idx]
                    SUBTYPES = SUBTYPES[idx]
                    Data = Data[idx]
                    
                #--------------------------------------------------------------
                
                # Filter feature set
                if FILTER_FEATURES == 1:
                    FEATURE_NAMES = Data.columns
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
                    else:
                        print('Feature_Set invalid!')
                        sys.exit()
                                
                #--------------------------------------------------------------
                
                # Loop through array of feature numbers
                cnt_feat = 0
                for FNUM in FNUM_ALL:
                
                    cnt_feat = cnt_feat + 1
                    print('        Features (%i/%i): %i'%(cnt_feat,len(FNUM_ALL),FNUM))
                    
                    # Preallocate CV split data and classification performance dataframe - MAIN OUTPUT
                    OUTPUT = pd.DataFrame(columns=
                                ["split_number","num_features","mrmr_features_selected","mrmr_best_solution_idx","mrmr_best_solution_nestedCV_accuracy",
                                 "num_patients_train","num_patients_test",
                                "patient_ids_train","patient_ids_test",
                                "num_samples_benign_train","num_samples_benign_test",
                                "num_samples_malign_train","num_samples_malign_test",
                                "num_adipose_samples_train",   "num_adipose_samples_test",   "accuracy_adipose",
                                "num_connective_samples_train","num_connective_samples_test","accuracy_connective",
                                "num_fcd_samples_train",       "num_fcd_samples_test",       "accuracy_fcd",
                                "num_idca_lg_samples_train",   "num_idca_lg_samples_test",   "accuracy_idca_lg",
                                "num_idca_ig_samples_train",   "num_idca_ig_samples_test",   "accuracy_idca_ig",
                                "num_idca_hg_samples_train",   "num_idca_hg_samples_test",   "accuracy_idca_hg",
                                "num_ilca_samples_train",      "num_ilca_samples_test",      "accuracy_ilca",
                                "perform_accuracy","perform_recall",
                                "perform_auc","perform_f1","perform_precision"])
                    
                    # Preallocate prediction malignancy probability array
                    prob_mal_MATRIX = [[] for jj in iter(range(Data.shape[0]))]
                                            
                    # Preallocate trained model (aka estimator) array and ROC curve arrays
                    Estimators = []
                    fprs = []
                    tprs = []
                    aucs = []
                    baseline_fpr = np.linspace(0, 1, 100)
                    baseline_tpr = np.linspace(0, 1, 100)
                    
                    # Create ROC curve plot
                    fig, ax = plt.subplots()
                    
                    # Preallocate y_predict and y_test (groundtruth) arrays
                    Y_PREDICT_ALL       = []
                    Y_PREDICT_PROBS_ALL = []
                    Y_TEST_ALL          = []
                    
                    #----------------------------------------------------------    
                    
                    # Loop through each iteration of cross-validation
                    DUD_SPLITS = 0
                    for i, (train, test) in enumerate(CV.split(Data,CLASSES,groups=PATIENTS)):
                        
                        # Isolate training set
                        X_train = Data.iloc[train]
                        y_train = pd.DataFrame(CLASSES[train])
                        p_train = PATIENTS.iloc[train]
                        s_train = SUBTYPES.iloc[train]
                        c_train = CLASSES[train]
                        
                        # Isolate testing set
                        X_test = Data.iloc[test]
                        y_test = pd.DataFrame(CLASSES[test])
                        p_test = PATIENTS.iloc[test]
                        s_test = SUBTYPES.iloc[test]
                        c_test = CLASSES[test]
                        
                        # Randomly undersample training set benign samples to balance training set
                        X_train_us, y_train_us = rus.fit_resample(X_train, y_train)
                        
                        # Record the indices and subtype totals after undersampling
                        idx_rus = rus.sample_indices_
                        s_train_us = s_train.iloc[idx_rus]
                        subtype_tally_train = collections.Counter(s_train_us)
                        subtype_tally_test  = collections.Counter(s_test)
                        
                        # Also apply undersampling of training set to other training arrays
                        # (just in case undersampling removes an entire patient)
                        p_train_us = p_train.iloc[idx_rus]
                        c_train_us = c_train[idx_rus]
                        
                        # Create standard scaler object based on all training features
                        scaler = stdscaler.fit(X_train_us)
                         
                        # Save the column names
                        colnames = X_train_us.columns
                        
                        # Scale training and testing sets  (Z scores)
                        # based on mean and variance of training set
                        X_train_us = scaler.transform(X_train_us)
                        X_test     = scaler.transform(X_test)
                                                
                        # Convert train back to DataFrame
                        X_train_us = pd.DataFrame(X_train_us,columns=colnames)
                        X_test     = pd.DataFrame(X_test,columns=colnames)
                    
                        #------------------------------------------------------
                        # Feature selection
                        if MRMReMethod == 'classic':
                            solutions = mrmr.mrmr_ensemble(features=X_train_us,targets=y_train_us,solution_length=int(FNUM),solution_count=1,return_index=False)
                        else:
                            solutions = mrmr.mrmr_ensemble(features=X_train_us,targets=y_train_us,solution_length=int(FNUM),solution_count=MRMRe,return_index=False,method=MRMReMethod)
                                            
                        # Keep only features selected by best MRMR ensemble solution
                        temp_nestedCV_acc  = np.zeros([MRMRe,N_SPLITS2])
                        if np.ndim(temp_nestedCV_acc) > 1 and MRMRe > 1 and MRMReMethod != 'classic':
                            for j in range(MRMRe):
                                
                                # Current MRMR ensemble solution
                                temp_fkeep      = solutions.iloc[0][j]
                                temp_X_train_us = X_train_us[temp_fkeep]
                                temp_X_test     = X_test[temp_fkeep]
                            
                                # Now split the training set in two - this is nested cross-validation
                                for k, (train2, test2) in enumerate(CV2.split(temp_X_train_us,y_train_us,groups=p_train_us)):
                                                          
                                    # Train the model on training set (of the training set!)
                                    model.fit(temp_X_train_us.iloc[train2], np.ravel(y_train_us.iloc[train2]))
                                    
                                    # Test the model on testing set (of the training set!)
                                    temp_y_predict = model.predict(temp_X_train_us.iloc[test2])
                    
                                    # Determine estimate of accuracy of current MRMR ensemble solution
                                    temp_nestedCV_acc[j,k] = accuracy_score(y_train_us.iloc[test2], temp_y_predict)
                                            
                            # Keep the MRMR ensemble solution with the highest testing set accuracy
                            temp_nestedCV_acc = np.mean(temp_nestedCV_acc,axis=1)
                            idx      = np.argmax(temp_nestedCV_acc)
                        
                        # Or if using classic MRMR, only one solution is used
                        else:
                            idx = 0
                            
                        # Isolate selected features
                        fkeep = solutions.iloc[0][idx]
                        X_train_us = X_train_us[fkeep]
                        X_test     = X_test[fkeep]
                        
                        # Train and test the model
                        model.fit(X_train_us, np.ravel(y_train_us))
                        y_predict = model.predict(X_test)  
                                        
                        #------------------------------------------------------
                        
                        # Save the trained model (aka the estimator)
                        Estimators.append(model)
                        
                        # Convert predictions to probabilities
                        y_predict_mal_probs = model.predict_proba(X_test)[:,1]
                    
                        # Calculate performance metrics
                        # However, verify that two classes are present in the test set
                        # Otherwise, just set all metrics to NaN so that they are skipped
                        # (When no fat is analyzed, this happens, albeit very infrequently)
                        if len(np.unique(y_test)) > 1:
                            acc    = accuracy_score(y_test, y_predict)
                            recall = recall_score(y_test,y_predict)
                            aucrve = roc_auc_score(y_test,y_predict_mal_probs)
                            f1     = f1_score(y_test,y_predict)
                            precis = precision_score(y_test,y_predict)
                            
                            # Append to prediction and ground truth arrays
                            Y_PREDICT_ALL.append(y_predict)
                            Y_TEST_ALL.append(y_test)
                            Y_PREDICT_PROBS_ALL.append(y_predict_mal_probs)
                            
                        else:
                            acc    = accuracy_score(y_test, y_predict) # Because trained on both classes, accuracy is still a meaningful metric
                            recall = np.nan
                            aucrve = np.nan
                            f1     = np.nan
                            precis = np.nan
                            
                        # Calculate percent accurately classified based on SUBTYPE 
                        subtype_options          = np.unique(SUBTYPES)
                        subtype_accuracy         = np.zeros(len(subtype_options))
                        for kk in range(len(subtype_options)):
                            idx = s_test == subtype_options[kk]
                            if subtype_options[kk] == 'Adi' or subtype_options[kk] == 'Conn' or subtype_options[kk] == 'FCD':
                                truth = 1
                            else:
                                truth = 2
                            
                            # Ignore NaN element warning LOCALLY here; it is okay and the dataframe is fine to handle it 
                            # Credit: https://stackoverflow.com/questions/34955158/what-might-be-the-cause-of-invalid-value-encountered-in-less-equal-in-numpy/34955622
                            with np.errstate(invalid='ignore'):
                                subtype_accuracy[kk] = np.sum(np.equal(y_predict[idx],truth))/len(y_predict[idx])
                            
                        # Save accuracies in dataframe
                        subtype_accuracy_df = pd.DataFrame([subtype_accuracy], columns=subtype_options)
                                        
                        #------------------------------------------------------
                        
                        # If ANALYZE_ADIPOSE == "No", we need to create these columns in the OUTPUT
                        if ANALYZE_ADIPOSE == "No":
                            subtype_tally_train['Adi'] = np.nan
                            subtype_tally_test['Adi']  = np.nan
                            subtype_accuracy_df['Adi'] = np.nan
                            
                        # Append to OUTPUT dataframe
                        new_row = pd.DataFrame(
                            data=
                                [[i+1,FNUM,fkeep,idx,temp_nestedCV_acc,
                                len(np.unique(p_train_us)),len(np.unique(p_test)),
                                np.unique(p_train_us),np.unique(p_test),
                                collections.Counter((pd.DataFrame.to_numpy(y_train_us)).ravel())[1],collections.Counter((pd.DataFrame.to_numpy(y_test)).ravel())[1],
                                collections.Counter((pd.DataFrame.to_numpy(y_train_us)).ravel())[2],collections.Counter((pd.DataFrame.to_numpy(y_test)).ravel())[2],
                                subtype_tally_train['Adi'],   subtype_tally_test['Adi'],    subtype_accuracy_df['Adi'][0],
                                subtype_tally_train['Conn'],  subtype_tally_test['Conn'],   subtype_accuracy_df['Conn'][0],
                                subtype_tally_train['FCD'],   subtype_tally_test['FCD'],    subtype_accuracy_df['FCD'][0],
                                subtype_tally_train['IDCaLG'],subtype_tally_test['IDCaLG'], subtype_accuracy_df['IDCaLG'][0],
                                subtype_tally_train['IDCaIG'],subtype_tally_test['IDCaIG'], subtype_accuracy_df['IDCaIG'][0],
                                subtype_tally_train['IDCaHG'],subtype_tally_test['IDCaHG'], subtype_accuracy_df['IDCaHG'][0],
                                subtype_tally_train['ILCa'],  subtype_tally_test['ILCa'],   subtype_accuracy_df['ILCa'][0],
                                acc,recall,aucrve,f1,precis]],
                            columns=
                                ["split_number","num_features","mrmr_features_selected","mrmr_best_solution_idx","mrmr_best_solution_nestedCV_accuracy",
                                 "num_patients_train","num_patients_test",
                                "patient_ids_train","patient_ids_test",
                                "num_samples_benign_train","num_samples_benign_test",
                                "num_samples_malign_train","num_samples_malign_test",
                                "num_adipose_samples_train",   "num_adipose_samples_test",   "accuracy_adipose",
                                "num_connective_samples_train","num_connective_samples_test","accuracy_connective",
                                "num_fcd_samples_train",       "num_fcd_samples_test",       "accuracy_fcd",
                                "num_idca_lg_samples_train",   "num_idca_lg_samples_test",   "accuracy_idca_lg",
                                "num_idca_ig_samples_train",   "num_idca_ig_samples_test",   "accuracy_idca_ig",
                                "num_idca_hg_samples_train",   "num_idca_hg_samples_test",   "accuracy_idca_hg",
                                "num_ilca_samples_train",      "num_ilca_samples_test",      "accuracy_ilca",
                                "perform_accuracy","perform_recall",
                                "perform_auc","perform_f1","perform_precision"])
                        OUTPUT = OUTPUT.append(new_row, ignore_index=True)
                        
                        # Print results of current split to console
                        print('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b          Split %04i/%04i'%(i+1,N_SPLITS),end='')
                                            
                        #------------------------------------------------------
                        
                        # Dynamically allocate these malignant probabilities
                        for jj in range(len(test)):
                            prob_mal_MATRIX[test[jj]].append(y_predict_mal_probs[jj])
                                            
                        #------------------------------------------------------
                        # Add to the ROC curve
                        # Code below credit of:
                        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
                        
                        # Again, only record ROC curve data is the test set had two classes                
                        if len(np.unique(y_test)) > 1:
                            
                            viz = plot_roc_curve(model, X_test, y_test,
                                                  #name='Fold {}'.format(i),
                                                  name='_nolegend_',
                                                  alpha=0, lw=1, ax=ax,color='black')
                            interp_tpr = np.interp(baseline_fpr, viz.fpr, viz.tpr)
                            interp_tpr[0] = 0.0
                            
                            interp_fpr = np.interp(baseline_tpr, viz.tpr,viz.fpr)
                            interp_fpr[0] = 0.0
        
                            fprs.append(interp_fpr)
                            tprs.append(interp_tpr)
                            aucs.append(viz.roc_auc)
                            
                        else:
                            
                            # Consider this split a "dud"
                            DUD_SPLITS = DUD_SPLITS + 1
        
                    #----------------------------------------------------------
                    # ^End of cross-validation loop
                    #----------------------------------------------------------
                    
                    print('')
                    
                    # First, find the sample that was classified the least
                    min_len = 10000000000
                    for jj in range(len(prob_mal_MATRIX)):
                        temp_len = len(prob_mal_MATRIX[jj])
                        if temp_len < min_len:
                            min_len = temp_len
                                
                    # Second, truncate all sample probabilities to match the shortest one
                    for jj in range(len(prob_mal_MATRIX)):
                        prob_mal_MATRIX[jj] = prob_mal_MATRIX[jj][: min_len]
                    
                    # Convert to dataframe, save probabilities
                    df = pd.DataFrame(prob_mal_MATRIX)
                    fname = "%s%iFeatures_SampleMalignantProbs.csv"%(pth_out,FNUM)
                    df.to_csv(fname)
                    
                    #----------------------------------------------------------
                    # Add ROC curve figure details
                    mean_tpr = np.nanmean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    
                    # Get standard deviations along both dimensions
                    std_tpr = np.nanstd(tprs,axis=0)
                    std_fpr = np.nanstd(fprs,axis=0)
                    
                    # True positive rate (vertical) bounds
                    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                    
                    # False positive rate (horizontal) bounds
                    mean_fpr = np.nanmean(fprs, axis=0)
                    mean_fpr[-1] = 1.0
                    fprs_upper = np.minimum(mean_fpr + std_fpr, 1)
                    fprs_lower = np.maximum(mean_fpr - std_fpr, 0)
                    corrected_fprs_upper = np.interp(baseline_fpr,fprs_upper,baseline_tpr)
                    corrected_fprs_lower = np.interp(baseline_fpr,fprs_lower,baseline_tpr)
        
                    # Combine std upper and lower bounds
                    STD_upper = tprs_upper
                    STD_lower = tprs_lower
                    idx_replace_lower = np.less(corrected_fprs_lower,STD_lower)
                    STD_lower[idx_replace_lower] = corrected_fprs_lower[idx_replace_lower]
                    idx_replace_upper = np.greater(corrected_fprs_upper,STD_upper)
                    STD_upper[idx_replace_upper] = corrected_fprs_upper[idx_replace_upper]
                    ax.fill_between(baseline_fpr, STD_lower, STD_upper, color=CI_colors[cnt_fs-1], alpha=0.2,linewidth=0,
                                    label=r'$\pm$ 1 std.')
                    
                    # Plot the mean ROC curve
                    mean_auc = auc(baseline_fpr, mean_tpr)
                    std_auc = np.std(aucs)
                    ax.plot(baseline_fpr, mean_tpr, color='black',
                            label=r'Mean AUC=%0.2f $\pm$ %0.2f' % (mean_auc, std_auc),
                            lw=2, alpha=1)
                    
                    # Add final figure details, save figure          
                    ax.legend(loc="lower right")
                    ax.set_aspect('equal', adjustable='box')
                    ax.set_xlim([0,1])
                    ax.set_ylim([0,1])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    fname = "%s%iFeatures_ROC_noCBs.png"%(pth_out,FNUM)
                    plt.savefig(fname,bbox_inches='tight',dpi=400)
                            
                    # For another version of the ROC curve figure, add 95% predictive confidence bands
                    
                    # First, find the "confidence" fraction of all ROC curves that are
                    # closest to the mean ROC curve
                    mse = np.zeros(N_SPLITS-DUD_SPLITS)
                    cutoff_n_rocs = np.round(confidence*(N_SPLITS-DUD_SPLITS))
                    for kk in range(N_SPLITS-DUD_SPLITS):
                        mse[kk] = (np.square(mean_tpr - tprs[kk])).mean(axis=0)
                    mse_sorted_idx  = np.argsort(mse)
                    tprs_CI = [tprs[i] for i in mse_sorted_idx]
                    
                    # Preallocate predictive confidence bands
                    # Credit for terminology from Wikipedia: https://en.wikipedia.org/wiki/Prediction_interval#cite_note-1
                    # which references this: Geisser (1993, p. 6): Chapter 2: Non-Bayesian predictive approaches
                    CI_lower = np.ones(len(baseline_fpr))*100000
                    CI_upper = np.zeros(len(baseline_fpr))
                    
                    # Loop through all N_SPLITS ROC curves, updating bands
                    for kk in range(int(cutoff_n_rocs)):
                        temp_tpr = tprs_CI[kk]
                        if type(temp_tpr) != float: # Only continue here if the ROC curve is not NaN
                            idx_replace_lower = np.less(temp_tpr,CI_lower)
                            CI_lower[idx_replace_lower] = temp_tpr[idx_replace_lower]
                            idx_replace_upper = np.greater(temp_tpr,CI_upper)
                            CI_upper[idx_replace_upper] = temp_tpr[idx_replace_upper]
                       
                    # Determine (c)onfidence (b)and AUCs
                    auc_CI_upper = auc(baseline_fpr, CI_upper)
                    auc_CI_lower = auc(baseline_fpr, CI_lower)
                    
                    # Add CBs to figure
                    ax.plot(baseline_fpr,CI_upper,color=CI_colors[cnt_fs-1], alpha=0.2,linestyle='--',label=r'%i%% CB AUC [%.2f-%.2f]'%(100*confidence,auc_CI_lower,auc_CI_upper))
                    ax.plot(baseline_fpr,CI_lower,color=CI_colors[cnt_fs-1], alpha=0.2,linestyle='--',label=r'_nolegend_')
                    
                    # Refresh the legend
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend([handles[0], handles[2], handles[1]],[labels[0], labels[2], labels[1]],loc="lower right")
        
                    # Save second version of figure
                    fname = "%s%iFeatures_ROC_withCBs.png"%(pth_out,FNUM)
                    plt.savefig(fname,bbox_inches='tight',dpi=400)
                    
                    #----------------------------------------------------------
                    # Save individual ROC curves and STD and CI bands just in case
                    # (Potentially needed for DeLong testing later on)
                    fname = "%s%iFeatures_ROCs.pickle"%(pth_out,FNUM)
                    with open(fname,'wb') as f:
                        pickle.dump([baseline_fpr, mean_tpr,STD_lower, STD_upper,CI_lower,CI_upper], f)
                    
                    #----------------------------------------------------------
                    # Save individual split predictions and ground trough
                    # (Potentially needed for DeLong testing later on)
                    fname = "%s%iFeatures_DeLong_AllPredictsGroundtruth.pickle"%(pth_out,FNUM)
                    with open(fname,'wb') as f:
                        pickle.dump([Y_PREDICT_ALL, Y_PREDICT_PROBS_ALL, Y_TEST_ALL], f)
                            
                    #----------------------------------------------------------
                    # Pickle trained models (aka estimators)
                    fname = "%s%iFeatures_Estimators.sav"%(pth_out,FNUM)
                    with open(fname,'wb') as f:
                        pickle.dump(Estimators, f)
                    
                    # To load a pickled file in general:
                    # content = pickle.load(open(fname, 'rb'))
                        
                    # # Pickle ROC curve figure
                    # fname = "%s%iFeatures_ROCfig.pickle"%(pth_out,FNUM)
                    # with open(fname,'wb') as f:
                    #     pickle.dump(fig, f)
                    plt.close('all')
                    
                    # To load a pickled figure specifically:
                    # fig = pickle.load(open(fname, 'rb'))
                    # ax = fig.axes[0]
                    
                    # Save OUTPUT dataframe
                    fname = "%s%iFeatures_Summary.csv"%(pth_out,FNUM)
                    OUTPUT.to_csv(fname)
                    