import os
import copy
import time
import math
import random
import itertools
import scipy.io
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

"""
Simulation settings
"""

path = 'scans/ip_only_data.mat'
roi = 1                            # V1-roi: 0, MT-roi: 1
n_cores = 4

def df_to_arr(df):
    vals = []
    for _, row in df.iterrows():
        vals.extend(row.tolist())
    return np.array([x for x in vals if str(x) != 'nan'])

def get_subjects(path):
    mat = scipy.io.loadmat(path)['ipOnly_roi_scanData']
    
    subjects = list(set([s[0][:2] for s in mat[1]]))
    subjects.sort()
    return subjects

def score_params(params, x_train, y_train, x_test, y_test):
    svclassifier = SVC(kernel=params['kernel'], gamma=params['gamma'], C=params['C'], max_iter=-1)
    svclassifier.fit(x_train, y_train)
    curr_acc = svclassifier.score(x_test, y_test)
    return curr_acc, params

def get_optimal_run(x_train, y_train, x_test, y_test, kernels, gamma_range, C_range):
    gamma_vals = np.logspace(gamma_range['start'], gamma_range['stop'], gamma_range['num'], base=gamma_range['base'])
    C_vals = np.logspace(C_range['start'], C_range['stop'], C_range['num'], base=C_range['base'])

    param_grid = ParameterGrid({'kernel': kernels, 'gamma': gamma_vals, 'C': C_vals})
    
    best_acc = 0
    best_params = None
    results = Parallel(n_jobs=n_cores)(delayed(score_params)(params, x_train, y_train, x_test, y_test) for params in list(param_grid))

    # Tests each parameter combination to find best one for given testing data
    results.sort(key=lambda x: [x[1]['C'], x[1]['gamma'], x[1]['kernel']])
    for result in results:
        curr_acc, params = result
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_params = params
    
    return best_params, best_acc

def extract_tr_subject_data(mat, subject, roi):
    x_data = []
    y_data = []
    
    split = 0
    test_trs, pre_test_trs, post_test_trs = [], [], []

    subject_runs = []
    for i in range(len(mat[1])):
        if mat[1][i][0][:2] == subject:
            subject_runs.append(i)

    # Extract all TR data from all blocks from all scans
    is_post_run = True
    for sub_run in subject_runs:
        for scan in range(len(mat[0][sub_run][0][roi][0])):
            for cond in range(len(mat[0][sub_run][0][roi][0][scan][0])):
                blocks = [x for x in range(len(mat[0][sub_run][0][roi][0][scan][0][cond][0]))]
                for block in blocks:
                    for tr in range(len(mat[0][sub_run][0][roi][0][scan][0][cond][0][block][0])):
                        tr_data = mat[0][sub_run][0][roi][0][scan][0][cond][0][block][0][tr][0][0][0].tolist()
                        
                        if tr == 0 or tr == len(mat[0][sub_run][0][roi][0][scan][0][cond][0][block][0]) - 1:
                            test_trs.append(len(x_data))
                            
                        x_data.append(tr_data)
                        y_data.append(mat[0][sub_run][0][roi][0][scan][1][cond][0].replace('_large', '').replace('_small', ''))

        if is_post_run:
            post_test_trs = test_trs.copy()
            is_post_run = False
        else:
            pre_test_trs = test_trs.copy()[len(post_test_trs):]
    
    # MinMaxScaler scales each feature to values between 0 and 1 among all x data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_standardized = scaler.fit_transform(x_data)

    # Separate training, pre-testing, and post-testing data
    pre_test_x_data, pre_test_y_data = [], []
    for i in pre_test_trs:
        pre_test_x_data.append(x_standardized[i])
        pre_test_y_data.append(y_data[i])
    post_test_x_data, post_test_y_data = [], []
    for i in post_test_trs:
        post_test_x_data.append(x_standardized[i])
        post_test_y_data.append(y_data[i])

    train_x_data = []
    train_y_data = []
    for i in range(len(x_standardized)):
        if i not in test_trs:
            train_x_data.append(x_standardized[i])
            train_y_data.append(y_data[i])
    
    pre_data = {'train_x': train_x_data, 'train_y': train_y_data, 'test_x': pre_test_x_data, 'test_y': pre_test_y_data}
    post_data = {'train_x': [], 'train_y': [], 'test_x': post_test_x_data, 'test_y': post_test_y_data}
    return pre_data, post_data

def split_tr_dataset(data):
    train_x, inner_test_x, outer_test_x, train_y, inner_test_y, outer_test_y = [], [], [], [], [], []
    
    # Split test data into outer loop TRs and inner loop TRs
    test_indices = [i for i in range(len(data['test_x']))]
    test_indices, outer_test_indices, test_y, outer_test_y = train_test_split(test_indices, data['test_y'], test_size=2, stratify=data['test_y'])

    # Separate outer loop TRs and remove same block TRs from inner test list
    for i in outer_test_indices:
        if i%2 == 0:
            index_to_remove = test_indices.index(i+1)
        else:
            index_to_remove = test_indices.index(i-1)
        del test_y[index_to_remove]
        del test_indices[index_to_remove]
        train_x.append(data['test_x'][index_to_remove])
        train_y.append(data['test_y'][index_to_remove])

    # Split test data into inner loop TRs and rest
    train_indices, inner_test_indices, _, inner_test_y = train_test_split(test_indices, test_y, test_size=6, stratify=test_y)
    
    outer_test_x = [data['test_x'][i] for i in outer_test_indices]
    inner_test_x = [data['test_x'][i] for i in inner_test_indices]
    
    # Set up rest of training data with leftover unused TRs
    train_x.extend([data['test_x'][i] for i in train_indices])
    train_x.extend(data['train_x'])
    train_y.extend([data['test_y'][i] for i in train_indices])
    train_y.extend(data['train_y'])

    return train_x, inner_test_x, outer_test_x, train_y, inner_test_y, outer_test_y

def train_within_subjects_combined(data_params, grid_params, runs=200):
    subjects = get_subjects(data_params['path'])
    mat = scipy.io.loadmat(data_params['path'])['ipOnly_roi_scanData']
    
    # Sets up DataFrames used to track inner and outer subject test accuracies
    cols = [n for n in range(runs)]
    inner_acc_report = pd.DataFrame(index=subjects, columns=cols)
    outer_acc_report_post = pd.DataFrame(index=subjects, columns=cols)
    outer_acc_report_pre = pd.DataFrame(index=subjects, columns=cols)
    
    for subject in subjects:
        print(f"Currently on subject {subject}.")
        start_time = time.time()
        
        subject_data_pre, subject_data_post = extract_tr_subject_data(mat, subject, roi)
        
        x_train_pre, x_test_inner_pre, x_test_outer_pre, y_train_pre, y_test_inner_pre, y_test_outer_pre = split_tr_dataset(subject_data_pre)
        x_train_post, x_test_inner_post, x_test_outer_post, y_train_post, y_test_inner_post, y_test_outer_post = split_tr_dataset(subject_data_post)
        
        x_train = x_train_pre + x_train_post
        x_test_inner = x_test_inner_pre + x_test_inner_post
        x_test_outer = x_test_outer_pre + x_test_outer_post
        y_train = y_train_pre + y_train_post
        y_test_inner = y_test_inner_pre + y_test_inner_post
        y_test_outer = y_test_outer_pre + y_test_outer_post
        
        print(f"Training data size: {len(x_train)}")
        print(f"Inner testing data size: {len(x_test_inner)}")
        print(f"Outer testing data size: {len(x_test_outer)}")
        
        for run in range(runs):
            if (run+1) % 10 == 0:
                print(f"On run #{run+1} of {runs}.")
                
            x_train_pre, x_test_inner_pre, x_test_outer_pre, y_train_pre, y_test_inner_pre, y_test_outer_pre = split_tr_dataset(subject_data_pre)
            x_train_post, x_test_inner_post, x_test_outer_post, y_train_post, y_test_inner_post, y_test_outer_post = split_tr_dataset(subject_data_post)
            
            # Combine data for training and inner testing
            x_train = x_train_pre + x_train_post
            x_test_inner = x_test_inner_pre + x_test_inner_post
            y_train = y_train_pre + y_train_post
            y_test_inner = y_test_inner_pre + y_test_inner_post
            
            # Gets optimal params for training dataset from grid search
            opt_params, inner_acc = get_optimal_run(x_train, y_train, x_test_inner, y_test_inner, grid_params['kernels'], grid_params['gamma'], grid_params['C']) 

            # Trains model using optimal params for this set
            svclassifier = SVC(kernel=opt_params['kernel'], gamma=opt_params['gamma'], C=opt_params['C'], max_iter=-1)
            svclassifier.fit(x_train, y_train)
            
            outer_acc_pre = svclassifier.score(x_test_outer_pre, y_test_outer_pre)
            outer_acc_post = svclassifier.score(x_test_outer_post, y_test_outer_post)
            
            # Logs inner and outer subject accuracy data in DataFrame
            inner_acc_report.at[subject, run] = inner_acc
            outer_acc_report_pre.at[subject, run] = outer_acc_pre
            outer_acc_report_post.at[subject, run] = outer_acc_post
        
        inner_acc_report.to_csv(f'output/ip_combined/inner_accs_within.csv')
        outer_acc_report_pre.to_csv(f'output/ip_combined/outer_accs_pre_within.csv')
        outer_acc_report_post.to_csv(f'output/ip_combined/outer_accs_post_within.csv')

        # Prints how long it took for last outer subject test
        end_time = time.time()
        exec_time = end_time - start_time
        minutes = exec_time // 60
        seconds = exec_time % 60
        print(f"Last turn took {minutes} minutes and {seconds} seconds.")
        
    return inner_acc_report, outer_acc_report_pre, outer_acc_report_post

gamma_range = {'start': -15, 'stop': 3, 'num': 19, 'base': 2.0}
C_range = {'start': -3, 'stop': 15, 'num': 19, 'base': 2.0}
kernels = ['rbf', 'sigmoid']

data_params = {'path': path, 'roi': roi}
grid_params = {'gamma': gamma_range, 'C': C_range, 'kernels': kernels}

start = time.time()

inner_accs, outer_accs_pre, outer_accs_post = train_within_subjects_combined(data_params, grid_params)
inner_accs.to_csv(f'output/ip_combined/inner_accs_within.csv')
outer_accs_pre.to_csv(f'output/ip_combined/outer_accs_pre_within.csv')
outer_accs_post.to_csv(f'output/ip_combined/outer_accs_post_within.csv')

end = time.time()
print(f'Done in {round((end-start)/60, 2)} minutes.')