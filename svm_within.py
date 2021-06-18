#!/usr/bin/python

import os
import sys
import copy
import time
import math
import json
import random
import argparse
import itertools
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

'''
Constants
'''

gamma_range = {'start': -15, 'stop': 3, 'num': 19, 'base': 2.0}
C_range = {'start': -3, 'stop': 15, 'num': 19, 'base': 2.0}
kernels = ['rbf', 'sigmoid']

def get_subjects(path):
    
    '''
    Gets a list of subject IDs and the file suffix, given a path to the data files. 
    
    Note: subject ID must be only 2 characters for this to work, and all data files
    must have same suffix.
    
    Parameters
    ----------
    path: str
        directory to the data files
        
    Returns
    -------
    list
        a list of subject IDs
    str
        the suffix to the filenames
    '''
    
    files = os.listdir(path)
    subjects = [f[:2] for f in files]
    suffix = files[0][2:]
        
    subjects.sort()
    
    return subjects, suffix

def scramble_labels(y_data):
    
    '''
    Randomly selects half of the labels in the data to switch to the other class.
    
    Parameters
    ----------
    y_data: array-like
        label data to scramble
    '''
    
    classes = list(set(y_data))
    classes.sort()
    
    y_data_copy = y_data.copy()
    
    labels_0 = [i for i, x in enumerate(y_data) if x == classes[0]]
    labels_1 = [i for i, x in enumerate(y_data) if x == classes[1]]
    to_change = random.sample(labels_0, k=len(labels_0)//2)
    to_change.extend(random.sample(labels_1, k=len(labels_1)//2))
    
    for index in to_change:
        if y_data[index] == classes[0]:
            y_data[index] = classes[1]
        else:
            y_data[index] = classes[0]

def extract_tr_subject_data(path, subject, suffix, roi, conds):
    
    '''
    Extracts individual subject data from the .mat files.
    
    Parameters
    ----------
    path: str
        directory to data files
    subject: str
        ID of subject to load data for
    suffix: str
        ending suffix of the data filename
    roi: int
        0 for V1 data, 1 for MT data
    conds: list
        list of integers specifying the conditional datasets to extract
        (0 for trained_cp, 1 for trained_ip, 2 for untrained_cp, 3 for untrained_ip)
        
    Returns
    -------
    Lists of voxel data (x_data) separated by individual TRs and the corresponding labels (y_data)
    '''
    
    x_data = []
    y_data = []
    
    path_to_file = path + subject + suffix
    mat = scipy.io.loadmat(path_to_file)['roi_scanData'][0][roi]
    block_indices = []

    # Extract all TR data from all blocks from all scans
    for scan in range(len(mat[0])):

        for cond in conds:
            
            for block in range(len(mat[0][scan][0][cond][0])):
                
                for tr in range(len(mat[0][scan][0][cond][0][block][0])):

                    tr_data = mat[0][scan][0][cond][0][block][0][tr][0][0][0].tolist()
                    x_data.append(tr_data)
                    y_data.append(mat[0][scan][1][cond][0].replace('_post', ''))
                
                block_indices.append(len(x_data))
    
    # MinMaxScaler scales each feature to values between 0 and 1 among all x data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_data = scaler.fit_transform(x_data)
    
    x_data_by_block = {}
    y_data_by_block = {}
    for block_id, idx in enumerate(block_indices):
        if idx == block_indices[0]:
            x_data_by_block[block_id] = x_data[0:idx]
            y_data_by_block[block_id] = y_data[0:idx]
        else:
            x_data_by_block[block_id] = x_data[block_indices[block_id-1]:idx]
            y_data_by_block[block_id] = y_data[block_indices[block_id-1]:idx]
    
    return x_data_by_block, y_data_by_block

def get_optimal_run(x_train, y_train, x_test, y_test, kernels, gamma_range, C_range, n_cores):
    
    '''
    Gets best hyperparameters (kernel, C, and gamma values) that optimize SVM's predictions for given
    x and y test dataset.
    
    Parameters
    ----------
    x_train: array-like
        dataset of block data used to train classifier
    y_train: array-like
        dataset of label data used to train classifier
    x_test: array-like
        testing dataset of block data used to optimize hyperparameters on
    y_test: array-like
        testing dataset of label data used to optimize hyperparameters on
    kernels: list
        kernels to test (recommended options are 'linear', 'rbf', and 'sigmoid')
    gamma_range: dict
        dict that specifies the range of values of gamma to test; should include start, stop to range,
        num of values, and the exponential base
    C_range: dict
        dict that specifies the range of values of C to test; should include start, stop to range,
        num of values, and the exponential base
        
    Returns
    -------
    dict
        best combination of parameters found from grid search
    float
        best accuracy obtained from testing
    '''
    
    gamma_vals = np.logspace(gamma_range['start'], gamma_range['stop'], gamma_range['num'], base=gamma_range['base'])
    C_vals = np.logspace(C_range['start'], C_range['stop'], C_range['num'], base=C_range['base'])

    param_grid = ParameterGrid({'kernel': kernels, 'gamma': gamma_vals, 'C': C_vals})

    def evaluate_acc(params):
        svclassifier = SVC(kernel=params['kernel'], gamma=params['gamma'], C=params['C'], max_iter=-1)
        svclassifier.fit(x_train, y_train)
        
        return svclassifier.score(x_test, y_test), params

    search = Parallel(n_jobs=n_cores)(delayed(evaluate_acc)(params) for params in list(param_grid))

    best_acc = 0
    best_params = None

    for result in search:
        if result[0] > best_acc:
            best_acc = result[0]
            best_params = result[1]
            
    return best_params, best_acc

def train(data_params, grid_params, scramble=False, n_cores=1):
    
    '''
    Trains and tests the classifier for accuracy using SVMs.
    
    Parameters
    ----------
    data_params: dict
        path: str
            the path to the data files
        roi: int
            0 for V1 data, 1 for MT data
        conds: list
            list of integers specifying the conditional datasets to extract
            (0 for trained_cp, 1 for trained_ip, 2 for untrained_cp, 3 for untrained_ip)   
    grid_params: dict
        kernels: list
            kernels to test (recommended options are 'linear', 'rbf', and 'sigmoid')
        gamma: dict
            dict that specifies the range of values of gamma to test; should include start, stop to range,
            num of values, and the exponential base
        C: dict
            dict that specifies the range of values of C to test; should include start, stop to range,
            num of values, and the exponential base
    scramble: boolean, optional
        whether or not to scramble the labels when training, 
        default is False
        
    Returns
    -------
    DataFrame
        data of inner subject combination testing accuracy
    DataFrame
        data of outer subject testing accuracy
    '''
    
    subjects, suffix = get_subjects(data_params['path'])
    inner_result = {}
    outer_result = {}
    
    start_time = time.time()
    for subject in subjects:
        
        inner_result[subject] = []
        outer_result[subject] = []
        
        print(f"Currently on subject {subject}.")
        
        x_data, y_data = extract_tr_subject_data(path, subject, suffix, roi, conds)
        samples = x_data.keys()
        for outer_sample in samples:
            
            inner_sample = [s for s in samples if s != outer_sample]
            
            opt_inner_acc = -1
            opt_inner_params = None
            for inner_sample in inner_sample:
                
                x_train, y_train, x_test, y_test = [], [], [], []
                for sample in samples:
                    if sample == inner_sample:
                        x_test.extend(x_data[sample])
                        y_test.extend(y_data[sample])
                    elif sample != outer_sample:
                        x_train.extend(x_data[sample])
                        y_train.extend(y_data[sample])
            
                opt_params, inner_acc = get_optimal_run(x_train, y_train, x_test, y_test, grid_params['kernels'], grid_params['gamma'], grid_params['C'], n_cores=n_cores) 
                if inner_acc > opt_inner_acc:
                    opt_inner_acc = inner_acc
                    opt_inner_params = opt_params
                    
                inner_result[subject].append(inner_acc)
                    
            x_train, y_train, x_test, y_test = [], [], [], []
            for sample in samples:
                if sample == outer_sample:
                    x_test.extend(x_data[sample])
                    y_test.extend(y_data[sample])
                else:
                    x_train.extend(x_data[sample])
                    y_train.extend(y_data[sample])
                    
            svclassifier = SVC(kernel=opt_inner_params['kernel'], gamma=opt_inner_params['gamma'], C=opt_inner_params['C'], max_iter=-1)
            svclassifier.fit(x_train, y_train)
            outer_acc = svclassifier.score(x_test, y_test)
            outer_result[subject].append(outer_acc)
            
    end_time = time.time()
    exec_time = end_time - start_time
    minutes = exec_time // 60
    print(f"Last turn took {round(minutes, 3)} minutes.")

    return inner_result, outer_result

def permutation(data_params, grid_params, inner_dist, outer_dist, runs=100, output_path='', n_cores=1):
    
    '''
    Performs a specified number of runs where data labels are scrambled.
    
    Parameters
    ----------
    data_params: dict
        contains specifications for data processing (see train method for documentation)
    grid_params: dict
        contains values for grid search (see train method for documentation)
    inner_dist: list
        holds accuracy values for individual inner subject tests
    outer_dist: list
        holds accuracy values for individual outer subject tests
    runs: int
        number of runs to perform, default is 100
    '''
    
    subjects, _ = get_subjects(data_params['path'])
    if inner_dist is None and outer_dist is None:
        inner_dist = {subject: [] for subject in subjects}
        outer_dist = {subject: [] for subject in subjects}
        
    for n in range(runs):
        print(f'On run #{n+1} of {runs}.')
        inner_accs, outer_accs = train(data_params, grid_params, scramble=True, n_cores=n_cores)
        
        for subject in inner_accs.keys():
            inner_dist[subject].append(inner_accs)
            outer_dist[subject].append(outer_accs)

        with open(f'{output_path}/within_outer_perms.json', 'w') as f:
            json.dump(outer_dist, f)
        with open(f'{output_path}/within_inner_perms.json', 'w') as f:
            json.dump(inner_dist, f)

parser = argparse.ArgumentParser()
parser.add_argument("indir", metavar="INDIR", help="directory to input data")
parser.add_argument("outdir", metavar="OUTDIR", help="directory to output data")
parser.add_argument("-p", "--permute", action="store_true", help="permute training data")

parser.add_argument("-n", "--run-count", action="store", default=1, type=int, help="run RUN_COUNT permutations")

parser.add_argument("-r", "--roi", action="store", type=int, help="ROI: V1 is 0, MT is 1")
parser.add_argument("-c", "--cond", action="store", type=int, help="conditions: 0 is CP/Large, 1 is IP/Small")

parser.add_argument("--cores", action="store", default=1, type=int, help="use CORES number of CPU cores for parallelization")

args, extra = parser.parse_known_args()

if args.roi is None:
    print('Need to specify ROI.')
    exit()
if args.cond is None:
    print('Need to specify cond.')
    exit()

roi = args.roi
conds = [args.cond, args.cond+2]
run_count = args.run_count
num_cores = args.cores

output_path = args.outdir
if output_path[-1] != '/':
    output_path += '/'

path = args.indir
if path[-1] != '/':
    path += '/'

data_params = {'path': path, 'roi': roi, 'conds': conds}
grid_params = {'gamma': gamma_range, 'C': C_range, 'kernels': kernels}

if args.permute:
    try:
        with open(f'{output_path}/within_outer_perms.json', 'r') as f:
            outer_dist = json.load(f)
        with open(f'{output_path}/within_inner_perms.json', 'r') as f:
            inner_dist = json.load(f)
    except FileNotFoundError:
        print('Creating new permutation distribution...')
        inner_dist = None
        outer_dist = None

    permutation(data_params, grid_params, inner_dist, outer_dist, runs=run_count, output_path=output_path, n_cores=num_cores)

    with open(f'{output_path}/within_outer_perms.json', 'w') as f:
            json.dump(outer_dist, f)
    with open(f'{output_path}/within_inner_perms.json', 'w') as f:
        json.dump(inner_dist, f)

else:
    start = time.time()
    inner_result, outer_result = train(data_params, grid_params, n_cores=num_cores)
    end = time.time()
    print(f'Done in {round((end-start)/60, 2)} minutes.')

    with open(f'{output_path}/within_outer_accs.json', 'w') as f:
        json.dump(outer_result, f)

    with open(f'{output_path}/within_inner_accs.json', 'w') as f:
        json.dump(inner_result, f)
