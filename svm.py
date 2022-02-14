#!/usr/bin/python3

import os
import sys
import math
import random
import argparse
import itertools
import scipy.io
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

"""
Constants
"""
MAX_ITERS = -1
TOL = 1e-3

# Limit the number of blocks to use (set to None to use averaging method)
BLOCK_LIM = 12

gamma_range = {'start': -15, 'stop': 3, 'num': 7, 'base': 2.0}
C_range = {'start': -3, 'stop': 15, 'num': 7, 'base': 2.0}
kernels = ['rbf', 'sigmoid']

def write_readme(path):
    with open(f'{path}/README', 'w') as f:
        f.write('Command line arguments:\n')
        f.write(f'{" ".join(sys.argv)}\n\n')
        
        f.write('Hyperparameter set:\n')
        f.write(f'Gamma: {gamma_range}\n')
        f.write(f'C: {C_range}\n')
        f.write(f'Kernels: {kernels}\n')
        f.write(f'Block Limit: {BLOCK_LIM}\n')

def get_subjects(path):    
    """
    Gets a list of subject IDs and the file suffix, given a path to the data files. 
    """
    
    files = os.listdir(path)
    subjects = sorted([f[:2] for f in files])
    suffix = files[0][2:]
    return subjects, suffix

def scramble_labels(y_data):
    """
    Randomly selects half of the labels in the data to switch to the other class.
    """
    
    classes = sorted(list(set(y_data)))
    
    y_data_copy = y_data.copy()
    to_change = random.sample([i for i, x in enumerate(y_data) if x == classes[0]], k=len(y_data)//4)
    to_change.extend(random.sample([i for i, x in enumerate(y_data) if x == classes[1]], k=len(y_data)//4))
    
    for index in to_change:
        if y_data[index] == classes[0]:
            y_data[index] = classes[1]
        else:
            y_data[index] = classes[0]
    
    # Makes sure labels are scrambled properly
    num_diff = sum(i != j for i, j in zip(y_data, y_data_copy))  
    if num_diff != len(y_data)//2:
        raise ValueError

def extract_subject_data(path, subject, suffix, roi, conds):
    """
    Extracts individual subject data from the .mat files. Rank-orders based on
    specific block number within subject.
    
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
    Lists of voxel data (x_data) separated by individual blocks and the corresponding 
    labels (y_data)
    """
    
    path_to_file = path + subject + suffix
    mat = scipy.io.loadmat(path_to_file)['roi_scanData'][0][roi]
    
    block_count = 0
    x_data = [[] for _ in conds]
    for scan in mat[0]:
        for c, cond in enumerate(conds):
            for block in scan[0][cond][0]:
                block_count += 1
                if BLOCK_LIM and block_count > BLOCK_LIM:
                    continue

                block_data = []

                if len(block[0]) == 8:
                    trs = block[0]
                elif len(block[0]) == 9:
                    trs = block[0][0:8]
                else:
                    trs = block[0][1:9]
                for tr in trs:
                    # Extract all voxel data from individual TRs
                    block_data.append(tr[0][0][0].tolist())

                x_data[c].append(block_data)
                # y_data.append('untrained' if 'untrained' in scan[1][cond][0] else 'trained')

    # Standardize to 8 blocks of 8 TRs
    if BLOCK_LIM == 8:
        if block_count == 8:
            x_data = x_data
        elif block_count == 12:
            x_data[0] = np.concatenate((np.mean([x_data[0][:2], x_data[0][-2:]], axis=0), x_data[0][2:-2]))
            x_data[1] = np.concatenate((np.mean([x_data[1][:2], x_data[1][-2:]], axis=0), x_data[1][2:-2]))
        elif block_count == 16:
            x_data[0] = np.mean([x_data[0][:4], x_data[0][-4:]], axis=0)
            x_data[1] = np.mean([x_data[1][:4], x_data[1][-4:]], axis=0)
        else:
            print("Undefined number of blocks!")
    elif BLOCK_LIM == 12:
        if block_count == 12:
            x_data = x_data
        elif block_count == 16:
            x_data[0] = np.concatenate((np.mean([x_data[0][:2], x_data[0][-2:]], axis=0), x_data[0][2:-2]))
            x_data[1] = np.concatenate((np.mean([x_data[1][:2], x_data[1][-2:]], axis=0), x_data[1][2:-2]))
        else:
            print("Undefined number of blocks!")

    x_data_f, y_data_f = [], []

    # Flatten and transform data into units by voxel
    for c in range(len(conds)):
        trs = []
        for a in x_data[c]:
            trs.extend(a)
        trs = np.array(trs).T
        x_data_f.extend(trs)
        y_data_f.extend([c for _ in trs])

    data = {'x': x_data_f, 'y': y_data_f}
    return data

def generate_dataset(subjects, path, suffix, roi, conds):
    """
    Generates entire dataset from subject list, partitioned by subject.
    
    Parameters
    ----------
    subjects: list
        a list of subject IDs to extract data from
    path: str
        the path to the data files
    suffix: str
        ending suffix of the data filename
    roi: int
        0 for V1 data, 1 for MT data
    conds: list
        list of integers specifying the conditional datasets to extract
        (0 for trained_cp, 1 for trained_ip, 2 for untrained_cp, 3 for untrained_ip)
    
    Returns
    -------
    dict
        voxel data with subject key
    dict
        label data with subject key
    """
    
    x_data = []
    x_data_indices = []
    y_data_by_subject = dict()
    
    for subject in subjects:
        subject_data = extract_subject_data(path, subject, suffix, roi, conds)
        x_data_indices.append(len(x_data))
        y_data_by_subject[subject] = subject_data['y']
        
        x_data.extend(subject_data['x'])
    
    # MinMaxScaler scales each feature to values between 0 and 1 among all x data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_standardized = scaler.fit_transform(x_data)
    
    # Sorts block data into respective subject
    x_data_by_subject = dict()
    for i in range(len(subjects)):
        subject = subjects[i]
        start_index = x_data_indices[i]
        end_index = x_data_indices[i+1] if i+1 < len(x_data_indices) else len(x_data)
        
        x_data_by_subject[subject] = x_standardized[start_index:end_index]
    
    x_data_by_subject = {k: v for k, v in sorted(x_data_by_subject.items(), key=lambda item: len(item[1]))}
    y_data_by_subject = {k: v for k, v in sorted(y_data_by_subject.items(), key=lambda item: len(item[1]))}
    
    return x_data_by_subject, y_data_by_subject

def split_dataset(x_data, y_data, inner_subjects, outer_subject, scramble):
    """
    Splits voxel and label data into appropriate testing and training data for nested
    cross-validation with SVM.
    
    Parameters
    ----------
    x_data: dict
        voxel data with subject key
    y_data: dict
        label data with subject key
    inner_subjects: list
        list of subject IDs of the inner test subjects
    outer_subject: str
        the ID of the outer test subject
    scramble: boolean, optional
        whether or not to scramble the labels when training, 
        default is False
    
    Returns
    -------
    list
        blocks of voxel data for training use
    list
        training labels for respective blocks
    list
        blocks of voxel data from inner test subject(s) for testing use
    list 
        labels for inner test subject(s)
    list
        blocks of voxel data from outer test subject for testing use
    list
        labels for outer test subject    
    """
    
    x_train = []
    y_train = []
    
    x_test_inner = []
    y_test_inner = []
    
    x_test_outer = []
    y_test_outer = []
    
    for subject in x_data.keys():
        if subject == outer_subject:
            x_test_outer.extend(x_data[subject])
            y_test_outer.extend(y_data[subject])
        elif subject in inner_subjects:
            x_test_inner.extend(x_data[subject])
            y_test_inner.extend(y_data[subject])
        else:
            x_train.extend(x_data[subject])
            if scramble:
                y_scrambled = y_data[subject].copy()
                scramble_labels(y_scrambled)
                y_train.extend(y_scrambled)
            else:
                y_train.extend(y_data[subject])
            
    return x_train, y_train, x_test_inner, y_test_inner, x_test_outer, y_test_outer

def get_optimal_run(x_train, y_train, x_test, y_test, kernels, gamma_range, C_range, n_cores):
    """
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
    """
    
    gamma_vals = np.logspace(gamma_range['start'], gamma_range['stop'], gamma_range['num'], base=gamma_range['base'])
    C_vals = np.logspace(C_range['start'], C_range['stop'], C_range['num'], base=C_range['base'])
    param_grid = ParameterGrid({'kernel': kernels, 'gamma': gamma_vals, 'C': C_vals})
    
    def evaluate_acc(params):
        svclassifier = SVC(kernel=params['kernel'], gamma=params['gamma'], C=params['C'], max_iter=MAX_ITERS, tol=TOL)
        svclassifier.fit(x_train, y_train)
        
        return svclassifier.score(x_test, y_test), params

    search = Parallel(n_jobs=n_cores)(delayed(evaluate_acc)(params) for params in list(param_grid))
    search.sort(key=lambda x: tuple(x[1].values()))

    best_acc = 0
    best_params = None
    for result in search:
        if result[0] > best_acc:
            best_acc = result[0]
            best_params = result[1]
            
    return best_params, best_acc

def train(data_params, grid_params, num_inner=1, scramble=False, n_cores=1):
    """
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
    num_inner: int
        number of inner subjects to test classifier on,
        default is 1
    scramble: boolean, optional
        whether or not to scramble the labels when training, 
        default is False
        
    Returns
    -------
    list
        data of inner subject combination testing accuracy
    list
        data of outer subject testing accuracy
    """
    
    subjects, suffix = get_subjects(data_params['path'])
    x_data, y_data = generate_dataset(subjects, data_params['path'], suffix, data_params['roi'], data_params['conds'])
        
    inner_acc_report = []
    outer_acc_report = []
    for outer_subject in tqdm(subjects, leave=(not scramble)):
        opt_inner_params = None
        opt_inner_acc = -1

        inner_subjects = [s for s in subjects if s != outer_subject]
        
        # Iterate through inner folds
        for inner_test_subjects in itertools.combinations((inner_subjects), num_inner):
            inner_test_subjects = list(inner_test_subjects)
            
            x_train, y_train, x_test_inner, y_test_inner, x_test_outer, y_test_outer = split_dataset(x_data, y_data, inner_test_subjects, outer_subject, scramble)

            # Gets optimal params for training dataset from grid search
            opt_params, inner_acc = get_optimal_run(x_train, y_train, x_test_inner, y_test_inner, grid_params['kernels'], grid_params['gamma'], grid_params['C'], n_cores) 
            assert(len(set(y_train)) == 2)
            assert(len(set(y_test_inner)) == 2)
            assert(len(set(y_test_outer)) == 2)
            
            if opt_params is not None:
                # Trains model using optimal params for this set
                svclassifier = SVC(kernel=opt_params['kernel'], gamma=opt_params['gamma'], C=opt_params['C'], max_iter=MAX_ITERS, tol=TOL)
                svclassifier.fit(x_train, y_train)
                
                # Track optimal params among all inner folds
                if inner_acc > opt_inner_acc:
                    opt_inner_acc = inner_acc
                    opt_inner_params = opt_params
                    
                inner_acc_report.append(inner_acc)

        # Test outer subject using optimal params across all inner folds
        x_train, y_train, _, _, x_test_outer, y_test_outer = split_dataset(x_data, y_data, [], outer_subject, scramble)
        svclassifier = SVC(kernel=opt_inner_params['kernel'], gamma=opt_inner_params['gamma'], C=opt_inner_params['C'], max_iter=MAX_ITERS, tol=TOL)
        svclassifier.fit(x_train, y_train)
        
        outer_acc = svclassifier.score(x_test_outer, y_test_outer)
        outer_acc_report.append(outer_acc)

        np.save(f'{output_path}/outer_accs.npy', outer_acc_report)
        np.save(f'{output_path}/inner_accs.npy', inner_acc_report)
    
    return inner_acc_report, outer_acc_report

def permutation(data_params, grid_params, inner_dist, outer_dist, runs=30, n_cores=1, output_path=''):
    """
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
        number of runs to perform, default is 30
    n_cores: int
        number of CPU cores for parallelization, default is 1
    num_rank_blocks: int
        the number of blocks to rank order by, from #1 to #(num_rank_blocks)
        default is 1
    output_path: str
        path to which files should be saved,
        default is current directory
    """
        
    for _ in tqdm(range(runs)):
        inner_result, outer_result = train(data_params, grid_params, scramble=True, n_cores=n_cores)

        inner_dist.append(inner_result)
        outer_dist.append(outer_result)

        np.save(f'{output_path}/outer_perms.npy', outer_dist)
        np.save(f'{output_path}/inner_perms.npy', inner_dist)

"""
Parse arguments and run appropriate analysis
"""
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

write_readme(output_path)

if args.permute:
    try:
        inner_dist = np.load(f'{output_path}/inner_perms.npy').tolist()
        outer_dist = np.load(f'{output_path}/outer_perms.npy').tolist()
    except FileNotFoundError:
        print('Creating new permutation distribution...')
        inner_dist = []
        outer_dist = []

    permutation(data_params, grid_params, inner_dist, outer_dist, runs=run_count, output_path=output_path, n_cores=num_cores)

    np.save(f'{output_path}/outer_perms.npy', outer_dist)
    np.save(f'{output_path}/inner_perms.npy', inner_dist)

else:
    inner_accs = []
    outer_accs = []
    inner_result, outer_result = train(data_params, grid_params, n_cores=num_cores)
        
    inner_accs.append(inner_result)
    outer_accs.append(outer_result)

    np.save(f'{output_path}/outer_accs.npy', outer_accs)
    np.save(f'{output_path}/inner_accs.npy', inner_accs)
