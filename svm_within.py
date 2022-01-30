#!/usr/bin/python3

import os
import sys
import random
import argparse
import scipy.io
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

"""
Constants
"""
gamma_range = {'start': -15, 'stop': 3, 'num': 19, 'base': 2.0}
C_range = {'start': -3, 'stop': 15, 'num': 19, 'base': 2.0}
kernels = ['rbf', 'sigmoid']

def write_readme(path):
    with open(f'{path}/README', 'w') as f:
        f.write('Command line arguments:\n')
        f.write(f'{" ".join(sys.argv)}\n\n')
        f.write('Hyperparameter set:\n')
        f.write(f'Gamma: {gamma_range}\n')
        f.write(f'C: {C_range}\n')
        f.write(f'Kernels: {kernels}\n')

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
    """
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
    """
    
    x_data = []
    y_data = []
    
    path_to_file = path + subject + suffix
    mat = scipy.io.loadmat(path_to_file)['roi_scanData'][0][roi]
    
    # Extract all TR data from all blocks from all scans
    test_TRs = []
    for scan in mat[0]:
        for cond in conds:
            for block in scan[0][cond][0]:
                for tr_idx, tr in enumerate(block[0]):
                    tr_data = tr[0][0][0].tolist()
                    
                    if tr_idx == 0 or tr_idx == len(block[0]) - 1:
                        test_TRs.append(len(x_data))
                        
                    x_data.append(tr_data)
                    y_data.append(scan[1][cond][0].replace('_post', ''))
    
    # MinMaxScaler scales each feature to values between 0 and 1 among all x data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_standardized = scaler.fit_transform(x_data)        
    
    test_x_data = []
    test_y_data = []
    for i in test_TRs:
        test_x_data.append(x_standardized[i])
        test_y_data.append(y_data[i])
    
    train_x_data = []
    train_y_data = []
    for i in range(len(x_standardized)):
        if i not in test_TRs:
            train_x_data.append(x_standardized[i])
            train_y_data.append(y_data[i])
    
    data = {'train_x': train_x_data, 'train_y': train_y_data, 'test_x': test_x_data, 'test_y': test_y_data}
    return data

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

def split_tr_dataset(data, scramble):
    """
    Splits the TR dataset into inner testing, outer testing, and training folds.
    
    Parameters
    ----------
    data: dict
        dictionary that contains data split into training and testing partition
    scramble: boolean
        whether or not to scramble the labels when training
        
    Returns
    -------
    list
        TR voxel data for training use
    list
        TR voxel data for inner fold testing use
    list
        TR voxel data for outer fold testing use
    list 
        labels for training use
    list
        labels for inner fold testing use
    list
        labels for outer fold testing use
    """
    
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

    if scramble:
        scramble_labels(train_y)
    
    return train_x, inner_test_x, outer_test_x, train_y, inner_test_y, outer_test_y

def train(data_params, grid_params, runs=200, scramble=False, n_cores=1):    
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
    runs: int
        number of runs to test on for each subject
    scramble: boolean, optional
        whether or not to scramble the labels when training, 
        default is False
        
    Returns
    -------
    DataFrame
        data of inner subject combination testing accuracy
    DataFrame
        data of outer subject testing accuracy
    """
    
    subjects, suffix = get_subjects(data_params['path'])
    inner_result = []
    outer_result = []
    
    for subject in subjects:
        print(f"Currently on subject {subject}.")

        subject_data = extract_tr_subject_data(path, subject, suffix, roi, conds)
        x_train, x_test_inner, x_test_outer, y_train, y_test_inner, y_test_outer = split_tr_dataset(subject_data, scramble)
        print(f"Training data size: {len(x_train)}")
        print(f"Inner testing data size: {len(x_test_inner)}")
        print(f"Outer testing data size: {len(x_test_outer)}")

        inner_result_subject = []
        outer_result_subject = []
        for _ in tqdm(range(runs), leave=(not scramble)):
            x_train, x_test_inner, x_test_outer, y_train, y_test_inner, y_test_outer = split_tr_dataset(subject_data, scramble)
            
            # Gets optimal params for training dataset from grid search
            opt_params, inner_acc = get_optimal_run(x_train, y_train, x_test_inner, y_test_inner, grid_params['kernels'], grid_params['gamma'], grid_params['C'], n_cores) 

            # Trains model using optimal params for this set
            svclassifier = SVC(kernel=opt_params['kernel'], gamma=opt_params['gamma'], C=opt_params['C'], max_iter=-1)
            svclassifier.fit(x_train, y_train)
            
            outer_acc = svclassifier.score(x_test_outer, y_test_outer)
            
            # Logs inner and outer subject accuracy data in DataFrame
            inner_result_subject.append(inner_acc)
            outer_result_subject.append(outer_acc)

        inner_result.append(inner_result_subject)
        outer_result.append(outer_result_subject)

    return inner_result, outer_result

def permutation(data_params, grid_params, inner_dist, outer_dist, runs=100, output_path='', n_cores=1):
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
        number of runs to perform, default is 100
    """
    
    if inner_dist is None and outer_dist is None:
        inner_dist = []
        outer_dist = []
        
    for _ in tqdm(range(runs)):
        inner_accs, outer_accs = train(data_params, grid_params, scramble=True, n_cores=n_cores)
        inner_dist.append(inner_accs)
        outer_dist.append(outer_accs)

        np.save(f'{output_path}/within_outer_perms.npy', outer_dist)
        np.save(f'{output_path}/within_inner_perms.npy', inner_dist)

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

write_readme(output_path)

data_params = {'path': path, 'roi': roi, 'conds': conds}
grid_params = {'gamma': gamma_range, 'C': C_range, 'kernels': kernels}

if args.permute:
    try:
        outer_dist = np.load(f'{output_path}/within_outer_perms.npy')
        inner_dist = np.load(f'{output_path}/within_inner_perms.npy')
    except FileNotFoundError:
        print('Creating new permutation distribution...')
        inner_dist = None
        outer_dist = None

    permutation(data_params, grid_params, inner_dist, outer_dist, runs=run_count, output_path=output_path, n_cores=num_cores)

    np.save(f'{output_path}/within_outer_perms.npy', outer_dist)
    np.save(f'{output_path}/within_inner_perms.npy', inner_dist)
else:
    inner_result, outer_result = train(data_params, grid_params, n_cores=num_cores)

    np.save(f'{output_path}/within_outer_accs.npy', outer_result)
    np.save(f'{output_path}/within_inner_accs.npy', inner_result)
