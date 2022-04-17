#!/usr/bin/python3

import os
import sys
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

USE_AVG = False
BLOCK_LIM = 8

CP_PATH = 'scans/output/PRE/'

gamma_range = {'start': -15, 'stop': 3, 'num': 7, 'base': 2.0}
C_range = {'start': -3, 'stop': 15, 'num': 7, 'base': 2.0}
kernels = ['sigmoid']

def write_readme(path):
    with open(f'{path}/README', 'w') as f:
        f.write('Command line arguments:\n')
        f.write(f'{" ".join(sys.argv)}\n\n')
        
        f.write('Hyperparameter set:\n')
        f.write(f'Gamma: {gamma_range}\n')
        f.write(f'C: {C_range}\n')
        f.write(f'Kernels: {kernels}\n')

def get_subjects(path):    
    mat = scipy.io.loadmat(path)['ipOnly_roi_scanData']
    
    subjects = list(set([s[0][:2] for s in mat[1]]))
    subjects.sort()
    return subjects

def get_subjects_cp(path):    
    files = os.listdir(path)
    subjects = sorted([f[:2] for f in files])
    suffix = files[0][2:]
    return subjects, suffix

def scramble_labels(y_data):
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
    if abs(len(y_data)//2 - num_diff) > 1:
        raise ValueError

def extract_subject_data(mat, subject, roi, use_pre):
    subject_runs = []
    for i in range(len(mat[1])):
        if mat[1][i][0][:2] == subject:
            subject_runs.append(i)
    
    assert(len(subject_runs) == 2)

    block_count = 0
    x_data = [[] for _ in range(2)]
    for scan in mat[0][subject_runs[use_pre]][0][roi][0]:
        for c, cond in enumerate(scan[0]):
            for block in cond[0]:
                block_count += 1
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

                if use_pre:
                    x_data[c].append(block_data)
                else:
                    x_data[c // 2].append(block_data)
                # y_data.append('untrained' if 'untrained' in scan[1][cond][0] else 'trained')

    x_data_f, y_data_f = [], []

    # Flatten and transform data into units by voxel
    for c in range(2):
        trs = []
        for a in x_data[c]:
            trs.extend(a)
        trs = np.array(trs).T

        lim = []
        for tr in trs:
            lim.append(tr[:32])

        x_data_f.extend(lim)
        y_data_f.extend([c for _ in lim])

    data = {'x': x_data_f, 'y': y_data_f}
    return data

def extract_subject_data_cp(path, subject, suffix, roi, conds):    
    path_to_file = path + subject + suffix
    mat = scipy.io.loadmat(path_to_file)['roi_scanData'][0][roi]
    
    block_count = 0
    x_data = [[] for _ in conds]
    for scan in mat[0]:
        for c, cond in enumerate(conds):
            for block in scan[0][cond][0]:
                block_count += 1
                if not USE_AVG and block_count > BLOCK_LIM:
                    continue

                block_data = []
                if len(block[0]) == 8:
                    trs = block[0]
                elif len(block[0]) == 9:
                    trs = block[0][:8]
                else:
                    trs = block[0][1:9]
                for tr in trs:
                    # Extract all voxel data from individual TRs
                    block_data.append(tr[0][0][0].tolist())

                x_data[c].append(block_data)

    if USE_AVG and BLOCK_LIM == 8:
        # Standardize to 8 blocks of 8 TRs
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
    if USE_AVG and BLOCK_LIM == 12:
        # Standardize to 12 blocks of 8 TRs
        if block_count == 12:
            x_data = x_data
        elif block_count == 16:
            assert(len(x_data[0]) == 8 and len(x_data[1]) == 8)
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

def generate_dataset(mat, subjects, roi, use_pre):
    x_data = []
    x_data_indices = []
    y_data_by_subject = dict()
    
    for subject in subjects:
        subject_data = extract_subject_data(mat, subject, roi, use_pre)
        x_data_indices.append(len(x_data))
        y_data_by_subject[subject] = subject_data['y']
        
        x_data.extend(subject_data['x'])

    cp_subjects, suffix = get_subjects_cp(CP_PATH)
    for subject in cp_subjects:
        subject_data = extract_subject_data_cp(CP_PATH, subject, suffix, roi, [1, 3])
        x_data_indices.append(len(x_data))
        y_data_by_subject[subject] = subject_data['y']
        
        x_data.extend(subject_data['x'])
    
    # MinMaxScaler scales each feature to values between 0 and 1 among all x data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_standardized = scaler.fit_transform(x_data)
    
    # Sorts block data into respective subject
    x_data_by_subject = dict()
    subjects = subjects + cp_subjects
    for i in range(len(subjects)):
        subject = subjects[i]
        start_index = x_data_indices[i]
        end_index = x_data_indices[i+1] if i+1 < len(x_data_indices) else len(x_data)
        
        x_data_by_subject[subject] = x_standardized[start_index:end_index]

    x_data_by_subject = {k: v for k, v in sorted(x_data_by_subject.items(), key=lambda item: len(item[1]))}
    y_data_by_subject = {k: v for k, v in sorted(y_data_by_subject.items(), key=lambda item: len(item[1]))}    

    y_data_by_subject_shuffled = dict()
    for k, v in y_data_by_subject.items():
        scrambled = v.copy()
        scramble_labels(scrambled)
        y_data_by_subject_shuffled[k] = scrambled

    return x_data_by_subject, y_data_by_subject, y_data_by_subject_shuffled

def split_dataset(x_data, y_data, y_data_shuffled, inner_subjects, outer_subject, scramble):
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
                y_train.extend(y_data_shuffled[subject])
            else:
                y_train.extend(y_data[subject])
            
    return x_train, y_train, x_test_inner, y_test_inner, x_test_outer, y_test_outer

def get_optimal_run(x_train, y_train, x_test, y_test, kernels, gamma_range, C_range, n_cores):
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
    subjects = get_subjects(data_params['path'])
    cp_subjects, _ = get_subjects_cp(CP_PATH)
    if data_params['cond'] == 0:
       subjects.remove('CG')

    mat = scipy.io.loadmat(data_params['path'])['ipOnly_roi_scanData']
    x_data, y_data, y_data_shuffled = generate_dataset(mat, subjects, data_params['roi'], data_params['cond'])
    
    inner_acc_report = []
    outer_acc_report = []
    subjects = subjects + cp_subjects
    for outer_subject in tqdm(subjects, leave=(not scramble)):
        opt_inner_params = None
        opt_inner_acc = -1

        inner_subjects = [s for s in subjects if s != outer_subject]
        
        # Iterate through inner folds
        for inner_test_subjects in itertools.combinations((inner_subjects), num_inner):
            inner_test_subjects = list(inner_test_subjects)
            
            x_train, y_train, x_test_inner, y_test_inner, x_test_outer, y_test_outer = split_dataset(x_data, y_data, y_data_shuffled, inner_test_subjects, outer_subject, scramble)

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
        x_train, y_train, _, _, x_test_outer, y_test_outer = split_dataset(x_data, y_data, y_data_shuffled, [], outer_subject, scramble)
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
parser.add_argument("-c", "--cond", action="store", type=int, help="conditions: 0 is post-training, 1 is pre-training")

parser.add_argument("--cores", action="store", default=1, type=int, help="use CORES number of CPU cores for parallelization")

args, extra = parser.parse_known_args()

if args.roi is None:
    print('Need to specify ROI.')
    exit()
if args.cond is None:
    print('Need to specify cond.')
    exit()

roi = args.roi
cond = args.cond
run_count = args.run_count
num_cores = args.cores

output_path = args.outdir
if output_path[-1] != '/':
    output_path += '/'

path = args.indir

data_params = {'path': path, 'roi': roi, 'cond': cond}
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
