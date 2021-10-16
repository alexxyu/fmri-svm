#!/usr/bin/python3

import os
import time
import math
import random
import argparse
import itertools
import scipy.io
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

'''
Constants
'''

#gamma_range = {'start': -15, 'stop': 3, 'num': 19, 'base': 2.0}
#C_range = {'start': -3, 'stop': 15, 'num': 19, 'base': 2.0}

gamma_range = {'start': -11, 'stop': 3, 'num': 15, 'base': 2.0}
C_range = {'start': -3, 'stop': 11, 'num': 15, 'base': 2.0}
kernels = ['rbf', 'sigmoid']

def df_to_arr(df):
    vals = []
    for _, row in df.iterrows():
        vals.extend(row.tolist())
    return np.array([x for x in vals if str(x) != 'nan'])

def get_subjects(path):
    files = os.listdir(path)
    subjects = [f[:2] for f in files]
    suffix = files[0][2:]
        
    subjects.sort()
    
    return subjects, suffix

def scramble_labels(y_data):
    classes = list(set(y_data))
    classes.sort()
    
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

def get_min_max_block_length(path, subjects, suffix, roi, conds):
    min_bl, max_bl = math.inf, 0
    for subject in subjects:
        
        path_to_file = path + subject + suffix
        mat = scipy.io.loadmat(path_to_file)['roi_scanData'][0][roi]

        for scan in range(len(mat[0])):
            for cond in conds:
                for block in range(len(mat[0][scan][0][cond][0])):
        
                    block_data = []
                    for tr in range(len(mat[0][scan][0][cond][0][block][0])):
                        block_data.extend(mat[0][scan][0][cond][0][block][0][tr][0][0][0].tolist())
                    
                    min_bl = min(min_bl, len(block_data))
                    max_bl = max(max_bl, len(block_data))
                    
    print(f"Min block length: {min_bl}")
    print(f"Max block length: {max_bl}")

    return min_bl, max_bl

def extract_subject_data_pre(path, subject, suffix, roi, conds, block_length, rank_block, use_abs_to_rank):
    x_data = []
    y_data = []
    
    path_to_file = path + subject + suffix
    mat = scipy.io.loadmat(path_to_file)['roi_scanData'][0][roi]
    
    ranked_indices = None
    
    block_count = 0
    for scan in range(len(mat[0])):

        for cond in conds:

            blocks = [x for x in range(len(mat[0][scan][0][cond][0]))]

            for block in blocks:
                block_count += 1
                block_data = []
                for tr in range(len(mat[0][scan][0][cond][0][block][0])):
                    # Extract all voxel data from individual TRs
                    block_data.extend(mat[0][scan][0][cond][0][block][0][tr][0][0][0].tolist())
                
                if block_count == rank_block:
                    if use_abs_to_rank:    
                        ranked_indices = [i for i in (np.array([abs(n) for n in block_data])).argsort(kind='mergesort')[-block_length:]]
                        ranked_indices = np.flip(ranked_indices)
                    else:
                        ranked_indices = [i for i in (-np.array(block_data)).argsort(kind='mergesort')[:block_length]]
    
                x_data.append(block_data)
                y_data.append('untrained' if 'untrained' in mat[0][scan][1][cond][0] else 'trained')
    
    # Run through and rank-order based on subject block  
    for block_n in range(len(x_data)):
        x_data[block_n] = [x_data[block_n][i] if i  < len(x_data[block_n]) else 0.0 for i in ranked_indices]
                
    data = {'x': x_data, 'y': y_data}
    return data

def extract_subject_data_post(path_post, path_ls, subject, suffix_post, suffix_ls, roi, conds, block_length, rank_block, use_abs_to_rank):
    x_data = []
    y_data = []
    
    path_to_file = path_post + subject + suffix_post
    ranked_indices = None
    block_count = 0
    try:
        mat = scipy.io.loadmat(path_to_file)['roi_scanData'][0][roi]    
        for scan in range(len(mat[0])):

            for cond in conds:

                blocks = [x for x in range(len(mat[0][scan][0][cond][0]))]

                for block in blocks:
                    block_count += 1
                    block_data = []
                    for tr in range(len(mat[0][scan][0][cond][0][block][0])):
                        # Extract all voxel data from individual TRs
                        block_data.extend(mat[0][scan][0][cond][0][block][0][tr][0][0][0].tolist())
                    
                    if block_count == rank_block:
                        if use_abs_to_rank:    
                            ranked_indices = [i for i in (np.array([abs(n) for n in block_data])).argsort(kind='mergesort')[-block_length:]]
                            ranked_indices = np.flip(ranked_indices)
                        else:
                            ranked_indices = [i for i in (-np.array(block_data)).argsort(kind='mergesort')[:block_length]]
        
                    x_data.append(block_data)
                    y_data.append('untrained' if 'untrained' in mat[0][scan][1][cond][0] else 'trained')
    except FileNotFoundError:
        pass

    path_to_file = path_ls + subject + suffix_ls
    try:
        mat = scipy.io.loadmat(path_to_file)['roi_scanData'][0][roi]
        for scan in range(len(mat[0])):

            for cond in range(4):

                blocks = [x for x in range(len(mat[0][scan][0][cond][0]))]

                for block in blocks:
                    block_count += 1
                    block_data = []
                    for tr in range(len(mat[0][scan][0][cond][0][block][0])):
                        # Extract all voxel data from individual TRs
                        block_data.extend(mat[0][scan][0][cond][0][block][0][tr][0][0][0].tolist())
                    
                    if block_count == rank_block:
                        if use_abs_to_rank:    
                            ranked_indices = [i for i in (np.array([abs(n) for n in block_data])).argsort(kind='mergesort')[-block_length:]]
                            ranked_indices = np.flip(ranked_indices)
                        else:
                            ranked_indices = [i for i in (-np.array(block_data)).argsort(kind='mergesort')[:block_length]]
        
                    x_data.append(block_data)
                    y_data.append('untrained' if 'untrained' in mat[0][scan][1][cond][0] else 'trained')
    except FileNotFoundError:
        pass
        
    # Run through and rank-order based on subject block  
    for block_n in range(len(x_data)):
        x_data[block_n] = [x_data[block_n][i] if i  < len(x_data[block_n]) else 0.0 for i in ranked_indices]
                
    data = {'x': x_data, 'y': y_data}
    return data

def generate_dataset(subjects, paths, suffixes, roi, conds, block_length, rank_block, use_abs_to_rank):
    x_data = []
    
    x_data_indices_pre = []
    y_data_by_subject_pre = dict()
    for subject in subjects:
        subject_data = extract_subject_data_pre(paths['pre'], subject, suffixes['pre'], roi, conds, block_length, rank_block, use_abs_to_rank)
        
        x_data.extend(subject_data['x'])
        x_data_indices_pre.append(len(x_data))
        y_data_by_subject_pre[subject] = subject_data['y']
    
    x_data_indices_post = []
    y_data_by_subject_post = dict()
    for subject in subjects:
        subject_data = extract_subject_data_post(paths['post'], paths['ls'], subject, suffixes['post'], suffixes['ls'], roi, conds, block_length, rank_block, use_abs_to_rank)
        
        x_data.extend(subject_data['x'])
        x_data_indices_post.append(len(x_data))
        y_data_by_subject_post[subject] = subject_data['y']

    # MinMaxScaler scales each feature to values between 0 and 1 among all x data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_standardized = scaler.fit_transform(x_data)
    
    # Sorts block data into respective subject
    x_data_by_subject_pre = dict()
    start_idx = 0
    for i, end_idx in enumerate(x_data_indices_pre):
        subject = subjects[i]
        x_data_by_subject_pre[subject] = x_standardized[start_idx:end_idx]
        start_idx = end_idx
    
    x_data_by_subject_post = dict()
    for i, end_idx in enumerate(x_data_indices_post):
        subject = subjects[i]
        x_data_by_subject_post[subject] = x_standardized[start_idx:end_idx]
        start_idx = end_idx

    x_data_by_subject_pre = {k: v for k, v in sorted(x_data_by_subject_pre.items(), key=lambda item: len(item[1]))}
    y_data_by_subject_pre = {k: v for k, v in sorted(y_data_by_subject_pre.items(), key=lambda item: len(item[1]))}

    x_data_by_subject_post = {k: v for k, v in sorted(x_data_by_subject_post.items(), key=lambda item: len(item[1]))}
    y_data_by_subject_post = {k: v for k, v in sorted(y_data_by_subject_post.items(), key=lambda item: len(item[1]))}

    return x_data_by_subject_pre, y_data_by_subject_pre, x_data_by_subject_post, y_data_by_subject_post

def split_dataset(x_data, y_data, inner_subjects, outer_subject, scramble):
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

def get_optimal_run(x_train, y_train, x_test, y_test, kernels, gamma_range, C_range):
    gamma_vals = np.logspace(gamma_range['start'], gamma_range['stop'], gamma_range['num'], base=gamma_range['base'])
    C_vals = np.logspace(C_range['start'], C_range['stop'], C_range['num'], base=C_range['base'])

    param_grid = ParameterGrid({'kernel': kernels, 'gamma': gamma_vals, 'C': C_vals})
    
    best_acc = 0
    best_params = None
    
    # Tests each parameter combination to find best one for given testing data
    for params in list(param_grid):
        
        svclassifier = SVC(kernel=params['kernel'], gamma=params['gamma'], C=params['C'], max_iter=-1)
        svclassifier.fit(x_train, y_train)
        
        curr_acc = svclassifier.score(x_test, y_test)
        
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_params = params
            
    return best_params, best_acc

def train(data_params, grid_params, num_inner=1, scramble=False, rank_block=1, use_abs_to_rank=False):
    start_time = time.time()
    
    _, suffix_pre = get_subjects(data_params['pre_path'])
    subjects_post, suffix_post = get_subjects(data_params['post_path'])
    subjects_ls, suffix_ls = get_subjects(data_params['ls_path'])

    subjects = list(set(subjects_post).union(set(subjects_ls)))
    
    bmin_pre, _ = get_min_max_block_length(data_params['pre_path'], subjects, suffix_pre, data_params['roi'], data_params['conds'])
    bmin_post, _ = get_min_max_block_length(data_params['post_path'], subjects_post, suffix_post, data_params['roi'], data_params['conds'])
    bmin_ls, _ = get_min_max_block_length(data_params['ls_path'], subjects_ls, suffix_ls, data_params['roi'], data_params['conds'])
    block_length = min(bmin_pre, bmin_post, bmin_ls)
   
    paths = {'pre': data_params['pre_path'], 'post': data_params['post_path'], 'ls': data_params['ls_path']}
    suffixes = {'pre': suffix_pre, 'post': suffix_post, 'ls': suffix_ls}
    x_data_pre, y_data_pre, x_data_post, y_data_post = generate_dataset(subjects, paths, suffixes, data_params['roi'], data_params['conds'], block_length, rank_block, use_abs_to_rank)
        
    inner_acc_report = []
    outer_acc_report = []
    
    for outer_subject in subjects:
        
        print(f"Currently on outer subject #{subjects.index(outer_subject)+1}.")

        opt_inner_params = None
        opt_inner_acc = -1
        
        inner_subjects = [s for s in subjects if s != outer_subject]
        
        # Iterate through inner folds
        for inner_test_subjects in itertools.combinations((inner_subjects), num_inner):
   
            inner_test_subjects = list(inner_test_subjects)
            
            x_train_pre, y_train_pre, x_test_inner_pre, y_test_inner_pre, a, _ = split_dataset(x_data_pre, y_data_pre, inner_test_subjects, outer_subject, scramble)
            x_train_post, y_train_post, x_test_inner_post, y_test_inner_post, b, _ = split_dataset(x_data_post, y_data_post, inner_test_subjects, outer_subject, scramble)

            #print(f'PRE: {len(x_train_pre)}\t{len(x_test_inner_pre)}\t{len(a)}')
            #print(f'POST: {len(x_train_post)}\t{len(x_test_inner_post)}\t{len(b)}')

            x_train = x_train_pre + x_train_post
            y_train = y_train_pre + y_train_post
            x_test_inner = x_test_inner_pre + x_test_inner_post
            y_test_inner = y_test_inner_pre + y_test_inner_post

            # Gets optimal params for training dataset from grid search
            opt_params, inner_acc = get_optimal_run(x_train, y_train, x_test_inner, y_test_inner, grid_params['kernels'], grid_params['gamma'], grid_params['C']) 
            assert(len(set(y_train)) == 2)
            assert(len(set(y_test_inner)) == 2)
            
            if opt_params is not None:
                # Trains model using optimal params for this set
                svclassifier = SVC(kernel=opt_params['kernel'], gamma=opt_params['gamma'], C=opt_params['C'], max_iter=-1)
                svclassifier.fit(x_train, y_train)
                
                # Track optimal params among all inner folds
                if inner_acc > opt_inner_acc:
                    opt_inner_acc = inner_acc
                    opt_inner_params = opt_params
                    
                inner_acc_report.append(inner_acc)

        # Test outer subject using optimal params across all inner folds
        x_train_pre, y_train_pre, _, _, x_test_outer_pre, y_test_outer_pre = split_dataset(x_data_pre, y_data_pre, [], outer_subject, scramble)
        x_train_post, y_train_post, _, _, x_test_outer_post, y_test_outer_post = split_dataset(x_data_post, y_data_post, [], outer_subject, scramble)

        x_train = x_train_pre + x_train_post
        y_train = y_train_pre + y_train_post

        svclassifier = SVC(kernel=opt_inner_params['kernel'], gamma=opt_inner_params['gamma'], C=opt_inner_params['C'], max_iter=-1)
        svclassifier.fit(x_train, y_train)
        
        outer_acc_pre = svclassifier.score(x_test_outer_pre, y_test_outer_pre)
        outer_acc_post = svclassifier.score(x_test_outer_post, y_test_outer_post)
        outer_acc_report.append([outer_acc_pre, outer_acc_post])
        
    # Prints how long it took for last outer subject test
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Completed in {round(exec_time/60, 2)} minutes.")
    
    return inner_acc_report, outer_acc_report

def permutation(data_params, grid_params, inner_dist, outer_dist, runs=30, n_cores=1, num_rank_blocks=1, output_path=''):
    
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
        number of runs to perform, default is 30
    n_cores: int
        number of CPU cores for parallelization, default is 1
    num_rank_blocks: int
        the number of blocks to rank order by, from #1 to #(num_rank_blocks)
        default is 1
    output_path: str
        path to which files should be saved,
        default is current directory
    '''
        
    for n in range(runs):

        print(f'On run {n+1} of {runs}.')
        
        start = time.time()
        result = Parallel(n_jobs=n_cores)(delayed(train)(data_params, grid_params, scramble=True, rank_block=i) for i in range(1, num_rank_blocks+1))
        end = time.time()
        print(f'Done in {round((end-start)/60, 2)} minutes.')

        for rank_result in result:
            inner_result = rank_result[0]
            outer_result = rank_result[1]
            
            inner_dist.append(inner_result)
            outer_dist.append(outer_result)

        np.save(f'{output_path}/outer_perms.npy', outer_dist)
        np.save(f'{output_path}/inner_perms.npy', inner_dist)

parser = argparse.ArgumentParser()
parser.add_argument("outdir", metavar="OUTDIR", help="directory to output data")
parser.add_argument("-p", "--permute", action="store_true", help="permute training data")

parser.add_argument("-n", "--run-count", action="store", default=1, type=int, help="run RUN_COUNT permutations")
parser.add_argument("-b", "--block-count", action="store", default=1, type=int, help="rank-order BLOCK_COUNT number of blocks")

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
block_count = args.block_count
run_count = args.run_count
num_cores = args.cores

output_path = args.outdir
if output_path[-1] != '/':
    output_path += '/'

data_params = {'pre_path': 'scans/output/PRE/', 'post_path': 'scans/output/cp&ip/', 'ls_path': 'scans/output/large&small/', 'roi': roi, 'conds': conds}
grid_params = {'gamma': gamma_range, 'C': C_range, 'kernels': kernels}

if args.permute:
    try:
        inner_dist = np.load(f'{output_path}/inner_perms.npy').tolist()
        outer_dist = np.load(f'{output_path}/outer_perms.npy').tolist()
    except FileNotFoundError:
        print('Creating new permutation distribution...')
        inner_dist = []
        outer_dist = []

    permutation(data_params, grid_params, inner_dist, outer_dist, runs=run_count, output_path=output_path, num_rank_blocks=block_count, n_cores=num_cores)

    np.save(f'{output_path}/outer_perms.npy', outer_dist)
    np.save(f'{output_path}/inner_perms.npy', inner_dist)

else:
    start = time.time()
    result = Parallel(n_jobs=num_cores)(delayed(train)(data_params, grid_params, rank_block=i) for i in range(1, block_count+1))
    end = time.time()
    print(f'Done in {round((end-start)/60, 2)} minutes.')

    inner_accs = []
    outer_accs = []
    for rank_result in result:
        inner_result = rank_result[0]
        outer_result = rank_result[1]
        
        inner_accs.append(inner_result)
        outer_accs.append(outer_result)

    np.save(f'{output_path}/outer_accs.npy', outer_accs)
    np.save(f'{output_path}/inner_accs.npy', inner_accs)
