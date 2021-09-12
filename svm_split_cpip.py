import os
import time
import math
import itertools
import scipy.io
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

"""
Simulation settings
"""

OUTPUT_DIR = 'output/cpip/'

path = 'scans/output/PRE/'
roi = 1                            # V1-roi: 0, MT-roi: 1
use_alex = True

n_cores = 8
start_block = 1
n_blocks = 64

def get_subjects(path):
    files = os.listdir(path)

    subjects = [f[:2] for f in files]
    suffix = files[0][2:]
    subjects.sort()

    return subjects, suffix
    
def get_min_max_block_length(path, subjects, suffix, roi):
    min_bl, max_bl = math.inf, 0
    for subject in subjects:
        path_to_file = path + subject + suffix
        mat = scipy.io.loadmat(path_to_file)['roi_scanData'][0][roi]
        for scan in range(len(mat[0])):
            for cond in range(len(mat[0][scan][0])):
                for block in range(len(mat[0][scan][0][cond][0])):
                    block_data = []
                    for tr in range(len(mat[0][scan][0][cond][0][block][0])):
                        block_data.extend(mat[0][scan][0][cond][0][block][0][tr][0][0][0].tolist())
                    
                    min_bl = min(min_bl, len(block_data))
                    max_bl = max(max_bl, len(block_data))
                    
    print(f"Min block length: {min_bl}")
    print(f"Max block length: {max_bl}")

    return min_bl, max_bl

def extract_subject_data(path, subject, suffix, roi, block_length, rank_block, use_abs_to_rank): 
    x_data = []
    y_data = []
    splits = []
    ranked_indices = None
    
    path_to_file = path + subject + suffix
    mat = scipy.io.loadmat(path_to_file)['roi_scanData'][0][roi]
    
    block_count = 0
    for cond_set in [[0, 2], [1, 3]]:
        for scan in range(len(mat[0])):
            for cond in cond_set:
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
        splits.append(len(x_data))
    
    # Run through and rank-order based on subject block  
    for block_n in range(len(x_data)):
        x_data[block_n] = [x_data[block_n][i] if i  < len(x_data[block_n]) else 0.0 for i in ranked_indices]
                
    data = {'x': x_data, 'y': y_data, 'splits': splits}
    return data

def generate_dataset(path, subjects, suffix, roi, block_length, rank_block, use_abs_to_rank):
    x_data = []
    
    x_data_indices = []
    y_data_by_subject = dict()
    
    for subject in subjects:
        subject_data = extract_subject_data(path, subject, suffix, roi, block_length, rank_block, use_abs_to_rank)
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

def split_dataset(x_data, y_data, inner_subjects, outer_subject):
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
            y_train.extend(y_data[subject])
            
    return x_train, y_train, x_test_inner, y_test_inner, x_test_outer, y_test_outer

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
    if use_alex:
        results.sort(key=lambda x: [x[1]['C'], x[1]['gamma'], x[1]['kernel']])
    else:
        results.sort(key=lambda x: [x[1]['kernel'], x[1]['gamma'], x[1]['C']])
    for result in results:
        curr_acc, params = result
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_params = params
    
    return best_params, best_acc

def train(data_params, grid_params, num_inner=1, rank_block=1, use_abs_to_rank=False):
    subjects, suffix = get_subjects(data_params['path'])
    bmin, bmax = get_min_max_block_length(data_params['path'], subjects, suffix, data_params['roi'])
    block_length = bmin
    x_data, y_data = generate_dataset(path, subjects, suffix, data_params['roi'], block_length, rank_block, use_abs_to_rank)
        
    inner_acc_report = []
    outer_acc_report = [[] for _ in range(2)]
    
    for outer_subject_idx in tqdm(range(len(subjects))):
        outer_subject = subjects[outer_subject_idx]
        inner_subjects = [s for s in subjects if s != outer_subject]
        
        # Iterate through inner folds
        for inner_test_subjects in itertools.combinations((inner_subjects), num_inner):
            
            inner_test_subjects = list(inner_test_subjects)
            
            x_train, y_train, x_test_inner, y_test_inner, x_test_outer, y_test_outer = split_dataset(x_data, y_data, inner_test_subjects, outer_subject)
            
            # Gets optimal params for training dataset from grid search
            opt_params, inner_acc = get_optimal_run(x_train, y_train, x_test_inner, y_test_inner, grid_params['kernels'], grid_params['gamma'], grid_params['C']) 
            assert(len(set(y_train)) == 2)
            assert(len(set(y_test_inner)) == 2)
            assert(len(set(y_test_outer)) == 2)
            
            if opt_params is not None:
                # Trains model using optimal params for this set
                svclassifier = SVC(kernel=opt_params['kernel'], gamma=opt_params['gamma'], C=opt_params['C'], max_iter=-1)
                svclassifier.fit(x_train, y_train)
                inner_acc_report.append(inner_acc)

            # Test outer subject using optimal params for inner fold
            svclassifier = SVC(kernel=opt_params['kernel'], gamma=opt_params['gamma'], C=opt_params['C'], max_iter=-1)
            svclassifier.fit(x_train, y_train)
        
            split_start = 0
            data_temp = extract_subject_data(path, outer_subject, suffix, roi, block_length, rank_block, use_abs_to_rank)
            for i in range(2):
                split_end = data_temp['splits'][i]
                x_test, y_test = x_test_outer[split_start:split_end], y_test_outer[split_start:split_end]
                split_start = split_end
            
                outer_acc = svclassifier.score(x_test, y_test)
                outer_acc_report[i].append(outer_acc)
        
    return inner_acc_report, outer_acc_report

gamma_range = {'start': -15, 'stop': 3, 'num': 19, 'base': 2.0}
C_range = {'start': -3, 'stop': 15, 'num': 19, 'base': 2.0}
if not use_alex:
    gamma_range = {'start': -11, 'stop': 3, 'num': 15, 'base': 2.0}
    C_range = {'start': -3, 'stop': 11, 'num': 15, 'base': 2.0}
kernels = ['rbf', 'sigmoid']

data_params = {'path': path, 'roi': roi}
grid_params = {'gamma': gamma_range, 'C': C_range, 'kernels': kernels}
suffix = "alex" if use_alex else "danny"

start = time.time()
inner_accs = []
outer_accs = []
for i in range(start_block, n_blocks+1):
    inner_res, outer_res = train(data_params, grid_params, rank_block=i)

    inner_accs.append(inner_res)
    outer_accs.append(outer_res)
    np.save(f'{OUTPUT_DIR}outer_accs_{suffix}{start_block}.npy', outer_accs)
    np.save(f'{OUTPUT_DIR}inner_accs_{suffix}{start_block}.npy', inner_accs)
  
end = time.time()
print(f'Done in {round((end-start)/60, 2)} minutes.')
