#!/usr/bin/env python

'''
Split raw data into test and train folders, saved as numpy arrays
Written by Matt MacDonald 2019
'''

import pandas as pd
import numpy as np

import os
from pathlib import Path
from tqdm import tqdm
import dill as pickle

from pdb import set_trace

PATH = Path.cwd() / 'data/hydraulic/'
# Warning: functions assume the current working directory is PATH, cd into it first

def load_data():
    with open(Path('scratch/data.pkl'), 'rb') as file:
        data = pickle.load(file)

    with open(Path('scratch/faults.pkl'), 'rb') as file:
        faults = pickle.load(file)
    
    return data, faults

        
def prep_folders(key):
    folders = [Path(key + '/train/train/norm/'),
               Path(key + '/train/val/norm/'),
               Path(key + '/test/norm/'),
               Path(key + '/test/fail/')]
    for folder in folders:
        if not folder.is_dir():
            os.makedirs(folder)
        if len(os.listdir(folder)):
            raise AssertionError(f"Directory {folder} is not empty")
    return folders


"""
1: Cooler condition / %:
	3: close to total failure
	20: reduced effifiency
	100: full efficiency

2: Valve condition / %:
	100: optimal switching behavior
	90: small lag
	80: severe lag
	73: close to total failure

3: Internal pump leakage:
	0: no leakage
	1: weak leakage
	2: severe leakage

4: Hydraulic accumulator / bar:
	130: optimal pressure
	115: slightly reduced pressure
	100: severely reduced pressure
	90: close to total failure

5: stable flag:
	0: conditions were stable
	1: static conditions might not have been reached yet
"""


def show_faults(df):
    for col in df.columns:
        print(col)
        for val in sorted(df[col].unique()):
            count = (df[col] == val).sum()
            print(f"{val:5} - {count} samples")

def define_failures(df):
    masks = {}
    
    # Cooler failure
    mask = df.iloc[:, 0] == 3
    masks['cooler'] = mask
    any_mask = mask
    
    # Valve failure
    mask = df.iloc[:, 1] == 73
    masks['valve'] = mask
    any_mask = any_mask | mask
    
    # Pump failure
    mask = df.iloc[:, 2] == 2
    masks['pump'] = mask
    any_mask = any_mask | masks
    
    # Accumulator failure
    mask = df.iloc[:, 3] == 90
    masks['accumulator'] = mask
    any_mask = any_mask | mask
    
    # Any fault
    masks['any'] = any_mask
    
    # Normal (optimal or sub-optimal) otherwise
    # Stability is ignored b/c monitoring must work for   
    # stable and unstable conditions
    return masks


class Log(object):
    def __init__(self, text=""):
        self.text = str(text)
    
    def __repr__(self):
        return self.text
    
    def log(self, text):
        text = str(text)
        if self.text:
            self.text = '\n'.join((self.text, text))
        else:
            self.text = text
    
    def save(self, filename):
        with open(filename, 'w') as file:
            file.write(self.text)


def log_breakdown(failures, log=None):
    if log is None:
        log = Log()
    log.log('Failures:')
    for i, key in enumerate(failures.keys()):
        if not i:
            mask = failures[key]
        else:
            mask = mask | failures[key]
        n = failures[key].sum()
        total = failures[key].count()
        log.log(f"{key}: {n}/{total} - {n / total:.2%}")
    n = mask.sum()
    total = mask.count()
    log.log(f"overall: {n}/{total} - {n / total:.2%}")
    return log
    

def save_test_train_val(data, faults, failures, key, size=512):
    # Split cycle indices
    mask = failures[key]
    test_idx_fail = faults[mask].index.values
    train_idx = faults[~mask].index.values
    n_tests = min(int(0.25 * train_idx.size), len(test_idx_fail)) 
    test_idx_norm = np.random.choice(train_idx,
                                     n_tests,
                                     replace=False)
    train_idx = train_idx[~np.in1d(train_idx, test_idx_norm)]
    val_idx = np.random.choice(train_idx,
                               int(0.2 * train_idx.size),
                               replace=False)
    train_idx = train_idx[~np.in1d(train_idx, val_idx)]
    
    indices = (train_idx, val_idx, test_idx_norm, test_idx_fail)
    assert sum([len(idx) for idx in indices]) == len(faults), "Split error"
    
    # Prepare for saving
    log = log_breakdown(failures)
    log.log('Failure used:')
    log.log(key)
    index_ref = sorted(data.sensor.unique())  # only arrays will be saved
    log.log('Sensors:')
    log.log(index_ref)
    folders = prep_folders(key)
    log.log('Folders')
    log.log([str(folder) for folder in folders])
    log.log('Folder sizes:')
    log.log([len(idx) for idx in indices])
    log.save('reference.txt')
    print('Starting split and save..')
    
    # Resample the time-series to size and save array to folders
    for num, idx in enumerate(tqdm(indices)):
        folder = folders[num]
        for cycle in idx:
            # Select only data for specific cycle
            df = data[data.cycle == cycle]
            df = df.set_index('sensor').sort_index()
            assert index_ref == list(df.index), "Missing sensors"
            
            # Create an array from data
            arr = np.zeros((len(df), size))  # row for each sensor
            for i, s in enumerate(df.index):
                vec = df.data[s]
                delta = 1 / float(re.search(r"([0-9.]+)",
                                            df.sample_rate[s]).group())
                delta = pd.Timedelta(delta, unit='s')
                ser = pd.Series(vec)
                ser.index *= delta  # turn into time series
                
                new_delta = len(vec) * delta / size
                ser = ser.resample(new_delta).mean()
                
                if new_delta < delta:  # upsample
                    ser = ser.interpolate()
                    while len(ser) < size:  # upsampling is usually short
                        end_time = ser.index[-1] + new_delta
                        ser[end_time] = vec[-1]  
                
                assert len(ser) == size, "Resampling size error"
                arr[i, :] = ser.values
            
            # Save array to folder
            filename = f"{cycle}.npy"
            path = folder / filename
            # No pickling or python 2 compatibility
            np.save(path, arr, False, False)
    
    print('Success!')

    
def save_normals(data):
    # Save mean and std array for sensor channels
    mean = []
    std = []
    sensors = sorted(data.sensor.unique())
    for s in sensors:
        vecs = np.stack(data[data.sensor == s].data)
        mu = vecs.mean(axis=1)
        sigma = vecs.std(axis=1)
        mean.append(mu.mean())
        std.append(sigma.max())
    norms = np.array([mean, std]).T
    np.save('normals.npy', norms, False, False)
    

def save_normals_sigmoid(data):
    # Save mean and std array that gaurantee data between 0 and 1
    mean = []
    std = []
    sensors = sorted(data.sensor.unique())
    for s in sensors:
        vecs = np.stack(data[data.sensor == s].data)
        mx = vecs.max(axis=1)
        mn = vecs.min(axis=1)
        mx = mx.max()
        mn = mn.min()
        mu = (mx + mn) / 2
        sigma = np.abs(mx - mn)
        mean.append(mu - 0.5 * sigma)
        std.append(sigma)
    norms = np.array([mean, std]).T
    np.save('normals.npy', norms, False, False)