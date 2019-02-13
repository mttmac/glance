'''
Define functions to generate training data from raw sensor data.
'''


import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns

import os, io
import platform
from pathlib import Path
import glob
import re
import dill as pickle

from pdb import set_trace


PATH = Path.cwd() / 'data/hydraulic/raw/'
# Warning: functions assume the current working directory is PATH, cd into it first


def save_scratch(obj, name):
    path = Path('scratch/')
    if not path.is_dir():
        os.mkdir(path)

    with open(path / (name + '.pkl'), 'wb') as file:
        pickle.dump(obj, file)


def read_raw(subset=None, need_faults=False):
    with open('description.txt', 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()

    sensors = ''.join(lines[25:40])
    fault_desc = ''.join(lines[46:71])
    fault_lines = [line for line in fault_desc.split('\n') if len(line)]
    fault_names = [line for line in fault_lines if line[0] != '\t']

    sensors = pd.read_csv(io.StringIO(sensors),
                          sep=r'\t+',
                          engine='python')
    names = ['cycle', 'sensor', 'data', 'description', 'unit', 'sample_rate']
    data = pd.DataFrame(columns=names)
    for num, sensor in sensors['Sensor'].iteritems():
        print('Reading dataset for '
              f'{sensor} ({num + 1}/{sensors.shape[0]})...')
        desc = sensors.loc[num, 'Physical quantity']
        unit = sensors.loc[num, 'Unit']
        rate = sensors.loc[num, 'Sampling rate']
        df = pd.read_csv(f'{sensor}.txt',
                         delim_whitespace=True,
                         header=None)
        if subset is not None:
            cycles = np.random.choice(df.index, subset)
        else:
            cycles = df.index
        for i in cycles:
            data_dict = {'cycle': i,
                         'sensor': sensor,
                         'data': [df.iloc[i, :].values],
                         'description': desc,
                         'unit': unit,
                         'sample_rate': rate}
            data = data.append(pd.DataFrame(data_dict), ignore_index=True)

    save_scratch(data, 'data')
    ret = data

    if need_faults:
        print('Reading fault targets...')
        faults = pd.read_csv('profile.txt',
                             delim_whitespace=True,
                             header=None,
                             names=fault_names)
        print('Fault legend:')
        print(fault_desc)

        save_scratch(faults, 'faults')
        ret = (ret, faults)

    return ret


def unique_faults(faults):
    u_faults = []
    for col in faults.columns:
        types = list(faults[col].unique())
        u_faults += zip([col] * len(types), types)
    return u_faults
