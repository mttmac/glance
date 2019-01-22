#!/usr/bin/env python

'''
Define functions to generate training data for CNN from raw sensor data.

Written by Matt MacDonald 2019
'''

import os, io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Constants

PATH = Path('~/github/glance/data/')


# Functions

def save_plots(data):

    for i in data.index:
        sensor = data.sensor[i]
        cycle = data.cycle[i]
        vector = data.data[i]

        path = PATH / f'sample/{sensor.upper()}'
        if not path.is_dir():
            os.mkdir(path)

        plt.figure(num=None, figsize=(2.24, 2.24), dpi=100,
                   facecolor='w', edgecolor='k')
        plt.plot(vector, 'k')
        plt.axis('off')
        plt.savefig(path / f'{cycle}.jpg', format='jpg')
        plt.close()


def make_plot(vector):
    plt.figure(num=None, figsize=(2.24, 2.24), dpi=100,
               facecolor='w', edgecolor='k')
    plt.plot(vector, 'k')
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    pix = np.array(im)
    buf.close()
    plt.close()
    return pix[:, :, :3]

def load_plot(data):
    sensors = data.sensor.unique()
    for i in data.index:
        sensor = data.sensor[i]
        cycle = data.cycle[i]
        vector = data.data[i]

        pix = make_plot(vector)
        X = torch.Tensor(pix).unsqueeze(0)
        y = sensor == sensors
        y = torch.Tensor(y.astype(np.int64))

        yield X, y
