#!/usr/bin/env python

'''
Define functions to generate training data for CNN from raw sensor data.

Written by Matt MacDonald 2019
'''

import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def save_plots(data):

    for i in data.index:
        sensor = data.sensor[i]
        cycle = data.cycle[i]
        vector = data.data[i]

        plt.figure(num=None, figsize=(2.24, 2.24), dpi=100,
                   facecolor='w', edgecolor='k')
        plt.plot(vector, 'k')
        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        im.show()
    # TODO finish


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
    return pix

def load_plot(data):
    for i in data.index:
        sensor = data.sensor[i]
        cycle = data.cycle[i]
        vector = data.data[i]

        pix = make_plot(vector)
        X = Tensor(pix)
        y = sensor

        yield X, y


# buf.close()

