# Import dependencies
import os
import numpy as np
from scipy.signal import spectrogram
from scipy.io import wavfile
import matplotlib.pyplot as plt
import colorsys

# Define values
DIRNAME = os.path.dirname(__file__)
INPUT_DIR = os.path.join(DIRNAME, '../input/birds')
OUTPUT_DIR = os.path.join(DIRNAME, '../data/png-2')

# Define colors
colors = {
    'Ben': '#640000',
    'Blu': '#642000',
    'Bom': '#643F00',
    'Bro': '#645F00',
    'Can': '#4A6400',
    'Car': '#2A6400',
    'Dar': '#0B6400',
    'Eas': '#006415',
    'Gre': '#006435',
    'Les': '#006454',
    'Ora': '#005464',
    'Ord': '#003564',
    'Pin': '#001564',
    'Pur': '#0B0064',
    'Qax': '#2A0064',
    'Que': '#4A0064',
    'Ros': '#64005F',
    'Scr': '#64003F',
    'Ver': '#640020'
}

# Make output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Convert files
for file in os.listdir(INPUT_DIR):
    name = file[:-4]
    sample_rate, samples = wavfile.read(os.path.join(INPUT_DIR, file))
    if (len(samples.shape) > 1):
        samples = np.mean(samples, axis = 1) # Join stereo signals
    plt.xlim((0,200000))
    plt.ylim((-40000,40000))
    plt.axis('off')
    plt.rcParams['figure.facecolor'] = 'black'
    plt.plot(samples, color = colors[name[0:3]])
    plt.savefig(os.path.join(OUTPUT_DIR, name), bbox_inches = 'tight', pad_inches = 0, facecolor = 'white', edgecolor = 'black', transparent = True)
    plt.clf()
    print('Converted {}'.format(name))
