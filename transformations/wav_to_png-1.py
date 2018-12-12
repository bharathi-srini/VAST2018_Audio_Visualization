# Import dependencies
import os
import numpy as np
from scipy.signal import spectrogram
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Define values
AXES = False # Whether to include axes in the resulting plots
DIRNAME = os.path.dirname(__file__)
INPUT_DIR = os.path.join(DIRNAME, '../input/birds')
OUTPUT_DIR = os.path.join(DIRNAME, '../data/png-1')

# Make output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Convert files
for file in os.listdir(INPUT_DIR):
    name = file[:-4]

    # Compute spectrum
    sample_rate, samples = wavfile.read(os.path.join(INPUT_DIR, file))
    if (len(samples.shape) > 1):
        samples = np.mean(samples, axis = 1) # Join stereo signals
    F, T, spectrum = spectrogram(samples, sample_rate)

    # Create plot
    plt.pcolormesh(T, F, spectrum, cmap = 'hot')
    plt.savefig(os.path.join(OUTPUT_DIR, name))
    plt.clf()
    print('Converted {}'.format(name))
