# Import dependencies
import os
from pydub import AudioSegment

# Define values
DIRNAME = os.path.dirname(__file__)
INPUT_DIR = os.path.join(DIRNAME, '../input/birds')
OUTPUT_DIR = os.path.join(DIRNAME, '../output/birds')

# Make output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Convert files
for file in os.listdir(INPUT_DIR):
    name = file[:-4]
    audio = AudioSegment.from_mp3(os.path.join(INPUT_DIR, file))
    audio.export(os.path.join(OUTPUT_DIR, name + '.wav'), format = 'wav', bitrate = '44.1kHz')
    print('Converted {}'.format(name))
