
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
from pydub import AudioSegment
import matplotlib.pyplot as plt 
import os
import AudioFeatures
import glob
import pickle
import re


# In[2]:


working_dir = '/Users/BharathiSrinivasan/Documents/DIMA_work/Visual_Analytics/VASt2018/2018_Mini-Challenge1/'
input_dir = os.path.join(working_dir, 'input')
output_dir = os.path.join(working_dir, 'output')


# In[7]:


for file in os.listdir(input_dir):
    name = file[:-4]
    audio = AudioSegment.from_mp3(os.path.join(input_dir, file))
    audio.export(os.path.join(output_dir, name + '.wav'), format = 'wav', bitrate = '44.1kHz')
    print('Converted {}'.format(name))


# In[3]:


def readAudioFile(path):
    '''
    This function returns a numpy array that stores the audio samples of a specified WAV of AIFF file
    '''
    extension = os.path.splitext(path)[1]

    try:
        #if extension.lower() == '.wav':
            #[Fs, x] = wavfile.read(path)
        if extension.lower() == '.aif' or extension.lower() == '.aiff':
            s = aifc.open(path, 'r')
            nframes = s.getnframes()
            strsig = s.readframes(nframes)
            x = numpy.fromstring(strsig, numpy.short).byteswap()
            Fs = s.getframerate()
        elif extension.lower() == '.mp3' or extension.lower() == '.wav' or extension.lower() == '.au':            
            try:
                audiofile = AudioSegment.from_file(path)
            #except pydub.exceptions.CouldntDecodeError:
            except:
                print("Error: file not found or other I/O error. (DECODING FAILED)")
                return (-1,-1)                

            if audiofile.sample_width==2:                
                data = numpy.fromstring(audiofile._data, numpy.int16)
            elif audiofile.sample_width==4:
                data = numpy.fromstring(audiofile._data, numpy.int32)
            else:
                return (-1, -1)
            Fs = audiofile.frame_rate
            x = []
            for chn in range(audiofile.channels):
                x.append(data[chn::audiofile.channels])
            x = numpy.array(x).T
        else:
            print("Error in readAudioFile(): Unknown file type!")
            return (-1,-1)
    except IOError: 
        print("Error: file not found or other I/O error.")
        return (-1,-1)

    if x.ndim==2:
        if x.shape[1]==1:
            x = x.flatten()

    return (Fs, x)


# In[66]:


allframes = []
os.chdir(output_dir)
for file in glob.glob("*.wav"):
    name = file[:-4]
    print(file)
    [Fs, x] = readAudioFile(os.path.join(output_dir, file))
    try:
        F = AudioFeatures.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
    except ValueError:
        F = AudioFeatures.stFeatureExtraction(x[:,0], Fs, 0.050*Fs, 0.025*Fs)                                      
    allframes += (F, name)


# In[69]:


print(" Number of wav files processed:")
print(len(allframes)/2)


# In[4]:


#Save converted list 
with open("features_birds.txt", "wb") as fp:   #Pickling
    pickle.dump(allframes, fp)


# In[4]:


# Import features of birds as a list
all_birds = [] 
target = "/Users/BharathiSrinivasan/Documents/DIMA_work/Visual_Analytics/VASt2018/2018_Mini-Challenge1/output/features_birds.txt"
if os.path.getsize(target) > 0:      
    with open(target, "rb") as f:
        unpickler = pickle.Unpickler(f)
        all_birds = unpickler.load()


# In[5]:


#Extract ID and average of each feature over all time frames
list_birds = np.array(all_birds)
birds_df = np.empty((0,34))
bird_id = np.array([0])
count = 0

for bird in list_birds:
    result = np.array([0])
    if count%2 == 0:
        for i in range(0,33):
            result = np.append(result,np.average(bird[i]))
        birds_df = np.append(birds_df,[result],axis =0)
    else:
        bird_id_result = re.findall(r'\d+', bird)
        bird_id = np.append(bird_id,bird_id_result)
    count += 1


# In[31]:


#Creating dataframe with features
bird_id = pd.DataFrame(bird_id,columns=['ID'])


# In[32]:


features = pd.DataFrame(birds_df, columns=['ZCR','Energy','Entropy_of_Energy','Sp_Centroid','Sp_Spread','Sp_Entropy','Sp_Flux','Sp_Rolloff','MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7','MFCC8','MFCC9','M10','M11','M12','M13','CromaVector1','CromaVector2','CromaVector3','CromaVector4','CromaVector5','CromaVector6','CromaVector7','CromaVector8','CromaVector9','CromaVector10','Croma_Vector11','Croma_vector12','Croma_Deviation'])


# In[33]:


#Dataframe with bird IDs and corresponding feature avergages over time - to use for further ML steps
bird_features = pd.concat([bird_id,features],axis=1,join_axes=[bird_id.index])

