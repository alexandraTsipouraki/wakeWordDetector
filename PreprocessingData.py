# IMPORTS 
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Loading the voice data for visualization
ws = "bup/0_127.wav"
data, sample_rate = librosa.load(ws)

# VISUALIZING WAVE FORM 
plt.title("Wake Word Wave Form")
librosa.display.waveshow(data, sr=sample_rate)
plt.show()


##### Doing this for every sample ##

all_data = []

data_path_dict = {
    0: ["background_sound/" + file_path for file_path in os.listdir("background_sound/")],
    1: ["cfinal/" + file_path for file_path in os.listdir("cfinal/")]
    #if you want to use "Hello Alex",uncomment line 34 and comment line 32
    #1: ["bup/" + file_path for file_path in os.listdir("bup/")]
}

# the background_sound/ directory has all the sounds that DO NOT CONTAIN the wake word
# the audio_data/ (or /bup or /cfinal) directories have all sounds containing the wake word

#MFCC : (found from the web)
#Mel-frequency cepstral coefficients (MFCCs) Warning.
#  If multi-channel audio input y is provided, the MFCC calculation will depend 
# on the peak loudness (in decibels) across all channels. The result may differ 
# from independent MFCC calculation of each channel. 
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        audio, sample_rate = librosa.load(single_file) ## Loading file
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40) ## Apllying mfcc
        mfcc_processed = np.mean(mfcc.T, axis=0) ## some pre-processing
        all_data.append([mfcc_processed, class_label])
    print(f"Info: Succesfully Preprocessed Class Label {class_label}")

df = pd.DataFrame(all_data, columns=["feature", "class_label"])

#Saving for future use
df.to_pickle("final_audio_data_csv/cfinal.csv")