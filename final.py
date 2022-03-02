# IMPORTS 
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import tensorflow as tf


fs = 44100 #like on cd recordings
seconds = 2
filename = "prediction.wav" #for "Hello Alex"
filename2="prediction2.wav" #for "Call Home"
class_names = ["Wake Word NOT Detected", "Wake Word Detected"]

#Loading the saved models and beginning the prediction process
model =tf.keras.models.load_model("saved_model/WWD.h5")
model2=tf.keras.models.load_model("saved_model/WWD1.h5")

print("Prediction Started: ")
i = 0
#change to while true if you need it to listen continuously
while i<50:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    myrecording2 = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, myrecording)
    write(filename2, fs, myrecording2)


    audio, sample_rate = librosa.load(filename)
    audio2, sample_rate2 = librosa.load(filename2)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sample_rate2, n_mfcc=40)
    mfcc_processed2= np.mean(mfcc2.T, axis=0)



    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    prediction2 = model2.predict(np.expand_dims(mfcc_processed2, axis=0))
    if prediction[:, 1] > 0.9995:
        print(f"Wake Word Detected for ({i})")
        print("Confidence:", prediction[:, 1])
        i += 1
    
    elif prediction2[:, 1] > 0.9995:
        print(f"Wake Word Detected for ({i})")
        print("Confidence:", prediction2[:, 1])
        i += 1

    else:
        print(f"Wake Word NOT Detected")
        print("Confidence:", prediction[:, 0])

    