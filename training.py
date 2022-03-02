# IMPORTS
import os
os.environ['KERAS_BACKEND']='tensorflow'
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
#from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

##### Loading saved csv ##############
df1= pd.read_pickle("final_audio_data_csv/cfinal.csv")
df = pd.read_pickle("final_audio_data_csv/bup.csv")

####### Making our data train-ready
#sound data
X = df["feature"].values
X = np.concatenate(X, axis=0).reshape(len(X), 40)
X1 = df1["feature"].values
X1 = np.concatenate(X1, axis=0).reshape(len(X1), 40)
#whether our sound contains the wakeword or not
y = np.array(df["class_label"].tolist())
y = tf.keras.utils.to_categorical(y)
y1= np.array(df1["class_label"].tolist())
y1 = tf.keras.utils.to_categorical(y1)

#train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Training -- neural networks
model = Sequential([
    #creating a dense layer of 256 neurons with relu activation function
    Dense(256, input_shape=X_train[0].shape),
    Activation('relu'),
    #dropout layer which will randomly drop 50% of neurons 
    #which will ensure that our model doesnt ovefit
    Dropout(0.5),
    Dense(256),
    Activation('relu'),
    Dropout(0.5),
    #output layer with 2 neurons and softmax activation
    Dense(2, activation='softmax')
])

model1= Sequential([
    #creating a dense layer of 256 neurons with relu activation
    Dense(256, input_shape=X1_train[0].shape),
    Activation('relu'),
    #dropout layer which will randomly drom 50% of neurons 
    #which will ensure that our model doesnt ovefit
    Dropout(0.5),
    Dense(256),
    Activation('relu'),
    Dropout(0.5),
    #output layer with 2 neurons and softmax activation
    Dense(2, activation='softmax')
])

print(model.summary())
print(model1.summary())

##models found online
model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)
model1.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)


print("Model Score: \n")
history = model.fit(X_train, y_train, epochs=1000)
model.save("saved_model/WWD.h5")
score = model.evaluate(X_test, y_test)
print(score)

print("Model Score: \n")
history = model1.fit(X1_train, y1_train, epochs=1000)
model1.save("saved_model/WWD1.h5")
score1 = model1.evaluate(X1_test, y1_test)
print(score1)


# Evaluating our model
print("Model Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
f=sns.heatmap(cm,annot=True)
print(classification_report(np.argmax(y_test, axis=1), y_pred))


# Evaluating our second model
print("Model Classification Report: \n")
y1_pred = np.argmax(model1.predict(X1_test), axis=1)
cm1 = confusion_matrix(np.argmax(y1_test, axis=1), y1_pred)
f1=sns.heatmap(cm1,annot=True)
print(classification_report(np.argmax(y1_test, axis=1), y1_pred))
