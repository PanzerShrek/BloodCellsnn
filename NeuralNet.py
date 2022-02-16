import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import keras
import numpy as np
import cv2
#based on https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/


Basophil=[cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Basophil/*.jpg')]
Eusinophil=[cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Eusinophil/*.jpg')]
Lymphocyte=[cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Lymphocyte/*.jpg')]
Monocyte=[cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Monocyte/*.jpg')]
Neutrophil=[cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Neutrophil/*.jpg')]
#640 × 480

B=np.array(Basophil);
E=np.array(Eusinophil);
L=np.array(Lymphocyte);
M=np.array(Monocyte);
N=np.array(Neutrophil);


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
             loss = 'sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(B, E, L, M, N) #epoch is full pass through dataset

#class_names = [1: 'Basophil', 2: 'Eusinophil', 3: 'Monocyte', 4: 'Neutrophil', 5: 'Lymphocyte']



#B=[1,0,0,0,0]*np.array(Basophil);
#E=[0,1,0,0,0]*np.array(Eusinophil);
#L=[0,0,1,0,0]*np.array(Lymphocyte);
#M=[0,0,0,1,0]*np.array(Monocyte);
#N=[0,0,0,0,1]*np.array(Neutrophil);



