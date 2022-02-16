import glob
import tensorflow as tf
from tensorflow.keras import layers
import cv2
#based on https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/


Basophil=[cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Basophil/*.jpg')]
Eusinophil=[cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Eusinophil/*.jpg')]
Lymphocyte=[cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Lymphocyte/*.jpg')]
Monocyte=[cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Monocyte/*.jpg')]
Neutrophil=[cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Neutrophil/*.jpg')]

#class_names = [1: 'Basophil', 2: 'Eusinophil', 3: 'Monocyte', 4: 'Neutrophil', 5: 'Lymphocyte']

B=[1,0,0,0,0]*Basophil;
E=[0,1,0,0,0]*Eusinophil;
L=[0,0,1,0,0]*Lympohcyte;
M=[0,0,0,1,0]*Monocyte;
N=[0,0,0,0,1]*Neutrophil;

Input = [B,E,L,M,N]
Hidden = tf.nn.relu(Input, name ='ReLU')

__init__(
    input_shape=(640, 480, 3),
    batch_size=None,
    dtype='int32',
    sparse=(False),
    name=none))

# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')


# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

# orginal code
def conv2d:
   return tf.keras.layers.Conv2D(BloodCells, r, strides=[1,1,1,1], padding='SAME')

def maxpool2d:
    #                           size of window          movement of window
    return  tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv1 = conv2d(x, weights['W_conv1'])
conv1 = maxpool2d(conv1)

conv2 = conv2d(conv1, weights['W_conv2'])
conv2 = maxpool2d(conv2)

fc = tf.reshape(conv2), [-1, 7*7*64];
fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

output = tf.matmul(fc, weights['out'])+biases['out']
