import glob
import tensorflow as tf
from tensorflow import keras
import cv2

BloodCells = [cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/*.jpg')]
class_names = ['Basophil', 'Eusinophil', 'Monocyte', 'Neutrophil']  

model = keras.Sequential([
   keras.layers.Flatten
def conv2d(w, W):
   return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                           size of window          movement of window
    return  tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv1 = conv2d(x, weights['W_conv1'])
conv1 = maxpool2d(conv1)

conv2 = conv2d(conv1, weights['W_conv2'])
conv2 = maxpool2d(conv2)

fc = tf.reshape(conv2), [-1, 7*7*64];
fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

output = tf.matmul(fc, weights['out'])+biases['out']
