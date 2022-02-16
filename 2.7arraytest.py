import glob
import cv2
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt

#img = [cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/*.jpg')]
img = [cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/*.jpg')]
img.shape
#print(img)
