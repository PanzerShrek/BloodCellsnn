import matplotlib.image as im
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob
marked = os.listdir("/Users/nickrizzolo/Desktop/Marked")   
marked2 = np.array(marked)

#images = [cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/Marked')]
#    marked3 = mpimg.imread(Screen_Shot_2018-05-21_at_3.13.44_PM.png) 
#    image = mpimg.open(marked3)
#    plt.imshow(marked3)
#    plt.show()
#read image
#img = cv2.imread(os.walk("/Users/nickrizzolo/Desktop/Marked"))
#display image
#for files in os.walk("/Users/nickrizzolo/Desktop/Marked"):
#    plt.imread(files)


img = [cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/Blood\ Cells\ Python/Blood\ Cells/*.jpg')]
print(img)
#img = [cv2.imread(files) for files in glob.glob('/Users/nickrizzolo/Desktop/Marked/*.png')]
#print(img)






