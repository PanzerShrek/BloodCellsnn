import cv2
import glob
Basophil=[cv2.IMREAD_GRAYSCALE(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Basophil/*.jpg')]
Eusinophil=[cv2.IMREAD_GRAYSCALE(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Eusinophil/*.jpg')]
Lymphocyte=[cv2.IMREAD_GRAYSCALE(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Lymphocyte/*.jpg')]
Monocyte=[cv2.IMREAD_GRAYSCALE(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Monocyte/*.jpg')]
Neutrophil=[cv2.IMREAD_GRAYSCALE(files) for files in glob.glob('/Users/nickrizzolo/Desktop/BloodCellsPython/BloodCells/Neutrophil/*.jpg')]

print(Basophil)
