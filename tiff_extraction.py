import numpy as np
import rasterio
from rasterio.plot import show
from matplotlib import pyplot as plt

#img = rasterio.open('/home/arismita/ML/landCover/data/archive/images/train/tokyo_37.tif')
#img = rasterio.open('/home/arismita/ML/landCover/data/archive/label/train/tokyo_37.tif')

lst = []
for i in range (0,11):
    emp = []
    for j in range (0, 11):
        emp.append(255)
    lst.append(emp)

#data = img.read()
#print("The read data: ", data)
show(lst)
'''
# X and Y are supposed to be latitude and longitude if u have the right metadata 

img_band = lst.read() #stands for the 1st band
#print ("The read metadata1: ", img_band1.meta)
show(img_band)
print (img_band)

arr = np.array(img_band)
unique_elements = np.unique(arr) 
print(unique_elements)

'''
