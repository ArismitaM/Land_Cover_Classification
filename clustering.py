import rasterio
from rasterio.plot import show
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle as box
import numpy as np
from sklearn.cluster import DBSCAN
import os
from PIL import Image

# Function to convert to YOLO format
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# Open image
img = rasterio.open('../data/archive/label/trialTrain/aachen_1.tif')

# Extract the full filename
full_filename = os.path.basename(img.name)  # img.name gives the file path used to open the image
# Extract the file name without extension
file_name_without_ext = os.path.splitext(full_filename)[0]

# Define folder paths and create directories if necessary
folder_path = "/home/arismita/ML/landCover/data/archive/labels_txt/trailTrain"
os.makedirs(folder_path, exist_ok=True)

# Create file path for the text file
txt_file_name = f"{file_name_without_ext}.txt"
file_path = os.path.join(folder_path, txt_file_name)

# Read image band
img_band = img.read(1)
uniqueVals = np.unique(img_band)
print("Unique Values:", uniqueVals)

# Loop through unique values in the band
for j in uniqueVals:
    band_matrixj = np.copy(img_band)
    band_matrixj[band_matrixj != j] = 0

    band_idxOfj = np.asarray(np.where(img_band == j))
    band_idxOfj_flipped = np.transpose(band_idxOfj)

    db_idxOfj_flipped = DBSCAN(eps=4, min_samples=15).fit(band_idxOfj_flipped)
    labels_idxOfj_flipped = db_idxOfj_flipped.labels_
    unique_labels = set(labels_idxOfj_flipped)

    print(f"Clustering info for pixel value {j}:")

    # Image dimensions for YOLO conversion
    img_width, img_height = img_band.shape[1], img_band.shape[0]

    for k in unique_labels:
        if k == -1:
            continue

        class_member_mask = labels_idxOfj_flipped == k
        xy = band_idxOfj_flipped[class_member_mask]

        if xy.size != 0:
            xmin = np.min(xy[:, 1])
            ymin = np.min(xy[:, 0])
            xmax = np.max(xy[:, 1])
            ymax = np.max(xy[:, 0])

            # Convert to YOLO format
            yolo_box = convert((img_width, img_height), (xmin, xmax, ymin, ymax))

            values_to_add = [j, *yolo_box]

            with open(file_path, 'a') as file:
                for value in values_to_add:
                    file.write(f"{value} ")
                file.write("\n")

print(f"Bounding boxes saved to {file_path}")
