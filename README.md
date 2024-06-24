# Land Cover Classification

## Analyzing Dataset

I analyzed the [data set](https://www.kaggle.com/datasets/aletbm/global-land-cover-mapping-openearthmap).
The dataset contains .tif images (geospatial satellite images) and the labels are in colour-coded format.
There are 3 directories under in the data - `test`, `train`, and `val` (stands for validation)
- Training set: Used to train the model.
- Validation set: Used to evaluate the model during training and tune hyperparameters.
- Test set: Used to evaluate the model's performance after training is complete

**This is an image present in the dataset under the directory images/train**

![Screenshot from 2024-05-18 19-19-43](https://github.com/abhisheks008/DL-Simplified/assets/146760434/82b95f82-e6b5-4265-89d6-e97c12b849dd)

**Each image has 3 layers (RGB), so below are the layers of the above image:**

Red Channel
![Screenshot from 2024-05-18 19-20-01](https://github.com/abhisheks008/DL-Simplified/assets/146760434/ec853ed6-b71c-4bbe-93ee-58ecce692b45)

Green Channel
![Screenshot from 2024-05-18 19-20-09](https://github.com/abhisheks008/DL-Simplified/assets/146760434/6e521154-e053-455d-9846-b7e324a8ec1e)

Blue Channel
![Screenshot from 2024-05-18 19-20-16](https://github.com/abhisheks008/DL-Simplified/assets/146760434/77b834e0-ea67-4a78-9917-144f9f9f39df)

**There is a corresponding label/train which has 1 layer with coloured label**

Colour(Hex)  | Class|
-------------|----------|
#800000	     |	Bareland |
#00FF24	     |	Rangeland |
#949494	     |	Developed space |
#FFFFFF	     |	Road |
#226126	     |	Tree |
#0045FF	     |	Water |
#4BB549	     |	Agriculture land |
#DE1F07	     |	Building |

![Screenshot from 2024-05-18 19-20-38](https://github.com/abhisheks008/DL-Simplified/assets/146760434/f87a5c3e-2ad7-495d-8472-aee35c3e30e8)

## Clustering (using DBScan)

Label displaying class 2 (Rangeland)
![Screenshot from 2024-05-31 10-43-17](https://github.com/abhisheks008/DL-Simplified/assets/146760434/cd5a808e-760d-465d-877e-a85fe76979e8)

Clusters drawn on the label using DBScan
![Screenshot from 2024-05-31 10-44-34](https://github.com/abhisheks008/DL-Simplified/assets/146760434/096187d0-1fcb-489c-9161-d28ca9d8d6c1)

Drawing bounding boxes around clusters
![Screenshot from 2024-06-01 11-00-27](https://github.com/abhisheks008/DL-Simplified/assets/146760434/a9867f79-698b-48ef-8751-cf118fb138a5)

So, the `clustering.py` file draws clusters for each class and makes bounding boxes around them. The coordinates of the boxes are used to generate labels and these labels are converted into YOLO format for training.
The YOLO format labels are then converted into Pascal VOC format for RetinaNet training.

