# Land Cover Classification

## Analyzing Dataset

I analyzed the [data set](https://www.kaggle.com/datasets/aletbm/global-land-cover-mapping-openearthmap).
There are 3 directories under in the data - `test`, `train` and `val` (stands for validation)
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





