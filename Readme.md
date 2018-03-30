# Bauta
Bauta is an easy-to-use toolset and python library to build image segmentation
systems. Its main aim is productivity and end-to-end capabilities.

Bauta is based on a Deep Neural network that implements state-of-the-art research
for image segmentation using *data augmentation*.

(TODO: List research papers)


## Requirements
The library supports Ubuntu Linux and Mac OS, and both CPU and GPU.
You will have to install [anaconda](https://conda.io/miniconda.html).
To know which anaconda you shall install you should check the default
python version you have on your system by running
```
python --version
```
That command should be either a *2.7.y* or *3.7.w* version, depending on that
you will have to install one version or another of anaconda.
If you don't have an initial python install on your system then you will
have to install python.

## Setup
To install bauta and set up the environment simply run
```
./install.sh
```
This will create the environment `conda`.
It is important that you have the environment enabled all the time you use
*bauta* as otherwise you will only get python errors.
To activate the environment run
```
conda activate bauta
```

## Dataset
The dataset is composed of two types of images:
- **Objects**: these are png images with transparent background of the different
classes you support. In case of a dataset of pets, you will have to provide
images for dogs and cats with transparent backgrounds
- **Backgrounds**: just images of backgrounds you expect to have the objects on.
In the case of a dataset of pets that would be photos of the street and photos of a backyard.


The dataset is stored in what's called the `data path` where models and dataset are stored.
The images should be stored in the following folders
```
DATA_FOLDER/dataset/augmentation/test/<CLASS_NAME>
DATA_FOLDER/dataset/augmentation/train/<CLASS_NAME>
```

There is a compulsory class called `background` that holds the background.
All the other classes are expected to be `png`s with alpha channel.


### Dataset Generation Tool
There is a script `setup_dataset.py` that allows you to create the whole
dataset together with the configuration file by using list of image paths or URLs.

First you will have to create a dataset and for that you will need either
paths to images or URLs to them.
The URLs/paths have to be stored in a `txt` file named after the class name (e.g. `cat.txt`, `dog.txt`, `background.txt`, ...).
Keep in mind that the file `background.txt` is compulsory.
Furthermore, all the other images must be `png` files with alpha channel.

In the pets example, you will have three files:
```
background.txt
dog.txt
cat.txt
```

Once you have the txt files in a single folder you will have to run
```
./setup_dataset
```

This script will create the a **data path** with seveal subfolders where the dataset and models are stored
as well as downloading the images and splitting them into test and train.
The dataset configuration will be stored in the file `DATA_FOLDER/config.yaml` and contains a single attribute with the
lists of classes.

For example, say that we have a dataset of pets supporting the class `dog`
and `cat`. This means that the `config.yaml` will be the following:
```
classes:
  - cat
  - dog
```

The script will also create the following folders and will fill them
with the train and test split.
```
DATA_FOLDER/dataset/augmentation/test/background
DATA_FOLDER/dataset/augmentation/train/background
DATA_FOLDER/dataset/augmentation/test/cat
DATA_FOLDER/dataset/augmentation/train/cat
DATA_FOLDER/dataset/augmentation/test/dog
DATA_FOLDER/dataset/augmentation/train/dog
```

### Optional questions for generating the dataset
During the script execution, a few questions will be asked:

* The Base path where all the data will be stored

* The path with the txt files containing the list of image URLs or paths (optional).

* If the path of the txt file is not provided, the class names will be asked


## Training
To start the first training run (changing the *DATA_FOLDER* for the actual
  full path of your dataset).
```
python train.py --data_path=DATA_FOLDER --reset_model=y --batch_size=4
```

For next time you try to train, you will have to reuse the model and thus
```
python train.py --data_path=DATA_FOLDER --batch_size=4
```
