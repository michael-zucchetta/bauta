# Bauta
[Bauta](https://en.wikipedia.org/wiki/Carnival_of_Venice#Bauta) is an easy-to-use toolset and python library to build image segmentation
systems. Its main aim is productivity and end-to-end capabilities.

Bauta is based on a Deep Neural network that implements state-of-the-art research
for image segmentation using *data augmentation*.

(TODO: List research papers)

* Lin, Tsung-Yi & Goyal, Priya & Girshick, Ross & He, Kaiming & Dollár, Piotr. (2017). [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

* He, Kaiming & Zhang, Xiangyu & Ren, Shaoqing & Sun, Jian. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) and the Facebook implementation [ResNet training in Torch](https://github.com/facebook/fb.resnet.torch)


## Requirements
The library supports Linux (we used Ubuntu) and Mac OS, and both CPU and GPU.
You will have to install [anaconda](https://conda.io/miniconda.html).
To know which anaconda you shall install you should check the default
python version you have on your system by running
```
python --version
```
That command should be *3.6.w* version as previous versions might work, but they have not
been tested. Depending on that you will have to install the correct version of anaconda.
If you do not have an initial python install on your system then you will
have to install python first.

## Setup
To install bauta and set up the environment simply run
```
./install.sh
```
This will create the environment `conda`.
It is important that you have the correct environment enabled for all the time you use
*bauta* as otherwise you will only get python errors.
To activate the environment run:
```
conda activate bauta
```
or:
```
source activate bauta
```
according to your configuration.

## Dataset
The dataset is composed of two types of images:
- **Objects**: these are png images with transparent background of the different
classes you support. In case of a dataset of pets, you will have to provide
images for dogs and cats with transparent backgrounds
- **Backgrounds**: just images of backgrounds you expect to have the objects on.
In the case of a dataset of pets that would be photos of the street and photos of a backyard.


The dataset is stored in the parameter called `data_path`, where the models and the
images composing the dataset are stored.
The images should be stored in the following folders:
```
DATA_FOLDER/dataset/augmentation/test/<CLASS_NAME>
DATA_FOLDER/dataset/augmentation/train/<CLASS_NAME>
```

The dataset folder contains the directories:

* augmentation: the path where the test and training images are being put.
Inside the test and training directory lies a list of directories equivalent
to the list of supported classes with the images themselves and a file which
contains the list of URLs along their ids

* Two directories, train and set, belonging to each of the sets containing
the list of augmented images belonging to that set

There is a compulsory class called `background` representing the background.
All the other classes are expected to be `png`s with an alpha channel.


### Dataset Generation Tool
There is a script `setup_dataset.py` that allows you to create the whole
dataset as well as the configuration file by using list of image paths or URLs.

First you will have to create a dataset and for that you will need either
paths to images or URLs to them.
The URLs/paths have to be stored in a `txt` file named after the class name (e.g. `cat.txt`, `dog.txt`, `background.txt`, ...).
Keep in mind that the file `background.txt` is compulsory.
Furthermore, all the other images must be `png` files with alpha channel.

In the pets example, there are three files:
```
background.txt
dog.txt
cat.txt
```

Once you have the txt files in a single folder you have to run
```
./setup_dataset
```

This script creates the main directory in the specified **data path** with several subfolders where the dataset and models are stored and split into test and train.
Each directory in the train and test dataset will contain a file with the list of images URLs. If not specified otherwise, it will download the images as well.
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

* A parameter can be specified to download the images, so that only the splitted images as a list
are put in a file

## Training
To start the first training run (changing the *DATA_FOLDER* for the actual
  full path of your dataset).
```
python train.py --data_path=DATA_FOLDER --reset_model=y --batch_size=4
```

For next time you try to train, you will have to reuse the model and thus:
```
python train.py --data_path=DATA_FOLDER --batch_size=4
```

Other optional parameters are:

* `learning_rate`: the learning rate for the bauta network

* `momentum`: the momentum of the Stochastic Gradient Descent

* `gpu`: which gpu you prefer to use

* `visual_logging`: provides the visual representation of what is going on during the training (eg. check the masks at some stage)
