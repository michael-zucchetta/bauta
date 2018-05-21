# Bauta
[Bauta](https://en.wikipedia.org/wiki/Carnival_of_Venice#Bauta) is a generic toolset and python library to build image segmentation systems with minimal effort.

`bauta` uses data augmentation to train a deep convolutional network to segment images. See [Foundations](#foundations) for further details.

## Requirements
The library supports Ubuntu Linux and Mac OS, and both CPU and GPU.
You will have to install [anaconda](https://conda.io/miniconda.html).
To know which anaconda you shall install you should check the default python version you have on your system by running
```
python --version
```
If you do not have an initial python install on your system then you will have to install python first.

To enable `anaconda` for first time, run:
 - Mac OS:
```
source  ~/.bash_profile
```

 - Ubuntu:
```
source  ~/.bashrc
```

## Setup
To install `bauta` and set up the environment simply run
```
./install.sh
```
This will create the environment `conda`.

To activate the environment run
```
source activate bauta
```

** WATCH OUT: It is important that you have the correct environment enabled for all the time you use `bauta` as otherwise `bauta` will not function in any capacity **

### Dataset Generation Tool
The script `setup_dataset.py` that allows creating a dataset and environment to get ready to train the segmentation network.

First you will have to have a list of files that contain URLs of the different objects you want to segment as well as backgrounds.
The URLs have to be stored in a `txt` file named after the class name (e.g. `cat.txt`, `dog.txt`, `background.txt`, ...).
Keep in mind that the file `background.txt` is compulsory.
Furthermore, all the object images must be `png` files with alpha channel or `jpg` images with flat-color background (for example, a cat in the center of the image with white background). Background images can be either `jpg` or `png`.

For the pets example, there would be three files:
```
background.txt
dog.txt
cat.txt
```

Once you have the txt files in a single folder you have to run
```
source activate bauta
./setup_dataset --images_path=/home/MY_USER/pets_images --data_path=/home/MY_USER/pets
```

 - `images_paths` shall point to the folder containing all the `txt` files holding the URLs to the images.
 - `data_path` shall point to a folder where the split will be stored and its images downloaded downloaded.


The `data_path` will contain several subfolders where the dataset, models and test and train images are stored.

Each directory in the train and test dataset will contain a file with the list of images URLs. If not specified otherwise, it will download the images as well.
The dataset configuration will be stored in the file `DATA_FOLDER/config.yaml` and contains a single attribute with the lists of classes.

For example, say that we have a dataset of pets supporting the class `dog` and `cat`. This means that the `config.yaml` will be the following:
```
classes:
  - cat
  - dog
```

The script will also create the following folders and will fill them with the train and test split.
```
DATA_FOLDER/dataset/augmentation/test/background
DATA_FOLDER/dataset/augmentation/train/background
DATA_FOLDER/dataset/augmentation/test/cat
DATA_FOLDER/dataset/augmentation/train/cat
DATA_FOLDER/dataset/augmentation/test/dog
DATA_FOLDER/dataset/augmentation/train/dog
```

## Training
To start the first training run:
```
source activate bauta
python train.py --data_path=/home/MY_USER/pets --batch_size=32
```

For a complete list of available parameters run:
```
source activate bauta
python train.py --help
```

## Inference
Inference can be done either for one single image or a set of images on a folder.
```
source activate bauta
python inference.py --data_path=/home/MY_USER/pets --path=/home/MY_USER/Downloads/garden.jpg --result_folder=/home/MY_USER/pets_in_garden
```

The `data_path` should point as usual to the data_path of the model and the configuration file that will be used. The `path` is either the path of an image or the path of a folder that contains images. The `result_folder` is the folder were the inference results are stored, creating one folder per image with the different objects found.

For a complete list of available parameters run:
```
source activate bauta
python inference.py --help
```

## Dataset
This section only discusses how the dataset is organized and the type and purpose of the different images required for training. If you want to directly generate the dataset see [Dataset Generation Tool](#dataset-generation-tool)

The dataset is composed of two types of images:
- **Objects**: these are `png` images with transparent background (with alpha channel) of the different classes you support. We also support both `png` without alpha channel and `jpg` images with flat color backgrounds and the object. For the latter case, Bauta will automatically remove the background as 'best effort'.

In case of a dataset of pets, you will have to provide images for dogs and cats. These images will have to either contain alpha channel in the background or a flat color as background.

- **Backgrounds**: just images of backgrounds you expect to have the objects on.
In the case of a dataset of pets that would be photos of the street and photos of a backyard.

All the binary files are stored in what we call `data_path`, where the models, dataset, and the images composing the dataset are stored.
The images should be stored in the following folders:
```
DATA_FOLDER/dataset/augmentation/test/<CLASS_NAME>
DATA_FOLDER/dataset/augmentation/train/<CLASS_NAME>
```
The dataset folder contains the directories:

* *augmentation*: the path where the test and training images are located.
Inside the test and training directory lies a list of directories equivalent to the list of supported classes with the images themselves and a file which contains the list of URLs along their ids

* Two directories, *train* and *test*, each containing the its list of augmented images.

There is a compulsory class called `background` representing the background.
All the other classes are expected to be `png`s with an alpha channel or a `jpg` with flat background color.

## Foundations

`Bauta` is based on a Deep Neural network that implements a mixture of state-of-the-art research for image segmentation together with custom modifications:
* Fisher Yu, Vladlen Koltun (2016). [Multi-Scale Context aggregation by dilated convolutions](https://arxiv.org/pdf/1511.07122.pdf).

* Lin, Tsung-Yi & Goyal, Priya & Girshick, Ross & He, Kaiming & Dollár, Piotr. (2017). [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

* He, Kaiming & Zhang, Xiangyu & Ren, Shaoqing & Sun, Jian. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) and the Facebook implementation [ResNet training in Torch](https://github.com/facebook/fb.resnet.torch)

These papers have served as a base for `bauta` but with the following important differences:
 - Instead of upsampling embeddings in the backbone, the
embeddings are downsampled to the smallest spatial dimensions of the embeddings.
 - The bounding box regression is substituted by a down-scaled mask.
   - Additionally the anchors are swapped by dilated convolutions, serving a similar purpose without requiring high amounts of memory as they work on downscaled spatial dimensions.
 - The object classifier is removed and it is assumed that if in the downscaled mask there is no activation then the object does not exists on the image (otherwise, it does exist and it is located on the activations).
 - There is a final refiner that is class-independent and smoothes the downsampled mask into an upsampled mask. As it only locally refines masks, there are no dilated convolutions and thus it is memory usage is low.
