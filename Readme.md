# Bauta
Bauta is a Deep Neural network that implements state-of-the-art research
for image segmentation using *data augmentation*.

(TODO: List research papers)

Its main aim is to be an easy-to-use tool and library that allow image
segmentation at scale with minimum effort.

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
First step is to create a dataset.
The dataset will contain the following elements:
- **Objects**: these are png images with transparent background of the different
classes you support. In case of a dataset of pets, you will have to provide
images for dogs and cats with transparent backgrounds
- **Backgrouds**: a list of available backgrounds. In the case of a dataset of
pets that would be photos of the street and photos of a backyard.

Once you have the data you should run the following script
```
./setup_dataset
```

This script will create a toy dataset. The dataset configuration will be in
the file `DATA_FOLDER/config.yaml` and contains a sigle attribute with the
lists of classes.

For example, say that we have a dataset of pets supporting the class `dog`
and `cat`. This means that the `config.yaml` will be the following:
```
classes:
  - cat
  - dog
```

Now it's time to move your images to the right folders.
You will have to first split the images into test and training set.

and then create the following folders:

* `DATA_FOLDER/dataset/augmentation/objects/test/dog`

* `DATA_FOLDER/dataset/augmentation/objects/train/dog`

* `DATA_FOLDER/dataset/augmentation/objects/test/dog`

* `DATA_FOLDER/dataset/augmentation/objects/train/cat`

* `DATA_FOLDER/dataset/augmentation/backgrounds/test`

* `DATA_FOLDER/dataset/augmentation/backgrounds/train`

And add your images into them.

# Optional questions for generating the dataset
During the script execution, a few questions will be asked:

* The Base path where all the data will be stored

* The path with the CSV files containing the list of images (optional).
Only the first two columns will be considered: the first will be used as id, the second as image url (if the CSV contains only one column, then, it will use the index as id)

* If the path of the CSVs file will not be provided, the class names will be asked

* The path of the CSVs with the list of background URLs (optional)


## Training
To start the first training run (changing the *DATA_FOLDER* for the actual
  full path of your dataset).
```
python train.py --data_path=DATA_FOLDER --reset_model=y  --batch_size=4
```

For next time you try to train, you'll have to reuse the model and thus
```
python train.py --data_path=DATA_FOLDER   --batch_size=4
```
