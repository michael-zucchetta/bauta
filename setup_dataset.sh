#!/bin/bash
#
# GET DATA PATH
#
echo "Please, introduce the 'data' path where the dataset, model and other binary files will be stored (e.g. /home/myusername/trees_classification)"
read data_path
if [ ! -d "$data_path" ]; then
  echo "Creating folder $data_path as it hasn't been found..."
  mkdir -p "$data_path"
fi
#
# CREATE DATASET FOLDER STRUCTURE
#
mkdir -p "$data_path/dataset/augmentation/backgrounds/test"
mkdir -p "$data_path/dataset/augmentation/backgrounds/train"
mkdir -p "$data_path/dataset/augmentation/objects/test"
mkdir -p "$data_path/dataset/augmentation/objects/train"
mkdir -p "$data_path/dataset/test"
mkdir -p "$data_path/dataset/train"
mkdir -p "$data_path/dataset/validation"
mkdir -p "$data_path/models"
echo $'classes:\n  -cat\n  -dog' > "$data_path/config.yaml"
echo "Setup finished."
