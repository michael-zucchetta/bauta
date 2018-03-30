#!/usr/bin/env python
import logging
import os
import threading
import sys
import time
import torch
import readline
from functools import reduce

from bauta.DatasetGenerator import DatasetGenerator

logging.getLogger('DatasetGenerator').setLevel(logging.WARNING)
readline.parse_and_bind("tab: complete")

def askInput(description, action, compulsory=False):
    print(description)
    result = input(action)
    if compulsory and result == '':
        print('It cannot be empty')
        askInput(description, action, compulsory)
    return result

if __name__ == '__main__':
    data_path = askInput('Please, introduce the "data path" where the dataset, model and other binary files will be stored (e.g. /home/myusername/trees_classification)', 'Enter the path: ', True)
    dataset_generator = DatasetGenerator(data_path)
    dataset_generator.makeDefaultDirs()

    datasets_path = askInput('Please, insert the path containing the txt files listing the images (the txt file name will be the class of those images). It can be empty', '(Optional) Enter the path: ')
    if datasets_path:
        class_names = dataset_generator.generateDatasetFromListOfImages(datasets_path, 0.1, 5)
        if class_names is None:
            print(f'ERROR: The file "background.txt" is compulsory but it was not found. Thus dataset cannot be created')
        else:
            print(f'Found the following classes: {class_names}')
