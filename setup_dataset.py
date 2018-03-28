#!/usr/bin/env python
import logging
import os
import threading
import sys
import time
import torch

from functools import reduce

from bauta.DatasetRetriever import DatasetRetriever

def makeDirIfNotExists(path):
    if not os.path.exists(path):
        os.makedirs(path)

logging.getLogger('DatasetRetriever').setLevel(logging.WARNING)

def makeDefaultDirs(data_path):
    makeDirIfNotExists(data_path)
    essential_paths = [
        f'{data_path}/dataset/augmentation/backgrounds/test',
        f'{data_path}/dataset/augmentation/backgrounds/train',
        f'{data_path}/dataset/augmentation/objects/test',
        f'{data_path}/dataset/augmentation/objects/train',
        f'{data_path}/dataset/test',
        f'{data_path}/dataset/train',
        f'{data_path}/dataset/validation',
        f'{data_path}/models'
    ]

    [makeDirIfNotExists(essential_path) for essential_path in essential_paths]

def askInput(msg1, msg2, compulsory=False):
    print(msg1)
    result = input(msg2)
    if compulsory and result == '':
        print('It cannot be empty')
        askInput(msg1, msg2, compulsory)
    return result

if __name__ == '__main__':
    data_path = askInput('Please, introduce the \'data'' path where the dataset, model and other binary files will be stored (e.g. /home/myusername/trees_classification)', 'Enter the path: ', True)
    makeDefaultDirs(data_path)
    extra_questions = askInput('Do you want to generate a dataset with training data and backgrounds?', 'Yes or No ') or 'No'

    if extra_questions.lower() == 'no':
        print('Setup completed')
        sys.exit(0)

    datasets_path = askInput('Please, insert the path containing the CSV files listing the images (the CSV file name will be the class of those images). It can be empty', '(Optional) Enter the path: ')
    threads = []
    if datasets_path:
        def nonBlockingRetrieveDatasets():
            DatasetRetriever(data_path, datasets_path)
        t1 = threading.Thread(target=nonBlockingRetrieveDatasets)
        t1.start()
        threads.append(t1)
        time.sleep(10) # waiting for classes creation
        class_names = os.listdir(f'{data_path}/dataset/augmentation/objects/train/')
        print(f'Found the following classes {class_names}')
    else:
        class_names = askInput('Insert the default class names', '(Optional) Insert classes separated by a space')
        if class_names == '':
            print('Setting dummy classes')
            class_names = 'cat dog'
        class_names = class_names.split(' ')

    backgrounds_path = askInput('Please, insert the path with the CSV with the background urls', '(Optional) Insert csv path file: ')
    if backgrounds_path:
        def nonBlockingRetrieveBackgrounds():
            DatasetRetriever(data_path, backgrounds_path, True)
        t2 = threading.Thread(target=nonBlockingRetrieveBackgrounds)
        t2.start()
        threads.append(t2)
  
    classes_as_string = [f'  - {class_name}\n' for class_name in class_names]
    classes_yaml  = 'classes:\n'

    config_yaml_file = open(f'{data_path}/config.yaml', 'w')
    config_yaml_file.write(classes_yaml)
    [config_yaml_file.write(class_as_string) for class_as_string in classes_as_string]
    config_yaml_file.close()
    print('config.yaml created')
    print('Waiting for the threads downloading the images')
    if len(threads) > 0:
        [thread.join() for thread in threads]
