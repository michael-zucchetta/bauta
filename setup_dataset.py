#!/usr/bin/env python
import logging
import os
import sys
import click
from bauta.DatasetGenerator import DatasetGenerator

@click.command()
@click.option('--data_path', default='', help='Data path where the models and the images as well as other binary files will be stored (e.g. /home/myusername/trees).')
@click.option('--images_path', default='', help='Path containing the txt files listing the images (the txt file name will be the class of those images, a "background.txt" file is compulsory)')
def datasetGenerator(data_path, images_path):
    logging.getLogger('DatasetGenerator').setLevel(logging.WARNING)
    if data_path is '':
        logging.error('"data_path" cannot be empty. Please, use "./setup_dataset.py --help" to see arguments')
        sys.exit(-1)
    if images_path is '':
        logging.error('"images_path" cannot be empty. Please, use "./setup_dataset.py --help" to see arguments')
        sys.exit(-1)
    dataset_generator = DatasetGenerator(data_path)
    dataset_generator.makeDefaultDirs()
    class_names = dataset_generator.generateDatasetFromListOfImages(images_path, 0.1, 5)
    if class_names is None:
        logging.error(f'Errors have been found and thus the data path could not be created. See list of errors above this message.')
        sys.exit(-1)
    else:
        print(f'Dataset folder structure in {data_path} successfully created. The following classes were found: {class_names}')

if __name__ == '__main__':
    datasetGenerator()
