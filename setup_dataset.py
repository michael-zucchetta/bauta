#!/usr/bin/env python
import logging
import os
import sys
import click
from bauta.DatasetGenerator import DatasetGenerator


@click.command()
@click.option('--data_path', default='', help='Data path where the models and the images as well as other binary files will be stored (e.g. /home/myusername/trees).')
@click.option('--images_path', default=None, help='Path containing the txt files listing the images (the txt file name will be the class of those images, a "background.txt" file is compulsory)')
@click.option('--start_image_download', default=True, help='The splitted dataset will be downloaded')
@click.option('--background_classes', default=[], multiple=True, help='The classes that are expected to appear frequently on background, hence not trained and used as part of the background. A good example can be objects for the house, cars or animals')
def datasetGenerator(data_path, images_path, start_image_download, background_classes):
    logging.getLogger('DatasetGenerator').setLevel(logging.WARNING)
    if data_path is '':
        logging.error('"data_path" cannot be empty. Please, use "./setup_dataset.py --help" to see arguments')
        sys.exit(-1)
    if type(start_image_download) == str and (start_image_download == 'no' or start_image_download == 'n'):
        start_image_download = False
    background_classes = list(background_classes)
    dataset_generator = DatasetGenerator(data_path)
    dataset_generator.makeDefaultDirs()
    class_names = dataset_generator.generateDatasetFromListOfImages(images_path, 0.1, 5, start_image_download, background_classes)
    if class_names is None:
        logging.error(f'Errors have been found and thus the data path could not be created. See list of errors above this message.')
        sys.exit(-1)
    else:
        print(f'Dataset folder structure in {data_path} successfully created. The following classes were found: {class_names}')

if __name__ == '__main__':
    datasetGenerator()
