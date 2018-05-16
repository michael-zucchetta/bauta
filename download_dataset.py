#!/usr/bin/env python
import logging
import os
import sys
import click
from bauta.DatasetGenerator import DatasetGenerator

@click.command()
@click.option('--data_path', default='', help='Data path where the models and the images as well as other binary files will be stored (e.g. /home/myusername/trees).')
def downloadDataset(data_path): 
    logging.getLogger('DatasetGenerator').setLevel(logging.WARNING)
    if data_path is '':
        logging.error('"data_path" cannot be empty. Please, use "./setup_dataset.py --help" to see arguments')
        sys.exit(-1)
    dataset_generator = DatasetGenerator(data_path)
    class_names = dataset_generator.downloadImages(5)
    if class_names is None:
        logging.error(f'Errors have been found and thus the data path could not be created. See list of errors above this message.')
        sys.exit(-1)
    else:
        print(f'Dataset folder structure in {data_path} successfully created. The following classes were found: {class_names}')

if __name__ == '__main__':
    downloadDataset()
