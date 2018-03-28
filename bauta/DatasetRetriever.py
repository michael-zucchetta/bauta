import csv
import os
import random
from shutil import copyfile

import itertools
import urllib.request
from urllib import parse

from bauta.Environment import Environment

CSV_FILE_EXTENSIONS = ['.csv', '.txt']
def is_csv_extension(file_extension):
  for csv_file_extension in CSV_FILE_EXTENSIONS:
    if file_extension == csv_file_extension:
      return True
  return False

class DatasetRetriever():
    def __init__(self,
            data_path,
            datasets_path,
            is_background=False,
            split_test_proportion=0.3,
            download_batch_size=5,
            ):
        self.split_test_proportion = float(split_test_proportion)
        self.download_batch_size = download_batch_size
        self.datasets_path = datasets_path
        self.is_background = is_background
        self.env = Environment(data_path=data_path)
        self.logger = self.env.getLogger(type(self).__name__)
        datasets_with_attributes = []
        if is_background:
            self.logger.info('Retrieving backgrounds')
            if os.path.isfile(datasets_path):
                if is_csv_extension(os.path.splitext(datasets_path)[1]):
                    (train_set, test_set) = self.readCsvAndSplit(datasets_path)
                    self.retrieveImages(train_set, dataset_type='train')
                    self.retrieveImages(test_set, dataset_type='test')
            else:
                for file_in_path in os.listdir(datasets_path):
                    if is_csv_extension(os.path.splitext(datasets_path)[1]):
                        (train_set, test_set) = self.readCsvAndSplit('{datasets_path}/{file_in_path}')
                        self.retrieveImages(train_set, dataset_type='train')
                        self.retrieveImages(test_set, dataset_type='test')
        else:
            self.retrieveDatasets()


    def retrieveDatasets(self):
        datasets_with_attributes = []
        for file_in_path in os.listdir(self.datasets_path):
            if is_csv_extension(os.path.splitext(file_in_path)[1]):
                class_name = os.path.splitext(file_in_path)[0]
                (training_set, test_set) = self.readCsvAndSplit(f'{self.datasets_path}/{file_in_path}', True)
                self.env.makeDirIfNotExists(f'{self.env.objects_path}train/{class_name}')
                self.env.makeDirIfNotExists(f'{self.env.objects_path}test/{class_name}')
                datasets_with_attributes.append({'dataset': self.createGroupedDataset(training_set), 'class': class_name, 'type': 'train'})
                datasets_with_attributes.append({'dataset': self.createGroupedDataset(test_set), 'class': class_name, 'type': 'test'})
        datasets = [dataset_with_attributes['dataset'] for dataset_with_attributes in datasets_with_attributes]
        for zipped_dataset in itertools.zip_longest(*datasets):
            for (index, batch_dataset) in enumerate(zipped_dataset):
                if batch_dataset is not None:
                    self.retrieveImages(batch_dataset, datasets_with_attributes[index]['class'], datasets_with_attributes[index]['type']) 

    def createGroupedDataset(self, dataset):
        grouped_dataset = []
        index = 0
        while index < len(dataset):
            grouped_dataset.append(dataset[index : index + self.download_batch_size])
            index = index + self.download_batch_size
        return grouped_dataset

    def readCsvAndSplit(self, csv_file_path, split=False):
        ids_and_images = []
        try:
            with open(csv_file_path, 'r') as csv_file:
                ids_and_images = list(csv.reader(csv_file))
                if len(ids_and_images) > 0 and len(ids_and_images[0]) == 1:
                  # if the csv file has only 1 element per row, we're assuming it's the image url, and we are adding its index
                  ids_and_images = [(index, image[0]) for (index, image) in enumerate(ids_and_images) if len(image) > 0]

                if len(ids_and_images) > 0 and len(ids_and_images[0]) > 2:
                  # if the csv contains extra columns
                  ids_and_images = [row[0:2] for row in ids_and_images]

                #if split:
                random.shuffle(ids_and_images)
                dataset_size = len(ids_and_images)
                test_size = int(dataset_size * self.split_test_proportion)

                (training_ids_and_images, test_ids_and_images) = ids_and_images[test_size:], ids_and_images[:test_size]
                ids_and_images = [training_ids_and_images, test_ids_and_images]
        except BaseException as e:
            self.logger.error(f'Something went wrong while retrieving and parsing the csv file {csv_file_path}', e)
        except ValueError as e:
            self.logger.error(f'Something went wrong while retrieving and parsing the csv file {csv_file_path}', e)
        return ids_and_images

    def retrieveImages(self, ids_and_images, class_name=None, dataset_type=None):
        if self.is_background:
            base_path = f'{self.env.backgrounds_path}{dataset_type}'
        else:
            base_path = f'{self.env.objects_path}{dataset_type}/{class_name}'
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        for (image_id, image_url) in ids_and_images:
            parsed_url = parse.urlparse(image_url)
            image_url_path = parsed_url.path
            image_ext = os.path.splitext(image_url_path)[1]
            local_file_path = f'{base_path}/{image_id}{image_ext}'
            if class_name and dataset_type:
                self.logger.info(f'Retrieving image {image_url} for class {class_name} and it is put on {base_path} on the {dataset_type} set') 
            def retrieveAndStoreImage():
                if parsed_url.scheme:
                    urllib.request.urlretrieve(image_url, local_file_path)
                else:
                    copyfile(image_url, local_file_path)
                return local_file_path
            def validateStoredImage(file_path_destination):
                return os.path.isfile(file_path_destination)
            try:
                self.env.tryToRun(retrieveAndStoreImage, validateStoredImage, 5)
            except BaseException as e:
                self.logger.error(f'Error while downloading {image_url} because of {e}, skipping')
