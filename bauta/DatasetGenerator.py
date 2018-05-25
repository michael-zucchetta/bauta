import os
import random
import shutil
import hashlib
import itertools
import socket

import urllib.request
from urllib import parse
from urllib.error import URLError
from bauta.utils.EnvironmentUtils import EnvironmentUtils
from bauta.utils.SystemUtils import SystemUtils
from bauta.Constants import constants
from random import shuffle

class DatasetGenerator():

    def __init__(self, data_path):
        socket.setdefaulttimeout(10)
        self.data_path = data_path
        self.system = SystemUtils()
        self.environment = EnvironmentUtils(self.data_path)
        self.logger = self.system.getLogger(self)

    def createConfigurationFile(self, class_names, background_classes):
        try:
            class_names_without_background = list(filter(lambda class_name: class_name != constants.background_label, class_names))
            class_names_without_background.sort()
            configuration_file_path = os.path.join(self.data_path, "config.yaml")
            if not os.path.isfile(configuration_file_path):
                with open(configuration_file_path, 'w') as config_yaml_file:
                    classes_as_string = [f'  - {class_name}\n' for class_name in class_names_without_background]
                    classes_yaml  = 'classes:\n'
                    config_yaml_file.write(classes_yaml)
                    [config_yaml_file.write(class_as_string) for class_as_string in classes_as_string]
                    background_classes_yaml = 'background_classes:\n'
                    config_yaml_file.write(background_classes_yaml)
                    [config_yaml_file.write(f'  - {background_class}\n') for background_class in background_classes]
                    self.logger.info('config.yaml created')
        except BaseException as e:
            self.logger.error(f'Cannot open configuration file "{configuration_file_path}"', e)

    def generateDatasetFromListOfImages(self, images_path, split_test_proportion, download_batch_size, download_images, background_classes):
        self.makeDefaultDirs()
        if images_path is not None:
            if not os.path.isdir(images_path):
                self.logger.error(f'"{images_path}" is not an existing directory')
                return None
            class_names = [os.path.splitext(file_in_path)[0] for file_in_path in os.listdir(images_path)]
            for background_class in background_classes:
                if not background_class in class_names:
                    self.logger.error(f'Specified noise class {background_class} is not in the directory {images_path}')
                    return None
            foreground_class_names = set(class_names) - set(background_classes)

            if constants.background_label not in foreground_class_names:
                 self.logger.error(f'The class "background" is compulsory but it was not found. Thus dataset cannot be created.', e)
                 return None
            if len(foreground_class_names) <= 1:
                self.logger.error(f'There should be at least one class besides the "background" one (e.g. "cat" and "background").', e)
                return None
            self.createConfigurationFile(foreground_class_names, background_classes)
            image_paths = [os.path.join(images_path, file_in_path) for file_in_path in os.listdir(images_path) \
                if self.system.hasExtension(os.path.join(images_path, file_in_path), ['txt'])]
            class_names = []
            for image_path in image_paths:
                file_in_path = os.path.basename(image_path)
                class_name = os.path.splitext(file_in_path)[0]
                train_path = self.environment.objectsFolder(class_name, is_train=True)
                test_path  = self.environment.objectsFolder(class_name, is_train=False)
                self.system.makeDirIfNotExists(train_path)
                self.system.makeDirIfNotExists(test_path)

                existing_train_images = self.readImages( os.path.join(train_path, f'{class_name}.txt'), True )
                existing_test_images = self.readImages( os.path.join(test_path, f'{class_name}.txt'), True )
                images = self.readImages(image_path)
                training_set, test_set = self.testTrainSplit(images, existing_train_images, existing_test_images, split_test_proportion)
                shuffle(training_set)
                shuffle(test_set)
                self.writeImages(f'{train_path}/{class_name}.txt', training_set)
                self.writeImages(f'{test_path}/{class_name}.txt', test_set)
                class_names.append(class_name)
        if download_images:
            self.downloadImages(download_batch_size)
        return class_names

    def downloadImages(self, download_batch_size):
        datasets_with_attributes = []
        class_names = []
        for is_train in [True, False]:
            for (class_name, class_path) in self.environment.classesInDatasetFolder(is_train):
                class_file = os.path.join(class_path, f'{class_name}.txt')
                dataset = self.readImages(class_file, True)
                datasets_with_attributes.append({'dataset': self.createGroupedDataset(dataset, download_batch_size), 'class': class_name, 'is_train': is_train})
                class_names.append(class_name)

        datasets = [dataset_with_attributes['dataset'] for dataset_with_attributes in datasets_with_attributes]
        for zipped_dataset in itertools.zip_longest(*datasets):
            for (index, batch_dataset) in enumerate(zipped_dataset):
                if batch_dataset is not None:
                    batch_dataset = list(filter(lambda batch_element: type(batch_element) == type(list()), batch_dataset))
                    self.retrieveImages(batch_dataset, datasets_with_attributes[index]['class'], datasets_with_attributes[index]['is_train'])
        return class_names

    def makeDefaultDirs(self):
        essential_paths = [
            f'{self.data_path}/dataset/augmentation/test',
            f'{self.data_path}/dataset/augmentation/train',
            f'{self.data_path}/dataset/test',
            f'{self.data_path}/dataset/train',
            f'{self.data_path}/dataset/validation',
            f'{self.data_path}/models'
        ]
        [self.system.makeDirIfNotExists(essential_path) for essential_path in essential_paths]

    def createGroupedDataset(self, dataset, download_batch_size):
        return [ dataset[batch : batch + download_batch_size ] for batch in range(0, len(dataset), download_batch_size)]

    def readImages(self, txt_file_path, read_id=False):
        try:
            if not os.path.isfile(txt_file_path):
                return []
            else:
                with open(txt_file_path, 'r') as txt_file:
                    lines = txt_file.read().splitlines()
                    if read_id == False:
                        return lines
                    else:
                        return [line.split(',') for line in lines]
        except BaseException as e:
            self.logger.error(f'Cannot open and parse txt file "{txt_file_path}"', e)
            return []

    def writeImages(self, txt_file_path, image_urls):
        try:
            with open(txt_file_path, 'w') as txt_file_buffer:
                [txt_file_buffer.write(f'{image_id},{image_url}\n') for (image_id, image_url) in image_urls]
        except BaseException as e:
            self.logger.error(f'Cannot open and parse txt file "{txt_file_path}"', e)
            return False
        return True

    def testTrainSplit(self, images, existing_train_images, existing_test_images, split_test_proportion):
        ids_to_images = [(hashlib.md5(bytes(image, encoding='utf-8')).hexdigest(), image) for image in images]
        existing_image_ids = [os.path.splitext(image[0])[0] for image in existing_train_images + existing_test_images]
        new_ids_to_images = list(filter(lambda id_to_image: id_to_image[0] not in existing_image_ids, ids_to_images))
        wanted_test_size = min( len(new_ids_to_images), \
            int( ( len(new_ids_to_images) + len(existing_test_images) + len(existing_train_images) ) * split_test_proportion ) )
        current_test_size = len(existing_test_images)
        test_to_add = wanted_test_size - current_test_size
        if test_to_add > 0:
            ids_to_train_images, ids_to_test_images = new_ids_to_images[test_to_add:], new_ids_to_images[:test_to_add]
        else:
            ids_to_train_images, ids_to_test_images = new_ids_to_images, []
        return ids_to_train_images + existing_train_images, ids_to_test_images + existing_test_images

    def retrieveImages(self, images, class_name, is_train):
        for (image_id, image_url) in images:
            parsed_url = urllib.parse.urlparse(image_url)
            image_url_path = parsed_url.path
            name_and_extension = os.path.splitext(image_url_path)
            if len(name_and_extension) == 2:
                image_extension = name_and_extension[1]
                if image_extension == '':
                    image_extension = '.jpg'
            else:
                image_extension = '.png'
            local_file_path = os.path.join(self.environment.objectsFolder(class_name, is_train), f'{image_id}{image_extension}')
            if os.path.isfile(local_file_path):
                continue
            self.logger.info(f'Retrieving image "{image_url}" for class "{class_name}" and saving it in "{local_file_path}"')
            def retrieveAndStoreImage():
                if parsed_url.scheme:
                    try:
                        urllib.request.urlretrieve(image_url, local_file_path)
                    except URLError as e:
                        self.logger.error(f'Error retrieving image with {image_url}')
                else:
                    shutil.copyfile(image_url, local_file_path)
                return local_file_path
            def validateStoredImage(file_path_destination):
                return os.path.isfile(file_path_destination)
            try:
                self.system.tryToRun(retrieveAndStoreImage, validateStoredImage, 2, silent=True)
            except BaseException as e:
                self.logger.error(f'Cannot download and/or move image id "{image_id}" with URI "{image_url}"', e)
