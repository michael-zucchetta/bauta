import math
import os, random, string
import sys
import yaml
import traceback
import operator, functools

from bauta.Constants import constants
from bauta.utils.SystemUtils import SystemUtils

class DatasetConfiguration():

    def __init__(self, is_train, data_path):
        system_utils = SystemUtils()
        self.data_path = data_path
        self.is_train  = is_train
        if is_train:
            self.dataset_type = "train"
        else:
            self.dataset_type = "test"
        self.objects_path      = os.path.join(os.path.join(data_path, "dataset/augmentation"), self.dataset_type)
        self.config_path       = os.path.join(self.data_path, "config.yaml")
        try:
            with open(self.config_path, 'r') as config_data:
                self.config = yaml.load(config_data)
        except BaseException as e:
            sys.stderr.write(f'Unable to load YAML configuration file {self.config_path}\n')
            sys.stderr.write(traceback.format_exc())
            sys.exit("Error loading dataset")
        self.objects = {}
        self.classes = self.config['classes']
        self.background_classes = []
        if 'background_classes' in self.config and self.config['background_classes'] is not None:
            self.background_classes = self.config['background_classes']
        for class_label in set(self.classes) | set(self.background_classes) | {constants.background_label}:
            class_path = os.path.join(self.objects_path, class_label)
            self.objects[class_label] = [os.path.join(class_path, image_file) for image_file in system_utils.imagesInFolder(class_path)]
            random.shuffle(self.objects[class_label])
            if len(self.objects[class_label]) is 0:
                sys.stderr.write(f'Not enough images for class "{class_label}".')
                sys.exit(-1)
        self.length = functools.reduce(operator.add, [len(images) for (object, images) in self.objects.items()], 0)
        self.max_classes_per_image = 1
        self.max_objects_per_class = 1
        self.max_background_objects_per_image = 2
        self.probability_using_cache = 0.95
        self.probability_no_objects = 0.05
        self.remove_corrupted_files = True
        if 'data_sampling' in self.config:
            data_sampling_config = self.config['data_sampling']
            if 'probability_no_objects' in data_sampling_config:
                self.probability_no_objects = data_sampling_config['probability_no_objects']
            if 'remove_corrupted_files' in data_sampling_config:
                self.remove_corrupted_files = data_sampling_config['remove_corrupted_files']
            if 'max_classes_per_image' in data_sampling_config:
                self.max_classes_per_image = max(min(len(self.classIndexesExcludingBackground()), data_sampling_config['max_classes_per_image']), 1)
            if 'max_objects_per_class' in data_sampling_config:
                self.max_objects_per_class = data_sampling_config['max_objects_per_class']
            if 'max_background_objects_per_image' in data_sampling_config:
                self.max_background_objects_per_image = data_sampling_config['max_background_objects_per_image']
            if 'probability_using_cache' in data_sampling_config:
                self.probability_using_cache = float(data_sampling_config['probability_using_cache'])


    def classIndexesExcludingBackground(self):
        return list(range(0, len(self.classes)))
