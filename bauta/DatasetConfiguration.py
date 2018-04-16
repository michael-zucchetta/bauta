import math
import os, random, string
import sys
import yaml
import traceback
import operator, functools
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
        self.classes.insert(0, 'background')
        for index, class_label in enumerate(self.classes):
            class_path = os.path.join(self.objects_path, class_label)
            self.objects[class_label] = [os.path.join(class_path, image_file) for image_file in system_utils.imagesInFolder(class_path)]
        self.length = functools.reduce(operator.add, [len(images) for (object, images) in self.objects.items()], 0)

    def classIndexesExcludingBackground(self):
        return list(range(1, len(self.classes)))
