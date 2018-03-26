import math
import os, random, string
import sys
import yaml
from bauta.Environment import Environment
import traceback
import operator, functools

class DatasetConfiguration():

    def __init__(self, is_train, data_path):
        self.data_path = data_path
        if is_train:
            self.dataset_type = "train"
        else:
            self.dataset_type = "test"
        self.dataset_path      = os.path.join(data_path, "dataset/augmentation")
        self.objects_path      = os.path.join(os.path.join(self.dataset_path, "objects"), self.dataset_type)
        self.backgrounds_path  = os.path.join(os.path.join(self.dataset_path, "backgrounds"), self.dataset_type)
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
        for index, class_label in enumerate(self.classes):
            class_path = os.path.join(self.objects_path, class_label)
            self.objects[class_label] = [os.path.join(class_path, image_file) for image_file in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, image_file))]
        self.backgrounds = [os.path.join(self.backgrounds_path, image_file) for image_file in os.listdir(self.backgrounds_path) if os.path.isfile(os.path.join(self.backgrounds_path, image_file))]
        self.length = functools.reduce(operator.add, [len(class_label) for class_label in self.objects], 0)
