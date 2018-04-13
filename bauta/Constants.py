import logging
import sys, os
import torch
import traceback
import subprocess

class Constants():

    def __init__(self):
        self.input_width  = 512
        self.input_height = 512
        self.background_mask_index = 0
        self.background_label = 'background'
        self.train_type = 'train'
        self.test_type = 'test'
        # TODO be careful when we'll allow multiple image types
        self.object_ext = '.png'
        self.dataset_item_filename = f'input{self.object_ext}'
        self.dataset_mask_prefix = r'_mask_'
        self.dataset_mask_prefix_regex = r'[0-9]+' + self.dataset_mask_prefix + '.+\.png$'
        # TODO: add this as a parameter
        self.max_objects_per_image = 10
        # TODO: add this as a parameter
        self.probability_checking_cache = 0.95
        self.max_image_retrieval_attempts = 5

    def datasetType(self, is_train):
        if is_train:
            return self.train_type
        else:
            return self.test_type

constants = Constants()
