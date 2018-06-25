import logging
import sys, os
import torch
import traceback
import subprocess

class Constants():

    def __init__(self):
        self.input_width  = 512
        self.input_height = 512
        self.background_label = 'background'
        self.train_type = 'train'
        self.test_type = 'test'
        # TODO be careful when we'll allow multiple image types
        self.object_ext = '.png'
        self.dataset_input_filename = f'input{self.object_ext}'
        self.dataset_original_object_areas_filename = 'original_object_areas'
        self.dataset_mask_prefix = r'_mask_'
        self.dataset_mask_prefix_regex = r'' + self.dataset_mask_prefix + '.+\.png$'
        self.max_image_retrieval_attempts = 2
        self.bounding_boxes_filename = 'bounding_boxes.json'
        self.max_threshold=0.4
        self.density_threshold=20
        self.area_thresold=0.05
        
    def datasetType(self, is_train):
        if is_train:
            return self.train_type
        else:
            return self.test_type

constants = Constants()
