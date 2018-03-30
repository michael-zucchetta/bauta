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

    def datasetType(self, is_train):
        if is_train:
            return self.train_type
        else:
            return self.test_type
