import math
import os, random, string
import sys
import numpy as np
import cv2
import yaml
import traceback
import operator, functools

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from bauta.ImageUtils import ImageUtils
from bauta.Environment import Environment
from bauta.BoundingBox import BoundingBox
from bauta.DatasetConfiguration import DatasetConfiguration
from bauta.ImageInfo import ImageInfo

class DataAugmentationDataset(Dataset):

    def __init__(self, is_train, data_path, visual_logging=False, seed=None):
        super(DataAugmentationDataset, self).__init__()
        random.seed(seed)
        self.visual_logging = visual_logging
        self.enviroment = Environment(data_path)
        self.config = DatasetConfiguration(is_train, data_path)
        self.image_utils = ImageUtils()

    def __len__(self):
        return self.config.length

    def scale(self, image):
        return cv2.resize(image, (self.enviroment.input_width, self.enviroment.input_height))

    def coverInputDimensions(self, image):
        image_info = ImageInfo(image)
        if image_info.width < self.enviroment.input_width:
            image = cv2.resize(image, (self.enviroment.input_width, int(self.enviroment.input_width / image_info.aspect_ratio)))
        image_info = ImageInfo(image)
        if image_info.height < self.enviroment.input_height:
            image = cv2.resize(image, (int(self.enviroment.input_height * image_info.aspect_ratio), self.enviroment.input_height))
        return image

    def randomBackground(self):
        background_index = np.random.randint(len(self.config.backgrounds), size=1)[0]
        background = self.scale(cv2.imread(self.config.backgrounds[background_index], cv2.IMREAD_COLOR))
        background = self.coverInputDimensions(background)
        #TODO: central and/or random crop (not necessarily top-left as now)
        return background[:, 0:self.enviroment.input_width, 0:self.enviroment.input_height]

    def imageWithinInputDimensions(self, image):
        image_info = ImageInfo(image)
        if image_info.width > self.enviroment.input_width:
            image = cv2.resize(image, (self.enviroment.input_width, int(self.enviroment.input_width / image_info.aspect_ratio)))
        image_info = ImageInfo(image)
        if image_info.height > self.enviroment.input_height:
            image = cv2.resize(image, (int(self.enviroment.input_height * image_info.aspect_ratio), self.enviroment.input_height))
        return image

    def randomObject(self, index):
        random_class_index = np.random.randint(len(self.config.classes), size=1)[0] #TODO: always picks the same!
        random_class = self.config.classes[random_class_index]
        object_index = index % len(self.config.objects[random_class])
        current_object = cv2.imread(self.config.objects[random_class][object_index], cv2.IMREAD_UNCHANGED)
        current_object = self.imageWithinInputDimensions(current_object)
        return random_class_index, current_object

    def __getitem__(self, index, max_attempts=10):
        class_index, current_object = self.randomObject(index)
        objects_in_image = torch.FloatTensor(len(self.config.classes) + 1)
        objects_in_image.zero_()
        objects_in_image[class_index + 1] = 1
        background = self.randomBackground()
        input_image, target_mask = self.image_utils.pasteRGBAimageIntoRGBimage(current_object, background, 0, 0)
        target_mask_all_classes = self.image_utils.blankImage(self.enviroment.input_width, self.enviroment.input_height, len(self.config.classes) + 1)
        target_mask_all_classes[:, :, class_index + 1 : class_index + 2] = target_mask[:, :]
        target_mask_all_classes[:, :, self.enviroment.background_mask_index:self.enviroment.background_mask_index + 1] = 255 - target_mask[:,:]
        return transforms.ToTensor()(input_image), transforms.ToTensor()(target_mask_all_classes), objects_in_image
