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

from bauta.utils.ImageUtils import ImageUtils
from bauta.utils.EnvironmentUtils import EnvironmentUtils
from bauta.BoundingBox import BoundingBox
from bauta.DatasetConfiguration import DatasetConfiguration
from bauta.ImageInfo import ImageInfo
from bauta.Constants import Constants

class DataAugmentationDataset(Dataset):

    def __init__(self, is_train, data_path, visual_logging=False, seed=None):
        super(DataAugmentationDataset, self).__init__()
        random.seed(seed)
        self.constants = Constants()
        self.visual_logging = visual_logging
        self.config = DatasetConfiguration(is_train, data_path)
        self.image_utils = ImageUtils()
        self.environment = EnvironmentUtils(data_path)

    def __len__(self):
        return self.config.length

    def scale(self, image):
        return cv2.resize(image, (self.constants.input_width, self.constants.input_height))

    def coverInputDimensions(self, image):
        image_info = ImageInfo(image)
        if image_info.width < self.constants.input_width:
            image = cv2.resize(image, (self.constants.input_width, int(self.constants.input_width / image_info.aspect_ratio)))
        image_info = ImageInfo(image)
        if image_info.height < self.constants.input_height:
            image = cv2.resize(image, (int(self.constants.input_height * image_info.aspect_ratio), self.constants.input_height))
        return image

    def randomBackground(self):
        background_index = np.random.randint(len(self.config.objects[self.constants.background_label]), size=1)[0] % len(self.config.objects[self.constants.background_label])
        background = self.scale(cv2.imread(self.config.objects[self.constants.background_label][background_index], cv2.IMREAD_COLOR))
        background = self.coverInputDimensions(background)
        #TODO: central and/or random crop (not necessarily top-left as now)
        return background[:, 0:self.constants.input_width, 0:self.constants.input_height]

    def imageWithinInputDimensions(self, image):
        image_info = ImageInfo(image)
        if image_info.width > self.constants.input_width:
            image = cv2.resize(image, (self.constants.input_width, int(self.constants.input_width / image_info.aspect_ratio)))
        image_info = ImageInfo(image)
        if image_info.height > self.constants.input_height:
            image = cv2.resize(image, (int(self.constants.input_height * image_info.aspect_ratio), self.constants.input_height))
        return image

    def randomObject(self, index):
        random_class_index = random.choice(self.config.classIndexesExcludingBackground())
        random_class = self.config.classes[random_class_index]
        object_index = index % len(self.config.objects[random_class])
        current_object = cv2.imread(self.config.objects[random_class][object_index], cv2.IMREAD_UNCHANGED)
        current_object = self.imageWithinInputDimensions(current_object)
        return random_class_index, current_object

    def __getitem__(self, index, max_attempts=10):
        input_image, target_mask_all_classes, objects_in_image = None, None, None
        if np.random.uniform(0, 1, 1)[0] <= 0.95:
            input_image, target_mask_all_classes, objects_in_image = self.environment.getSampleWithIndex(index, self.config.is_train)
        if input_image is None or  target_mask_all_classes is None or objects_in_image is None:
            class_index, current_object = self.randomObject(index)
            objects_in_image = torch.FloatTensor(len(self.config.classes))
            objects_in_image.zero_()
            objects_in_image[class_index] = 1
            background = self.randomBackground()
            input_image, target_mask = self.image_utils.pasteRGBAimageIntoRGBimage(current_object, background, 0, 0)
            target_mask_all_classes = self.image_utils.blankImage(self.constants.input_width, self.constants.input_height, len(self.config.classes))
            target_mask_all_classes[:, :, class_index : class_index + 1] = target_mask[:, :]
            target_mask_all_classes[:, :, self.constants.background_mask_index:self.constants.background_mask_index + 1] = 255 - target_mask[:,:]
            if target_mask_all_classes[:, :, self.constants.background_mask_index:self.constants.background_mask_index + 1].sum() > 0:
                objects_in_image[self.constants.background_mask_index] = 1
            input_image = transforms.ToTensor()(input_image)
            target_mask_all_classes = transforms.ToTensor()(target_mask_all_classes)
            self.environment.storeSampleWithIndex(index, self.config.is_train, input_image, target_mask_all_classes, objects_in_image)
        return input_image, target_mask_all_classes, objects_in_image
