from collections import OrderedDict
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

from bauta.utils.BasicBackgroundRemover import BasicBackgroundRemover
from bauta.utils.ImageUtils import ImageUtils
from bauta.utils.EnvironmentUtils import EnvironmentUtils
from bauta.utils.ImageDistortions import ImageDistortions
from bauta.utils.SystemUtils import SystemUtils
from bauta.BoundingBox import BoundingBox
from bauta.DatasetConfiguration import DatasetConfiguration
from bauta.ImageInfo import ImageInfo
from bauta.Constants import constants

class DataAugmentationDataset(Dataset):

    def __init__(self, is_train, data_path, visual_logging=False, seed=None):
        super(DataAugmentationDataset, self).__init__()
        random.seed(seed)
        self.environment = EnvironmentUtils(data_path)
        self.system_utils = SystemUtils()
        self.visual_logging = visual_logging
        self.config = DatasetConfiguration(is_train, data_path)
        self.image_utils = ImageUtils()
        self.image_distortions = ImageDistortions()
        self.basic_background_remover = BasicBackgroundRemover()
        self.logger = self.system_utils.getLogger(self)
        self.maximum_area = constants.input_width * constants.input_height

    def __len__(self):
        return self.config.length

    def scale(self, image):
        return cv2.resize(image, (constants.input_width, constants.input_height))

    def coverInputDimensions(self, image):
        image_info = ImageInfo(image)
        if image_info.width < constants.input_width:
            image = cv2.resize(image, (constants.input_width, int(constants.input_width / image_info.aspect_ratio)))
        image_info = ImageInfo(image)
        if image_info.height < constants.input_height:
            image = cv2.resize(image, (int(constants.input_height * image_info.aspect_ratio), constants.input_height))
        return image

    def randomBackground(self):
        background_index = np.random.randint(len(self.config.objects[constants.background_label]), size=1)[0] % len(self.config.objects[constants.background_label])
        background = self.scale(cv2.imread(self.config.objects[constants.background_label][background_index], cv2.IMREAD_COLOR))
        background = self.coverInputDimensions(background)
        #TODO: central and/or random crop (not necessarily top-left as now)
        return background[:, 0:constants.input_width, 0:constants.input_height]

    def imageWithinInputDimensions(self, image):
        image_info = ImageInfo(image)
        if image_info.width > constants.input_width:
            image = cv2.resize(image, (constants.input_width, int(constants.input_width / image_info.aspect_ratio)))
        image_info = ImageInfo(image)
        if image_info.height > constants.input_height:
            image = cv2.resize(image, (int(constants.input_height * image_info.aspect_ratio), constants.input_height))
        return image

    def randomObject(self, index):
        random_class_index = random.choice(self.config.classIndexesExcludingBackground())
        random_class = self.config.classes[random_class_index]
        object_index = index % len(self.config.objects[random_class])
        current_object = cv2.imread(self.config.objects[random_class][object_index], cv2.IMREAD_UNCHANGED)
        current_object = self.basic_background_remover.removeFlatBackgroundFromRGB(current_object)
        current_object = self.imageWithinInputDimensions(current_object)
        return random_class_index, current_object

    def subtractSubMaskFromMainMask(self, all_masks, sub_mask, object_index):
        all_masks[:, :, object_index : object_index + 1] = cv2.subtract(all_masks[:, :, object_index : object_index + 1], sub_mask[:,:]).reshape(sub_mask.shape)

    def addSubMaskFromMainMask(self, all_masks, sub_mask, object_index):
        all_masks[:, :, object_index : object_index + 1] = cv2.add(all_masks[:, :, object_index : object_index + 1], sub_mask[:,:]).reshape(sub_mask.shape)

    def generateAugmentedImage(self, index, random_number_of_objects=None):
        if random_number_of_objects is None:
            random_number_of_objects = random.randint(0, self.config.max_objects_per_image_sample)
        validateRandomObject = lambda result:  result is not None
        class_indexes_and_objects = [self.system_utils.tryToRun(lambda : self.randomObject(index), validateRandomObject, constants.max_image_retrieval_attempts)
                                     for _ in range(random_number_of_objects)]
        input_image = self.system_utils.tryToRun(self.randomBackground, validateRandomObject, constants.max_image_retrieval_attempts)
        target_masks, objects_in_image = self.environment.blankMasksAndObjectsInImage(self.config.classes)
        original_object_areas = torch.zeros(len(self.config.classes))
        target_masks[:, :, constants.background_mask_index:constants.background_mask_index + 1] = 255
        classes_in_input = {constants.background_mask_index}
        for (class_index, class_object) in class_indexes_and_objects:
            distorted_class_object = self.image_distortions.distortImage(class_object)
            input_image, object_mask = self.image_utils.pasteRGBAimageIntoRGBimage(distorted_class_object, input_image, 0, 0)
            original_object_areas[class_index] =  original_object_areas[class_index] + object_mask.sum()
            self.addSubMaskFromMainMask(target_masks, object_mask, class_index)
            # removes the current image from the existing masks that overlap it
            for current_class_in_input in classes_in_input - {class_index}:
                self.subtractSubMaskFromMainMask(target_masks, object_mask, current_class_in_input)
            classes_in_input.add(class_index)
        for current_class_in_input in classes_in_input:
            object_area = target_masks[:, :, current_class_in_input:current_class_in_input + 1].sum()
            original_object_area = original_object_areas[current_class_in_input]
            if object_area / self.maximum_area > self.config.minimum_object_area_proportion_to_be_present \
             and original_object_area > self.config.minimum_object_area_proportion_to_be_present \
             and object_area / original_object_area > self.config.minimum_object_area_proportion_uncovered_to_be_present:
                objects_in_image[current_class_in_input] = 1.0
        self.environment.storeSampleWithIndex(index, self.config.is_train, input_image, target_masks, classes_in_input, self.config.classes)
        return input_image, target_masks, objects_in_image

    def __getitem__(self, index, max_attempts=10):
        input_image, target_masks, objects_in_image = None, None, None
        if np.random.uniform(0, 1, 1)[0] <= self.config.probability_using_cache:
            input_image, target_masks, objects_in_image = self.environment.getSampleWithIndex(index, self.config.is_train, self.config.classes)

        if input_image is None or  target_masks is None or objects_in_image is None:
            input_image, target_masks, objects_in_image = self.generateAugmentedImage(index)
        input_image = transforms.ToTensor()(input_image)
        target_masks = transforms.ToTensor()(target_masks)
        return input_image, target_masks, objects_in_image
