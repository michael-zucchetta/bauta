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
        self.logger = self.system_utils.getLogger(self)

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
        current_object = self.imageWithinInputDimensions(current_object)
        return random_class_index, current_object

    def subtractSubMaskFromMainMask(self, all_masks, sub_mask, object_index):
        all_masks[:, :, object_index : object_index + 1] = cv2.subtract(all_masks[:, :, object_index : object_index + 1], sub_mask[:,:]).reshape(sub_mask.shape)

    def addSubMaskFromMainMask(self, all_masks, sub_mask, object_index):
        all_masks[:, :, object_index : object_index + 1] = cv2.add(all_masks[:, :, object_index : object_index + 1], sub_mask[:,:]).reshape(sub_mask.shape)

    def generateAugmentedImage(self, index):
        random_number_of_objects = random.randint(0, self.config.max_objects_per_image_sample)
        validateRandomObject = lambda result:  result is not None
        class_indexes_and_objects = [self.system_utils.tryToRun(lambda : self.randomObject(index), validateRandomObject, constants.max_image_retrieval_attempts)
                                     for _ in range(random_number_of_objects)]
        background = self.system_utils.tryToRun(self.randomBackground, validateRandomObject, constants.max_image_retrieval_attempts)
        target_mask_all_classes, objects_in_image = self.environment.blankMasksAndObjectsInImage(self.config.classes)
        input_image = background
        target_mask_all_classes[:, :, constants.background_mask_index:constants.background_mask_index + 1] = 255
        masks_ordering = []
        for (class_index, class_object) in class_indexes_and_objects:
            objects_in_image[class_index] = 1
            resulting_object_mask = None
            distorted_class_object = self.image_distortions.applyTransformationsAndDistortions(class_object)
            input_image, object_mask = self.image_utils.pasteRGBAimageIntoRGBimage(distorted_class_object, input_image, 0, 0)
            self.addSubMaskFromMainMask(target_mask_all_classes, object_mask, class_index)
            # possibly overwrites previous pasted images, hence, for every previous class, it subtract the new mask to the previous ones
            for previous_class_index in masks_ordering:
                if class_index != previous_class_index:
                    self.subtractSubMaskFromMainMask(target_mask_all_classes, object_mask, previous_class_index)
            self.subtractSubMaskFromMainMask(target_mask_all_classes, object_mask, constants.background_mask_index)
            if not class_index in masks_ordering:
                masks_ordering.append(class_index)
        masks_ordering = [0] + masks_ordering
        if target_mask_all_classes[:, :, constants.background_mask_index:constants.background_mask_index + 1].sum() > 0:
            objects_in_image[constants.background_mask_index] = 1
        self.environment.storeSampleWithIndex(index, self.config.is_train, input_image, target_mask_all_classes, masks_ordering, self.config.classes)
        return input_image, target_mask_all_classes, objects_in_image

    def __getitem__(self, index, max_attempts=10):
        input_image, target_mask_all_classes, objects_in_image = None, None, None
        if np.random.uniform(0, 1, 1)[0] <= self.config.probability_using_cache:
            input_image, target_mask_all_classes, objects_in_image = self.environment.getSampleWithIndex(index, self.config.is_train, self.config.classes)

        if input_image is None or  target_mask_all_classes is None or objects_in_image is None:
            input_image, target_mask_all_classes, objects_in_image = self.generateAugmentedImage(index)
        input_image = transforms.ToTensor()(input_image)
        target_mask_all_classes = transforms.ToTensor()(target_mask_all_classes)
        return input_image, target_mask_all_classes, objects_in_image
