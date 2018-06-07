from collections import OrderedDict
import math
import os, random, string
import sys
import numpy as np
import cv2
import yaml
import traceback
import operator, functools
import itertools
import json

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

    def __init__(self, is_train, data_path, visual_logging=False, max_samples=None, seed=None):
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
        if max_samples is None:
            self.length = self.config.length
        else:
            self.length = min(self.config.length, max_samples)

    def __len__(self):
        return self.length

    def cutImageBackground(self, image):
        image_info = ImageInfo(image)
        scale_x = constants.input_width / image_info.width
        scale_y = constants.input_height / image_info.height
        if scale_x > scale_y:
            image = cv2.resize(image, ( constants.input_width, int(image_info.height * scale_x)) )
        else:
            image = cv2.resize(image, ( int(image_info.width * scale_y), constants.input_height) )
        image_info = ImageInfo(image)

        x_seed = random.uniform(0, 1) * (image_info.width - constants.input_width)

        initial_width = int(0 + x_seed)
        final_width = int(x_seed + constants.input_width)

        y_seed = random.uniform(0, 1) * (image_info.height - constants.input_height)
        initial_height = int(0 + y_seed)
        final_height = int(y_seed + constants.input_height)
        return image[initial_height:final_height, initial_width:final_width, :]

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
        background_image = cv2.imread(self.config.objects[constants.background_label][background_index], cv2.IMREAD_COLOR)
        if background_image is not None and len(background_image.shape) == 3:
            background = self.cutImageBackground(background_image)
            background = self.coverInputDimensions(background)
            background = self.applyRandomBackgroundObjects(background)
            return background
        else:
            if self.config.remove_corrupted_files:
                self.logger.warning(f'Removing corrupted image {self.config.objects[constants.background_label][background_index]}')
                self.system_utils.rm(self.config.objects[constants.background_label][background_index])
            raise ValueError(f'Could not load background image {self.config.objects[constants.background_label][background_index]}')

    def imageWithinInputDimensions(self, image):
        image_info = ImageInfo(image)
        if image_info.width > constants.input_width:
            image = cv2.resize(image, (constants.input_width, int(constants.input_width / image_info.aspect_ratio)))
        image_info = ImageInfo(image)
        if image_info.height > constants.input_height:
            image = cv2.resize(image, (int(constants.input_height * image_info.aspect_ratio), constants.input_height))
        return image

    def objectInClass(self, index, class_index, background_classes=False):
        if not background_classes:
            class_label = self.config.classes[class_index]
        else:
            class_label = self.config.background_classes[class_index]
        object_index = index % len(self.config.objects[class_label])
        current_object = cv2.imread(self.config.objects[class_label][object_index], cv2.IMREAD_UNCHANGED)

        if current_object is not None:
            current_object = self.basic_background_remover.removeFlatBackgroundFromRGB(current_object)
            current_object = self.imageWithinInputDimensions(current_object)
            return current_object
        else:
            if self.config.remove_corrupted_files:
                self.logger.warning(f'Removing corrupted image {self.config.objects[class_label][object_index]}')
                self.system_utils.rm(self.config.objects[class_label][object_index])
            raise ValueError(f'Could not load object of class {class_label} in index {object_index}: {self.config.objects[class_label][object_index]}')

    def objectsInClass(self, index, class_index, count):
        class_indexes_and_objects = [(class_index, self.system_utils.tryToRun(lambda : self.objectInClass(index + current_object_in_class, class_index), \
            lambda result: result is not None, \
            constants.max_image_retrieval_attempts)) for current_object_in_class in range(count)]
        return list(itertools.chain(*class_indexes_and_objects))

    def subtractSubMaskFromMainMask(self, all_masks, sub_mask, object_index):
        all_masks[:, :, object_index : object_index + 1] = cv2.subtract(all_masks[:, :, object_index : object_index + 1], sub_mask[:,:]).reshape(sub_mask.shape)

    def addSubMaskToMainMask(self, all_masks, sub_mask, object_index):
        all_masks[:, :, object_index : object_index + 1] = cv2.add(all_masks[:, :, object_index : object_index + 1], sub_mask[:,:]).reshape(sub_mask.shape)

    def getRandomClassIndexToCount(self, random_number_of_objects):
        class_index_to_count = [0] * len(self.config.classes)
        if random.random() > self.config.probability_no_objects:
            if random_number_of_objects is None:
                random_number_of_objects = random.randint(1, min(self.config.max_classes_per_image, len(self.config.classes)) * self.config.max_objects_per_class)
            while sum(class_index_to_count) < random_number_of_objects:
                random_class_index = random.choice(list(range(0, len(self.config.classes))))
                if class_index_to_count[random_class_index] < self.config.max_objects_per_class:
                    class_index_to_count[random_class_index] = class_index_to_count[random_class_index] + 1
        return class_index_to_count

    def applyRandomBackgroundObjects(self, background):
        if len(self.config.background_classes) > 0 and self.config.max_background_objects_per_image > 0:
            size_random_background_objects = random.randint(1, self.config.max_background_objects_per_image)
            background_object_images = []
            for _ in range(size_random_background_objects):
                random_background_class_index = random.randint(0, len(self.config.background_classes) - 1)
                random_background_class = self.config.background_classes[random_background_class_index]
                random_object_index = random.randint(0, len(self.config.objects[random_background_class]) - 1)
                background_object_image = self.objectInClass(random_object_index, random_background_class_index, background_classes=True) 
                distorted_background_object = self.image_distortions.distortImage(background_object_image)
                background, _ = self.image_utils.pasteRGBAimageIntoRGBimage(distorted_background_object, background, 0, 0)
            return background
        else:
            return background

    def getObjectsInImage(self, target_masks, original_object_areas):
        objects_in_image = self.environment.objectsInImage(self.config.classes)
        for current_class_in_input in range(target_masks.shape[2]):
            object_area = target_masks[:, :, current_class_in_input:current_class_in_input + 1].sum() / 255
            if object_area > 0.0:
                original_object_area = original_object_areas[current_class_in_input] / 255
                if object_area / self.maximum_area > self.config.minimum_object_area_proportion_to_be_present \
                 and original_object_area > self.config.minimum_object_area_proportion_to_be_present \
                 and object_area / original_object_area > self.config.minimum_object_area_proportion_uncovered_to_be_present:
                    objects_in_image[current_class_in_input] = 1.0
        return objects_in_image

    def extractConnectedComponents(self, class_index, mask):
        connected_component = None
        image_utils = ImageUtils()
        image_info = ImageInfo(mask)
        if mask.sum() > 10.00:
            mask = (mask * 255).astype(np.uint8)
            _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            return torch.IntTensor([class_index, x, y, x + w, y + h])

    def generateAugmentedImage(self, index, random_number_of_objects=None):
        class_index_to_count = self.getRandomClassIndexToCount(random_number_of_objects)
        class_indexes_and_objects = [self.objectsInClass(index, class_index, count) for class_index, count in enumerate(class_index_to_count) if count > 0]
        random.shuffle(class_indexes_and_objects)
        input_image = self.system_utils.tryToRun(self.randomBackground, \
            lambda result: result is not None, \
            constants.max_image_retrieval_attempts)
        target_masks = self.environment.blankMasks(self.config.classes)
        original_object_areas = torch.zeros(len(self.config.classes))
        bounding_boxes = torch.zeros((self.config.max_classes_per_image * self.config.max_objects_per_class, 5)).int()
        classes_in_input = set()
        for object_index, (class_index, class_object) in enumerate(class_indexes_and_objects):
            distorted_class_object = self.image_distortions.distortImage(class_object)
            bounding_box = self.extractConnectedComponents(class_index, distorted_class_object[:,:,3:4])
            bounding_boxes[object_index:object_index+1, :] = bounding_box
            original_object_areas[class_index] =  original_object_areas[class_index] + distorted_class_object[:, :, 3].sum()
            input_image, object_mask = self.image_utils.pasteRGBAimageIntoRGBimage(distorted_class_object, input_image, 0, 0)
            self.addSubMaskToMainMask(target_masks, object_mask, class_index)
            classes_in_input.add(class_index)
        input_image = self.image_distortions.changeContrastAndBrightnessToImage(input_image)
        self.environment.storeSampleWithIndex(index, self.config.is_train, input_image, target_masks, original_object_areas, bounding_boxes, classes_in_input, self.config.classes)
        objects_in_image = self.getObjectsInImage(target_masks, original_object_areas)
        return input_image, target_masks, objects_in_image, bounding_boxes

    def isDataSampleConsistentWithDatasetConfiguration(self, input_image, target_masks, bounding_boxes):
        if input_image is not None and target_masks is not None and bounding_boxes is not None:
            if target_masks.shape[2] is len(self.config.classes):
                is_consistent = True
                for bounding_box_index in range(bounding_boxes.size()[0]):
                    is_consistent = is_consistent and bounding_boxes[bounding_box_index][0] < len(self.config.classes)
                return is_consistent
        return False

    def __getitem__(self, index, max_attempts=10):
        (input_image, target_masks, objects_in_image, bounding_boxes) = None, None, None, None
        if np.random.uniform(0, 1, 1)[0] <= self.config.probability_using_cache:
            try:
                input_image, target_masks, original_object_areas, bounding_boxes = self.environment.getSampleWithIndex(index, self.config.is_train, self.config.classes)
                if input_image is not None and target_masks is not None and original_object_areas is not None:
                    objects_in_image = self.getObjectsInImage(target_masks, original_object_areas)
            except BaseException as e:
                sys.stderr.write(traceback.format_exc())
        if not self.isDataSampleConsistentWithDatasetConfiguration(input_image, target_masks, bounding_boxes):
            index_path = self.environment.indexPath(index, self.config.is_train)
            self.system_utils.rm(index_path)
            current_attempt = 0
            while (not self.isDataSampleConsistentWithDatasetConfiguration(input_image, target_masks, bounding_boxes)) and current_attempt < max_attempts:
                try:
                    input_image, target_masks, objects_in_image, bounding_boxes = self.generateAugmentedImage(index)
                except BaseException as e:
                    sys.stderr.write(traceback.format_exc())
                    current_attempt = current_attempt + 1
                    index = index + 1
            if input_image is None or target_masks is None or objects_in_image is None:
                raise ValueError(f'There is a major problem during data sampling loading images. Please check error messages above.')
        input_image = transforms.ToTensor()(input_image)
        target_masks = transforms.ToTensor()(target_masks)
        return input_image, target_masks, objects_in_image, bounding_boxes
