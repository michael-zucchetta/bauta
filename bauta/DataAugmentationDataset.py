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
from bauta.utils.DatasetUtils import DatasetUtils
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
        self.system_utils = SystemUtils()
        self.visual_logging = visual_logging
        self.config = DatasetConfiguration(is_train, data_path)
        self.environment = EnvironmentUtils(data_path, self.config.data_real_images_path)
        self.dataset_utils = DatasetUtils(self.config)
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
        if final_height - initial_height == 1:
            initial_height = initial_height + 1
        if final_width - initial_width == 1:
            initial_width = initial_width + 1
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
        use_flat_background = random.uniform(0, 1) > 0.99
        if use_flat_background:
            background_image = np.ones( (constants.input_height, constants.input_width, 3), dtype=np.uint8) * 255
            use_white_background = bool(random.getrandbits(1))
            if not use_white_background:                
                # defalut to random background
                for channel in range(0, 3):
                    random_channel_value = random.uniform(0, 256)
                    background_image[:,:,channel] = random_channel_value
            return background_image
        else:
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
        width = None
        height = None
        if image_info.width > constants.input_width:
            width = constants.input_width
            height = int(constants.input_width / image_info.aspect_ratio)
        if image_info.height > constants.input_height:
            width = int(constants.input_height * image_info.aspect_ratio)
            height = constants.input_height
        if width is not None and height is not None:
          if width == constants.input_width + 1:
            width = constants.input_width
          if height == constants.input_height + 1:
            height = constants.input_height
          image = cv2.resize(image, (width, height))
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

    def generateAugmentedImage(self, index, random_number_of_objects=None):
        class_index_to_count = self.getRandomClassIndexToCount(random_number_of_objects)
        class_indexes_and_objects = [self.objectsInClass(index, class_index, count) for class_index, count in enumerate(class_index_to_count) if count > 0]
        random.shuffle(class_indexes_and_objects)
        input_image = self.system_utils.tryToRun(self.randomBackground, \
            lambda result: result is not None, \
            constants.max_image_retrieval_attempts)
        image_info = ImageInfo(input_image)
        target_masks = self.environment.blankMasks(self.config.classes)
        original_object_areas = torch.zeros(len(self.config.classes))
        #  bounding_boxes = torch.zeros((self.config.max_classes_per_image * self.config.max_objects_per_class, 5)).int()
        bounding_boxes = torch.zeros((50, 5)).int() # to be improved
        classes_in_input = set()
        for object_index, (class_index, class_object) in enumerate(class_indexes_and_objects):
            if self.config.classes[class_index] in self.config.special_composition_classes:
              composer = self.config.special_composition_classes[self.config.classes[class_index]]
              composed_image, only_item_composition, hue = composer.compose(class_object)
              homography_matrix = self.image_distortions.getHomographyMatrix(ImageInfo(composed_image))
              distorted_class_object = self.image_distortions.distortImage(composed_image, homography_matrix)
              extra_class_object = self.image_distortions.distortImage(only_item_composition, homography_matrix)
              extra_class_mask = self.image_utils.getAlphaChannel(extra_class_object).reshape(image_info.width, image_info.height, 1)
            else:
              distorted_class_object = self.image_distortions.distortImage(class_object)
            bounding_box = self.dataset_utils.extractConnectedComponents(class_index, distorted_class_object[:,:,3:4])
            bounding_boxes[object_index:object_index+1, :] = bounding_box
            original_object_areas[class_index] =  original_object_areas[class_index] + distorted_class_object[:, :, 3].sum()
            input_image, object_mask = self.image_utils.pasteRGBAimageIntoRGBimage(distorted_class_object, input_image, 0, 0)
            if self.config.classes[class_index] in self.config.special_composition_classes:
                object_mask = extra_class_mask
            self.addSubMaskToMainMask(target_masks, object_mask, class_index)
            # removes the current image from the existing masks that overlap it
            for current_class_in_input in classes_in_input - {class_index}:
                self.subtractSubMaskFromMainMask(target_masks, object_mask, current_class_in_input)
            classes_in_input.add(class_index)
        if self.visual_logging:
            cv2.imshow(f'Before Distortion', input_image)
        input_image = self.image_distortions.changeContrastAndBrightnessToImage(input_image)
        if self.visual_logging:
            cv2.imshow(f'After Distortion', input_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        self.environment.storeSampleWithIndex(index, self.config.is_train, input_image, target_masks, original_object_areas, bounding_boxes, classes_in_input, self.config.classes)
        return input_image, target_masks, bounding_boxes

    def getAndDistortRealImage(self, index):
        index_path_real, index_path = self.environment.indexPathReal(index, self.config.is_train, clean_dir=True)
        input_image = cv2.imread(self.environment.inputFilenamePath(index_path_real), cv2.IMREAD_COLOR)
        target_masks, class_indexes = self.environment.retrieveMasks(index_path_real, self.config.classes)
        image_info = ImageInfo(input_image)
        if np.random.uniform(0, 1) < 0.5:
          background = self.randomBackground()
          homography_matrix = self.image_distortions.getHomographyMatrix(ImageInfo(input_image))
          input_image = self.image_distortions.distortImage(input_image, homography_matrix)
          blank_mask = self.image_utils.blankImage(image_info.width, image_info.height, 1)[:, :, 0]
          blank_mask[:] = 255
          input_image, _ = self.image_utils.pasteRGBAimageIntoRGBimage(input_image, background, 0, 0, mask_image=blank_mask)

          for class_index in class_indexes:
            target_masks[:, :, class_index : class_index + 1] = \
                self.image_distortions.distortImage(target_masks[:, :, class_index : class_index + 1], homography_matrix).reshape(constants.input_width, constants.input_height, 1)
          self.environment.writeMaskFiles(index_path, target_masks, class_indexes, self.config.classes)
          self.environment.writeInputFile(index_path, input_image)
        bounding_boxes, original_object_areas = self.dataset_utils.createBoundingBoxesAndObjectAreas(index_path)
        return input_image, target_masks, bounding_boxes

    def isDataSampleConsistentWithDatasetConfiguration(self, input_image, target_masks, bounding_boxes):
        if input_image is not None and target_masks is not None and bounding_boxes is not None:
            if target_masks.shape[2] is len(self.config.classes):
                is_consistent = True
                for bounding_box_index in range(bounding_boxes.size()[0]):
                    is_consistent = is_consistent and bounding_boxes[bounding_box_index][0] < len(self.config.classes)
                return is_consistent
        return False

    def __getitem__(self, index, max_attempts=10):
        (input_image, target_masks, bounding_boxes) = None, None, None
        if np.random.uniform(0, 1, 1)[0] <= self.config.probability_using_cache:
            try:
                input_image, target_masks, original_object_areas, bounding_boxes = self.environment.getSampleWithIndex(index, self.config.is_train, self.config.classes)
            except BaseException as e:
                sys.stderr.write(traceback.format_exc())
        if not self.isDataSampleConsistentWithDatasetConfiguration(input_image, target_masks, bounding_boxes):
            index_path = self.environment.indexPath(index, self.config.is_train)
            self.system_utils.rm(index_path)
            use_real_images = np.random.uniform(0, 1, 1)[0] <= self.config.probability_using_real_images and self.config.real_images_available
            current_attempt = 0
            while (not self.isDataSampleConsistentWithDatasetConfiguration(input_image, target_masks, bounding_boxes)) and current_attempt < max_attempts:
                try:
                    if not use_real_images:
                      input_image, target_masks, bounding_boxes = self.generateAugmentedImage(index)
                    else:
                      input_image, target_masks, bounding_boxes = self.getAndDistortRealImage(index)
                except BaseException as e:
                    sys.stderr.write(traceback.format_exc())
                    current_attempt = current_attempt + 1
                    index = index + 1
            if input_image is None or target_masks is None:
                raise ValueError(f'There is a major problem during data sampling loading images. Please check error messages above.')
        input_image = transforms.ToTensor()(input_image)
        target_masks = transforms.ToTensor()(target_masks)
        return input_image, target_masks, bounding_boxes
