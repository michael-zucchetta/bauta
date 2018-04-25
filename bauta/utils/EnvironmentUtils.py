import cv2
import logging
import sys, os
import torch
import re
import subprocess
import traceback

from bauta.Constants import constants
from bauta.utils.ImageUtils import ImageUtils
from bauta.utils.SystemUtils import SystemUtils

class EnvironmentUtils():

    def __init__(self, data_path):
        self.data_path = data_path
        if not os.path.isdir(data_path):
            error_message = f"Data path '{data_path}' not found."
            #TODO: check all subfolders are in place
            raise Exception(error_message)
        self.models_path = os.path.join(self.data_path, "models/")
        self.objects_path = os.path.join(self.data_path, "dataset/augmentation/")
        self.dataset_path = os.path.join(self.data_path, "dataset/")
        self.best_model_file = "model.backup"
        self.logs_path = os.path.join(self.data_path, "logs")
        self.image_download_path = os.path.join(self.data_path, "image_download")
        self.system_utils = SystemUtils()
        self.image_utils = ImageUtils()

    def objectsFolder(self, class_name, is_train):
        return os.path.join(os.path.join(self.objects_path, constants.datasetType(is_train), class_name))

    def loadModelFromPath(self, path):
        return torch.load(path, map_location=lambda storage, loc: storage)

    def saveModel(self, mask_detector_model, name):
        path = os.path.join(self.models_path, name)
        torch.save(mask_detector_model.float(), path)

    def input_filename_path(self, index_path):
        index_filename_path = os.path.join(index_path, constants.dataset_item_filename)
        return index_filename_path

    def index_path(self, index, is_train, clean_dir=False):
        dataset_type = constants.datasetType(is_train)
        index_path = os.path.join(os.path.join(self.dataset_path, dataset_type), f"{index}")
        self.system_utils.makeDirIfNotExists(index_path)
        if clean_dir:
            self.system_utils.removeFilesFromDir(index_path)
        return index_path

    def blankMasksAndObjectsInImage(self, classes):
        target_masks = self.image_utils.blankImage(constants.input_width, constants.input_height, len(classes))
        objects_in_image = torch.FloatTensor(len(classes))
        objects_in_image.zero_()
        return target_masks, objects_in_image

    def _retrieveAlphaMasksAndObjects(self, alpha_mask_image_paths, classes, index_path):
        target_masks, objects_in_image = self.blankMasksAndObjectsInImage(classes)
        for alpha_image_path in alpha_mask_image_paths:
            splitted_alpha_file_path = re.sub(f'{constants.object_ext}$', '', alpha_image_path).split(constants.dataset_mask_prefix)
            (mask_order_index, class_name) = splitted_alpha_file_path
            class_index = classes.index(class_name)
            objects_in_image[class_index] = 1
            mask_class_image = cv2.imread(os.path.join(index_path, alpha_image_path), cv2.IMREAD_UNCHANGED)
            if mask_class_image is None:
                return None, None
            else:
                mask_class_image = mask_class_image.reshape(mask_class_image.shape[0], mask_class_image.shape[1], 1)
                target_masks[:, :, class_index : class_index + 1] = mask_class_image[:, :]
        return target_masks, objects_in_image

    def getSampleWithIndex(self, index, is_train, classes):
        index_path = self.index_path(index, is_train)
        input_filename_path = self.input_filename_path(index_path)
        alpha_mask_image_paths = self.system_utils.imagesInFolder(index_path, constants.dataset_mask_prefix_regex)
        if os.path.isfile(input_filename_path) and len(alpha_mask_image_paths) >= 2:
            input_image, target_masks, objects_in_image = None, None, None
            if len(alpha_mask_image_paths) > 0:
                input_image = cv2.imread(input_filename_path, cv2.IMREAD_COLOR)
                target_masks, objects_in_image = self._retrieveAlphaMasksAndObjects(alpha_mask_image_paths, classes, index_path)
                if target_masks is None or objects_in_image is None:
                     return None, None, None
            return input_image, target_masks, objects_in_image
        else:
             return None, None, None

    def storeSampleWithIndex(self, index, is_train, input_image, target_masks, masks_ordering, classes):
        index_path = self.index_path(index, is_train, clean_dir=True)
        input_filename_path = self.input_filename_path(index_path)
        for mask_order_index, class_index in enumerate(masks_ordering):
            class_name = classes[class_index]
            object_mask_filename = os.path.join(index_path, f'{mask_order_index}{constants.dataset_mask_prefix}{class_name}.png')
            cv2.imwrite(object_mask_filename, target_masks[:, :, class_index : class_index + 1], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cv2.imwrite(input_filename_path, input_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def loadModel(self, name):
        path = os.path.join(self.models_path, name)
        if os.path.isfile(path):
            return self.loadModelFromPath(path)
        else:
            sys.stderr.write(f"Model file not found in {path}.")
            return None
