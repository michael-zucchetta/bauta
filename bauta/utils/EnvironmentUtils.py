import cv2
import logging
import sys, os
import torch
import re
import subprocess
import os
import traceback
import json

from bauta.Constants import constants
from bauta.utils.ImageUtils import ImageUtils
from bauta.utils.SystemUtils import SystemUtils

class EnvironmentUtils():

    def __init__(self, data_path, data_real_images_path=None):
        self.data_path = data_path
        self.data_real_train_images_path = f'{data_real_images_path}/train'
        self.data_real_test_images_path = f'{data_real_images_path}/test'
        if data_real_images_path is not None:
            # to be improved
            self.real_train_images_len = len(os.listdir(self.data_real_train_images_path))
            self.real_test_images_len = len(os.listdir(self.data_real_test_images_path))
        if not os.path.isdir(data_path):
            error_message = f'Data path "{data_path}" not found.'
            #TODO: check all subfolders are in place
            raise Exception(error_message)
        self.models_path = os.path.join(self.data_path, 'models/')
        self.objects_path = os.path.join(self.data_path, 'dataset/augmentation/')
        self.dataset_path = os.path.join(self.data_path, 'dataset/')
        self.best_model_file = 'model.backup'
        self.best_classification_model_file = 'classification.backup'
        self.logs_path = os.path.join(self.data_path, 'logs')
        self.image_download_path = os.path.join(self.data_path, 'image_download')
        self.system_utils = SystemUtils()
        self.image_utils = ImageUtils()

    def objectsFolder(self, class_name, is_train):
        return os.path.join(os.path.join(self.objects_path, constants.datasetType(is_train), class_name))

    def classesInDatasetFolder(self, is_train):
        base_path = os.path.join(self.objects_path, constants.datasetType(is_train))
        classes = [class_directory for class_directory in os.listdir( base_path )]
        class_paths = []
        for class_directory in classes:
            class_path = os.path.join(base_path, class_directory)
            if os.path.isdir(class_path):
               class_paths.append(class_path)
        return list(zip(classes, class_paths))

    def loadModelFromPath(self, path):
        return torch.load(path, map_location=lambda storage, loc: storage)

    def saveModel(self, mask_detector_model, name):
        path = os.path.join(self.models_path, name)
        torch.save(mask_detector_model.float(), path)

    def inputFilenamePath(self, index_path, use_real_image=False):
        index_filename_path = os.path.join(index_path, constants.dataset_input_filename)
        return index_filename_path

    def boundingBoxesFilenamePath(self, index_path):
        index_filename_path = os.path.join(index_path, constants.bounding_boxes_filename)
        return index_filename_path

    def indexPath(self, index, is_train, clean_dir=False, use_real_image=False):
        dataset_type = constants.datasetType(is_train)
        if use_real_image:
            # to be improved
            if is_train:
                index_path = os.path.join(self.data_real_train_images_path,f'{index % self.real_train_images_len}')
            else:
                index_path = os.path.join(self.data_real_test_images_path,f'{index % self.real_test_images_len}')
        else:
            index_path = os.path.join(os.path.join(self.dataset_path, dataset_type), f'{index}')
            self.system_utils.makeDirIfNotExists(index_path)
            if clean_dir:
                self.system_utils.removeFilesFromDir(index_path)
        return index_path

    def blankMasks(self, classes):
        return self.image_utils.blankImage(constants.input_width, constants.input_height, len(classes))

    def objectsInImage(self, classes):
        objects_in_image = torch.FloatTensor(len(classes))
        objects_in_image.zero_()
        return objects_in_image

    def _retrieveAlphaMasksAndObjects(self, alpha_mask_image_paths, classes, index_path):
        target_masks = self.blankMasks(classes)
        objects_in_image = self.objectsInImage(classes)
        for alpha_image_path in alpha_mask_image_paths:
            splitted_alpha_file_path = re.sub(f'{constants.object_ext}$', '', alpha_image_path).split(constants.dataset_mask_prefix)
            (_, class_name) = splitted_alpha_file_path
            if class_name in classes:
                class_index = classes.index(class_name)
                mask_class_image = cv2.imread(os.path.join(index_path, alpha_image_path), cv2.IMREAD_UNCHANGED)
                if mask_class_image is None:
                    return None
                else:
                    mask_class_image = mask_class_image.reshape(mask_class_image.shape[0], mask_class_image.shape[1], 1)
                    target_masks[:, :, class_index : class_index + 1] = mask_class_image[:, :]
            else:
                return None    
        return target_masks

    def getSampleWithIndex(self, index, is_train, classes, take_real_image=False):
        index_path = self.indexPath(index, is_train, use_real_image=take_real_image)
        input_filename_path = self.inputFilenamePath(index_path)
        alpha_mask_image_paths = self.system_utils.imagesInFolder(index_path, constants.dataset_mask_prefix_regex)
        if os.path.isfile(input_filename_path):
            bounding_boxes = torch.load(self.boundingBoxesFilenamePath(index_path))
            input_image, target_masks = None, None
            input_image = cv2.imread(input_filename_path, cv2.IMREAD_COLOR)
            target_masks = self._retrieveAlphaMasksAndObjects(alpha_mask_image_paths, classes, index_path)
            if target_masks is None:
                return None, None, None, None
            original_object_areas_path = self.originalObjectAreasPath(index_path)
            original_object_areas = torch.load(original_object_areas_path)
            return input_image, target_masks, original_object_areas, bounding_boxes
        else:
            return None, None, None, None

    def originalObjectAreasPath(self, index_path):
        original_object_areas_path = os.path.join(index_path, constants.dataset_original_object_areas_filename)
        return original_object_areas_path

    def storeSampleWithIndex(self, index, is_train, input_image, target_masks, original_object_areas, bounding_boxes, mask_class_indexes, classes):
        index_path = self.indexPath(index, is_train, clean_dir=True)
        for class_index in mask_class_indexes:
            class_name = classes[class_index]
            object_mask_filename = os.path.join(index_path, f'{constants.dataset_mask_prefix}{class_name}.png')
            cv2.imwrite(object_mask_filename, target_masks[:, :, class_index : class_index + 1], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        torch.save(bounding_boxes.cpu(), self.boundingBoxesFilenamePath(index_path))
        input_filename_path = self.inputFilenamePath(index_path)        
        cv2.imwrite(input_filename_path, input_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        original_object_areas_path = self.originalObjectAreasPath(index_path)
        torch.save(original_object_areas.float(), original_object_areas_path)

    def loadModel(self, name):
        path = os.path.join(self.models_path, name)
        if os.path.isfile(path):
            return self.loadModelFromPath(path)
        else:
            sys.stderr.write(f'Model file not found in {path}.')
            return None
