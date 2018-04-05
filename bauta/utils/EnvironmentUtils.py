import logging
import sys, os
import torch
import traceback
import subprocess

from bauta.Constants import Constants
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
        self.constants = Constants()
        self.system_utils = SystemUtils()

    def objectsFolder(self, class_name, is_train):
        return os.path.join(os.path.join(self.objects_path, self.constants.datasetType(is_train), class_name))

    def loadModelFromPath(self, path):
        return torch.load(path, map_location=lambda storage, loc: storage)

    def saveModel(self, mask_detector_model, name):
        path = os.path.join(self.models_path, name)
        torch.save(mask_detector_model.float(), path)

    def getSampleWithIndex(self, index, is_train):
        dataset_type = self.constants.datasetType(is_train)
        index_path = os.path.join(os.path.join(self.dataset_path, dataset_type), f"{index}")
        self.system_utils.makeDirIfNotExists(index_path)
        if os.path.isfile(os.path.join(index_path, 'input_image')) and \
            os.path.isfile(os.path.join(index_path, 'target_mask_all_classes')) and \
            os.path.isfile(os.path.join(index_path, 'objects_in_image')):
            input_image = torch.load(os.path.join(index_path, 'input_image'))
            target_mask_all_classes = torch.load(os.path.join(index_path, 'target_mask_all_classes'))
            objects_in_image = torch.load(os.path.join(index_path, 'objects_in_image'))
            return input_image, target_mask_all_classes, objects_in_image
        else:
            return None, None, None

    def storeSampleWithIndex(self, index, is_train, input_image, target_mask_all_classes, objects_in_image):
        dataset_type = self.constants.datasetType(is_train)
        index_path = os.path.join(os.path.join(self.dataset_path, dataset_type), f"{index}")
        self.system_utils.makeDirIfNotExists(index_path)
        torch.save(input_image, os.path.join(index_path, 'input_image'))
        torch.save(target_mask_all_classes, os.path.join(index_path, 'target_mask_all_classes'))
        torch.save(objects_in_image, os.path.join(index_path, 'objects_in_image'))

    def loadModel(self, name):
        path = os.path.join(self.models_path, name)
        if os.path.isfile(path):
            return self.loadModelFromPath(path)
        else:
            sys.stderr.write(f"Model file not found in {path}.")
            return None
