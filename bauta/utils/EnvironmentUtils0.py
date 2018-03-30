import logging
import sys, os
import torch
import traceback
import subprocess

from bauta.Constants import Constants

class EnvironmentUtils():

    def __init__(self, data_path):
        self.data_path = data_path
        if not os.path.isdir(data_path):
            error_message = f"Data path '{data_path}' not found."
            #TODO: check all subfolders are in place
            raise Exception(error_message)
        self.models_path = os.path.join(self.data_path, "models/")
        self.objects_path = os.path.join(self.data_path, "dataset/augmentation/")
        self.best_model_file = "model.backup"
        self.logs_path = os.path.join(self.data_path, "logs")
        self.image_download_path = os.path.join(self.data_path, "image_download")
        self.constants = Constants()

    def objectsFolder(self, class_name, is_train):
        return os.path.join(os.path.join(self.objects_path, self.constants.datasetType(is_train), class_name))

    def loadModelFromPath(self, path):
        return torch.load(path, map_location=lambda storage, loc: storage)

    def saveModel(self, mask_detector_model, name):
        path = os.path.join(self.models_path, name)
        torch.save(mask_detector_model.float(), path)

    def loadModel(self, name):
        path = os.path.join(self.models_path, name)
        if os.path.isfile(path):
            return self.loadModelFromPath(path)
        else:
            sys.stderr.write(f"Model file not found in {path}.")
            return None
