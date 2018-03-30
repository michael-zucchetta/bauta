import logging
import sys, os
import torch
import traceback
import subprocess

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
        self.input_width  = 512
        self.input_height = 512
        self.background_mask_index = 0
        self.background_label = 'background'
        self.logs_path = os.path.join(self.data_path, "logs")
        self.image_download_path = os.path.join(self.data_path, "image_download")
        self.train_type = 'train'
        self.test_type = 'test'

    def datasetType(self, is_train):
        if is_train:
            return self.train_type
        else:
            return self.test_type

    def objectsFolder(self, class_name, is_train):
        return os.path.join(os.path.join(self.objects_path, self.datasetType(is_train), class_name))

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

    def getLogger(self, object_owner, level=logging.INFO):
        class_name = type(object_owner).__name__
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        handlers = [logging.StreamHandler()]
        logging.basicConfig(level=level, handlers=handlers)
        return logging.getLogger(class_name)
