import sys, os
import torch
import traceback
import subprocess

class Environment():
    """Util functions around enviroment variables"""

    def __init__(self, data_path):
        self.data_path = data_path
        if not os.path.isdir(data_path):
            error_message = f"Data path '{data_path}' not found."
            #TODO: check all subfolders are in place
            raise Exception(error_message)
        self.backgrounds_path = os.path.join(self.data_path, "datasets/images/backgrounds/")
        self.models_path = os.path.join(self.data_path, "models/")
        self.objects_path = os.path.join(self.data_path, "datasets/images/objects/")
        self.best_model_file = "model.backup"
        self.input_width  = 512
        self.input_height = 512
        self.background_mask_index = 0
        self.augmentation_cache_path = os.path.join(self.data_path, "datasets/images/augment")

    def loadModelFromPath(self, path ):
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

    def tryToRun(self, method, validate_result, max_attempts=10):
        current_attempt = 0
        successful_run = False
        result = None
        while not successful_run:
            try:
                result = method()
                successful_run = validate_result(result)
            except BaseException as e:
                sys.stderr.write(traceback.format_exc())
                current_attempt += 1
                sys.stderr.write(f"Error, trying again: {e}")
                if(current_attempt > max_attempts):
                    raise ValueError('It is not possible run method.', e)
        return result
