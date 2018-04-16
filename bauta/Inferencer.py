import cv2
import numpy as np
import os, random, math
import sys
import click

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from bauta.DataAugmentationDataset import DataAugmentationDataset
from bauta.utils.InferenceUtils import InferenceUtils
from bauta.utils.EnvironmentUtils import EnvironmentUtils
from bauta.utils.ImageUtils import ImageUtils
from bauta.utils.SystemUtils import SystemUtils

class Inferencer():

    def __init__(self, data_path, visual_logging):
        self.visual_logging = visual_logging
        self.data_path = data_path
        self.inference_utils = InferenceUtils(data_path, visual_logging)

    def inferenceOnImage(self, mask_detector, input_image):
        masks_found = self.inference_utils.extractMasks(mask_detector, input_image)
        connected_components = self.inference_utils.extractConnectedComponents(masks_found)
        objects = self.inference_utils.extractObjects(connected_components, input_image)
        return objects
