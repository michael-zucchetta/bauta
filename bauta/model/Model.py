from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch

import cv2
import numpy as np

from bauta.utils.CudaUtils import CudaUtils
from bauta.utils.ModelUtils import ModelUtils

from bauta.model.Backbone import Backbone
from bauta.model.MaskRefiner import MaskRefiner
from bauta.model.MaskDetectors import MaskDetectors

class Model(nn.Module):

    def __init__(self, classes, filter_banks, filter_size, scale):
        super(Model, self).__init__()
        self.scale = scale
        self.classes = classes
        self.backbone = Backbone()
        self.mask_detectors = MaskDetectors(classes, filter_banks, filter_size)
        self.mask_refiner = MaskRefiner()

    def forward(self, input):
        cuda_utils = CudaUtils()
        embeddings, embeddings_1_raw, embeddings_2_raw, embeddings_3_raw = self.backbone(input)
        masks = self.mask_detectors(embeddings)
        return masks, embeddings, embeddings_1_raw, embeddings_2_raw, embeddings_3_raw
