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
from bauta.model.MaskRefiners import MaskRefiners
from bauta.model.MaskDetectors import MaskDetectors
from bauta.model.Classifiers import Classifiers

class Model(nn.Module):

    def __init__(self, classes, filter_banks, filter_size, scale):
        super(Model, self).__init__()
        self.scale = scale
        self.classes = classes
        self.backbone = Backbone()
        self.mask_detectors = MaskDetectors(classes, filter_banks, filter_size)
        self.mask_refiners = MaskRefiners(classes)
        self.classifiers = Classifiers(classes)

    def forward(self, input):
        cuda_utils = CudaUtils()
        embeddings_merged, embeddings_2, embeddings_4, embeddings_8 = self.backbone(input)
        predicted_masks, mask_embeddings = self.mask_detectors(embeddings_merged)
        return predicted_masks, mask_embeddings, embeddings_merged, embeddings_2, embeddings_4, embeddings_8
