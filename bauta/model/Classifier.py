from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import cv2
import numpy as np

from bauta.utils.CudaUtils import CudaUtils
from bauta.utils.ModelUtils import ModelUtils
from bauta.utils.ImageUtils import ImageUtils

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.convolutional_16_reducer = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(92, 15, 13, 1)        
        self.convolutional_16_merger = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(16, 16, 13, 1)
        self.classifier_1 = nn.Conv2d(16, 128, 32)
        self.classifier_2 = nn.Conv2d(128, 1, 1)
        ModelUtils.xavier(self.classifier_2)
        #logits for initial output near 0.01, useful as most of targets are backgrounds
        self.classifier_2.weight.data = self.classifier_2.weight.data.abs() * -4.0

    def forward(self, input):
        predicted_mask, embeddings = input
        embeddings = F.relu(self.convolutional_16_reducer(embeddings))
        embeddings = F.relu(self.convolutional_16_merger(torch.cat([embeddings, predicted_mask], 1))) # 1:16
        embeddings = F.relu(self.classifier_1(embeddings))
        predictions = F.sigmoid(self.classifier_2(embeddings))
        return predictions.squeeze(2).squeeze(2)