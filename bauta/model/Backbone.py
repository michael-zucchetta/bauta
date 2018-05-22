from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch

import cv2
import numpy as np

from bauta.utils.CudaUtils import CudaUtils
from bauta.utils.ModelUtils import ModelUtils

class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
        hidden_filter_banks = 32
        initial_filter_banks = 3
        filter_size = 7
        self.dilation1_1_encode  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(initial_filter_banks, int(hidden_filter_banks / 16), filter_size, 1)
        self.dilation1_2_encode  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(int(hidden_filter_banks / 16), int(hidden_filter_banks / 8), filter_size, 1)
        self.dilation1_3_encode  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(int(hidden_filter_banks / 8), int(hidden_filter_banks / 4), filter_size, 1)
        self.dilation1_4_encode  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(int(hidden_filter_banks / 4), int(hidden_filter_banks), filter_size, 1)

    def forward(self, input):
        input_scaled = nn.AvgPool2d(2, 2)(input)
        embeddings_1_raw = F.relu(self.dilation1_1_encode(input_scaled))
        embeddings_1 = nn.AvgPool2d(8, 8)(embeddings_1_raw)

        embeddings_1_raw_scaled = nn.AvgPool2d(2, 2)(embeddings_1_raw)
        embeddings_2_raw = F.relu(self.dilation1_2_encode(embeddings_1_raw_scaled))
        embeddings_2 = nn.AvgPool2d(4, 4)(embeddings_2_raw)

        embeddings_2_raw_scaled = nn.AvgPool2d(2, 2)(embeddings_2_raw)
        embeddings_3_raw = F.relu(self.dilation1_3_encode(embeddings_2_raw_scaled))
        embeddings_3 = nn.AvgPool2d(2, 2)(embeddings_3_raw)

        embeddings_3_raw_scaled = nn.AvgPool2d(2, 2)(embeddings_3_raw)
        embeddings_4_raw = F.relu(self.dilation1_4_encode(embeddings_3_raw_scaled))

        return torch.cat([embeddings_1, embeddings_2, embeddings_3, embeddings_4_raw], 1), embeddings_1_raw, embeddings_2_raw, embeddings_3_raw
