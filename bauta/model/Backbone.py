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
        hidden_filter_banks = 64
        filter_size = 7
        self.dilation1_1_encode  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(3, int(hidden_filter_banks / 16), filter_size, 1)
        self.dilation1_2_encode  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(int(hidden_filter_banks / 16), int(hidden_filter_banks / 8), filter_size, 1)
        self.dilation1_3_encode  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(int(hidden_filter_banks / 8), int(hidden_filter_banks / 4), filter_size, 1)
        self.dilation1_4_encode  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(int(hidden_filter_banks / 4), int(hidden_filter_banks), filter_size, 1)


    def mergedEmbeddings(self, embeddings_2, embeddings_4, embeddings_8, embeddings_16):
        return  torch.cat([ \
                    nn.AvgPool2d(8, 8)(embeddings_2),
                    nn.AvgPool2d(4, 4)(embeddings_4), 
                    nn.AvgPool2d(2, 2)(embeddings_8), 
                    embeddings_16], 
                1)

    def forward(self, input):
        input_scaled = nn.AvgPool2d(2, 2)(input)
        embeddings_2 = F.relu(self.dilation1_1_encode(input_scaled))

        embeddings_2_scaled = nn.AvgPool2d(2, 2)(embeddings_2)
        embeddings_4 = F.relu(self.dilation1_2_encode(embeddings_2_scaled))

        embeddings_4_scaled = nn.AvgPool2d(2, 2)(embeddings_4)
        embeddings_8 = F.relu(self.dilation1_3_encode(embeddings_4_scaled))

        embeddings_8_scaled = nn.AvgPool2d(2, 2)(embeddings_8)
        embeddings_16 = F.relu(self.dilation1_4_encode(embeddings_8_scaled))

        return  self.mergedEmbeddings(embeddings_2, embeddings_4, embeddings_8, embeddings_16), \
            embeddings_2, embeddings_4, embeddings_8
