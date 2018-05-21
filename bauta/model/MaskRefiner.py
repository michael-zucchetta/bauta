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

class MaskRefiner(nn.Module):

    def __init__(self):
        super(MaskRefiner, self).__init__()
        self.embedding_reducer = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(448, 7, 13, 1)
        self.transformer = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(8, 8, 13, 1)
        self.merger = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(11, 8, 13, 1)
        self.merger_low_level = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(64+8, 8, 13, 1)
        self.fully_connected_1 = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(8, 8, 21, 1, False)
        self.fully_connected_2 = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(8, 1, 21, 1, False)
        ModelUtils.xavier(self.fully_connected_2)
        #logits for initial output near 0.98, useful as most of targets are foregrounds during refining
        self.fully_connected_2.weight.data = self.fully_connected_2.weight.data.abs() * +4.0

    def transform(self, embeddings):
        embeddings = F.upsample(embeddings, scale_factor=(2, 2), mode='bilinear')    
        embeddings = F.relu(self.transformer(embeddings))
        return embeddings

    def forward(self, input):
        embeding_low_raw, image, embeddings, mask = input
        embeddings = torch.cat([F.relu(self.embedding_reducer(embeddings)), mask], 1) # 1:32
        embeddings = self.transform(embeddings) # 1:16
        embeddings = self.transform(embeddings) # 1:8
        embeding_low = F.adaptive_max_pool2d(embeding_low_raw, output_size=(embeddings.size()[2], embeddings.size()[3])) # corrected 1:8
        embeding_low = F.relu(self.merger_low_level(torch.cat([embeding_low, embeddings], 1)))    
        embeding_low = F.upsample(embeding_low, size=(image.size()[2], image.size()[3]), mode='bilinear') # 1:1
        embeding_low = F.relu(self.fully_connected_1(embeding_low))
        refined_mask = F.sigmoid(self.fully_connected_2(embeding_low))
        return refined_mask
