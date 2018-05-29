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
        self.embedding_reducer = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(448, 6, 7, 1)
        # self.embedding_reducer2 = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(112, 55, 7, 1)
        self.fully_connected_final = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(7, 1, 7, 1, False)
        ModelUtils.xavier(self.fully_connected_final)
        #logits for initial output near 0.98, useful as most of targets are foregrounds during refining
        self.fully_connected_final.weight.data = self.fully_connected_final.weight.data.abs() * +4.0

    def forward(self, input):
        image, embeddings, mask = input
        height_scale_factor = image.size()[2] / mask.size()[2]
        width_scale_factor = image.size()[3] / mask.size()[3]
        height_scale_increase = max(2, int(height_scale_factor / 2))
        width_scale_increase = max(2, int(width_scale_factor / 2))
        embeddings = F.relu(self.embedding_reducer(embeddings))
        # embeddings = F.relu(self.embedding_reducer2(embeddings))
        embeddings_and_mask = torch.cat([mask, embeddings], 1)
        embeddings_and_mask = F.upsample(embeddings_and_mask, scale_factor=(height_scale_increase, width_scale_increase), mode='bilinear')
        refined_mask = F.sigmoid(self.fully_connected_final(embeddings_and_mask))
        refined_mask = F.upsample(refined_mask, size=(image.size()[2], image.size()[3]), mode='bilinear')
        return refined_mask
