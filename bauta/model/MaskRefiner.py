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
        hidden_filter_banks = 64
        filter_size = 7
        self.dilation1_4_decode  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(9, 1, filter_size, 1, False)
        self.dilation1_3_decode  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(17, int(hidden_filter_banks / 8), filter_size, 1)
        self.dilation1_2_decode  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(65, int(hidden_filter_banks / 4), filter_size, 1)
        self.dilation1_1_decode  = ModelUtils.createDilatedConvolutionPreservingSpatialDimensions(125, int(hidden_filter_banks), filter_size, 1)
        ModelUtils.xavier(self.dilation1_4_decode)
        #logits for initial output near 0.01, useful as most of targets are backgrounds
        self.dilation1_4_decode.weight.data = self.dilation1_4_decode.weight.data.abs() * -4.0

    def forward(self, input):
        input_image_size, predicted_masks_16, mask_embeddings_16, embeddings_16, embeddings_8, embeddings_4, embeddings_2 = input
        embeddings_16 = torch.cat([predicted_masks_16, mask_embeddings_16, embeddings_16], 1)
        embeddings_16 = F.relu(self.dilation1_1_decode(embeddings_16))

        embeddings_8 = torch.cat(\
            [F.upsample(embeddings_16, size=(embeddings_8.size()[2], embeddings_8.size()[3]), mode='nearest'),
            embeddings_8.sum(1).unsqueeze(1)], 
            1)
        embeddings_8 = F.relu(self.dilation1_2_decode(embeddings_8))

        embeddings_4 = torch.cat([\
            F.upsample(embeddings_8, size=(embeddings_4.size()[2], embeddings_4.size()[3]), mode='nearest'),
            embeddings_4.sum(1).unsqueeze(1)],
            1)
        embeddings_4 = F.relu(self.dilation1_3_decode(embeddings_4))

        embeddings_2 = torch.cat([\
            F.upsample(embeddings_4, size=(embeddings_2.size()[2], embeddings_2.size()[3]), mode='nearest'),\
            embeddings_2.sum(1).unsqueeze(1)], 1)
        mask = F.sigmoid(self.dilation1_4_decode(embeddings_2))
        return mask