from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
from bauta.ImageInfo import ImageInfo
from bauta.utils.ModelUtils import ModelUtils
from bauta.model.MaskDetector import MaskDetector
from bauta.model.MaskRefiner import MaskRefiner

class MaskRefiners(nn.Module):

    def __init__(self, classes):
        super(MaskRefiners, self).__init__()
        self.classes = classes
        mask_refiners = []
        for mask_refiner_index in range(self.classes):
            mask_refiners.append(MaskRefiner())
        self.mask_refiners = nn.ModuleList(mask_refiners)

    def forward(self, input):
        outputs = []
        # predicted_masks, embeddings, embeddings_1_raw, embeddings_2_raw, embeddings_3_raw = input
        image, embeddings, class_index, predicted_masks = input
        #print(f'AAA {predicted_masks.size()}')
        #input_image_size = ImageInfo(input)
        for class_index in range(self.classes):
            output = self.mask_refiners[class_index](\
                [
                image,
                embeddings,
                predicted_masks[:,class_index:class_index+1,:,:]
                ])
            outputs.append(output)
        return torch.cat(outputs, 1)
