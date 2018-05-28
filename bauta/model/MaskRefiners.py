from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
from bauta.utils.ModelUtils import ModelUtils
from bauta.model.MaskRefiner import MaskRefiner

class MaskRefiners(nn.Module):

    def __init__(self, classes):
        super(MaskRefiners, self).__init__()
        self.classes = classes
        mask_refiners = []
        for mask_refiner_index in range(self.classes):
            mask_refiners.append(MaskRefiner())
        self.mask_refiners = nn.ModuleList(mask_refiners)

    def forward(self, embeddings):
        outputs = []
        input_image_size, predicted_masks, embeddings_merged, embeddings_2, embeddings_4, embeddings_8 = embeddings
        for class_index in range(self.classes):
            output = self.mask_refiners[class_index](\
                [
                input_image_size, 
                predicted_masks[:,class_index:class_index+1,:,:], 
                embeddings_merged, 
                embeddings_8, 
                embeddings_4, 
                embeddings_2])
            outputs.append(output)
        return torch.cat(outputs, 1)
