from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
from bauta.utils.ModelUtils import ModelUtils
from bauta.model.Classifier import Classifier

class Classifiers(nn.Module):

    def __init__(self, classes):
        super(Classifiers, self).__init__()
        self.classes = classes
        classifiers = []
        for classifier_index in range(self.classes):
            classifiers.append(Classifier())
        self.classifiers = nn.ModuleList(classifiers)

    def forward(self, input):
        outputs = []
        predicted_masks, embeddings = input
        for class_index in range(self.classes):
            output = self.classifiers[class_index](\
                [
                predicted_masks[:,class_index:class_index+1,:,:], 
                embeddings])
            outputs.append(output)
        return torch.cat(outputs, 1)