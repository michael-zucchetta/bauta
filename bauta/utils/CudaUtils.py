import torch
import torch.nn as nn
from torch.autograd import Variable

class CudaUtils():

    def cudifyAsReference(self, tensors, reference_tensor):
        if isinstance(reference_tensor, (torch.cuda.FloatTensor)):
            return self.cudify(tensors, reference_tensor.get_device())
        else:
            return tensors

    def cudify(self, tensors, device_index):
        if torch.cuda.is_available():
            return [tensor.cuda(device_index) for tensor in tensors]
        else:
            return tensors

    def toVariable(self, tensors):
        return [Variable(tensor) for tensor in tensors]
