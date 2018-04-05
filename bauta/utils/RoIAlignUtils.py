import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from roi_align.roi_align import RoIAlign

class RoIAlignUtils():

    def createBoxesAndIndexes(bounding_boxes, object_found):
        indexes = []
        bounding_boxes_filtered = Variable(bounding_boxes.data.new(object_found.sum().data[0], 4))
        bounding_box_index = 0
        for input_index in range(object_found.size()[0]):
            for class_index in range(object_found.size()[1]):
                if object_found[input_index][class_index].data[0] == 1:
                    indexes.append(input_index)
                    bounding_boxes_filtered[bounding_box_index] = bounding_boxes[input_index][class_index]
                    bounding_box_index = bounding_box_index + 1
        indexes = Variable(torch.from_numpy(np.array(indexes)).int())
        if isinstance(bounding_boxes.data, (torch.cuda.FloatTensor)):
            indexes = indexes.cuda(bounding_boxes.data.get_device())
        return indexes, bounding_boxes_filtered

    def createBoxesAndIndexesOneToOne(bounding_boxes, object_found):
        indexes = []
        bounding_boxes_filtered = Variable(bounding_boxes.data.new(object_found.sum().data[0], 4))
        bounding_box_index = 0
        for index in range(object_found.size()[0]):
            if object_found[index].data[0] == 1:
                indexes.append(index)
                bounding_boxes_filtered[bounding_box_index] = bounding_boxes[index]
                bounding_box_index = bounding_box_index + 1
        indexes = Variable(torch.from_numpy(np.array(indexes)).int())
        if isinstance(bounding_boxes.data, (torch.cuda.FloatTensor)):
            indexes = indexes.cuda(bounding_boxes.data.get_device())
        return indexes, bounding_boxes_filtered

    def applyRoiAlignOneToOne(roi_align, input, bounding_boxes, object_found):
        input = input.view(input.size()[0] * input.size()[1], 1, input.size()[2], input.size()[3])
        bounding_boxes = bounding_boxes.view(-1, 4)
        object_found = object_found.view(-1)
        indexes, bounding_boxes_filtered = RoIAlignUtils.createBoxesAndIndexesOneToOne(bounding_boxes, object_found)
        output = roi_align(input, bounding_boxes_filtered, indexes)
        return output, bounding_boxes_filtered

    def applyRoiAlign(roi_align, input, bounding_boxes, object_found):
        indexes, bounding_boxes_filtered = RoIAlignUtils.createBoxesAndIndexes(bounding_boxes, object_found)
        output = roi_align(input, bounding_boxes_filtered, indexes)
        return output
