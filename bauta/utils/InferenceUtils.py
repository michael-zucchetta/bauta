from PIL import Image
import sys
import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from bauta.DatasetConfiguration import DatasetConfiguration
from bauta.BoundingBox import BoundingBox
from bauta.Constants import constants
from bauta.utils.ImageUtils import ImageUtils
from bauta.ImageInfo import ImageInfo
from bauta.InferenceResult import InferenceResult

class InferenceUtils():

    def __init__(self, threshold, config, visual_logging):
        self.config = config
        self.threshold = threshold
        self.visual_logging = visual_logging
        self.image_utils = ImageUtils()

    def cropRefinerDataset(self, connected_components_predicted, predicted_masks, input_image):
        image_utils = ImageUtils()
        connected_components = {}
        for batch_index in connected_components_predicted.keys():
            for class_index in connected_components_predicted[batch_index].keys():
                for predicted_connected_component in connected_components_predicted[batch_index][class_index]:
                    mask = predicted_connected_component['mask']
                    bounding_box_object = predicted_connected_component['bounding_box']
                    bounding_box_object_1 = bounding_box_object.resize(predicted_masks.size()[3], predicted_masks.size()[2], input_image.size()[3], input_image.size()[2])
                    if bounding_box_object.area > 1:
                        if not batch_index in connected_components:
                            connected_components[batch_index] = {}
                        if not class_index in connected_components[batch_index]:
                            connected_components[batch_index][class_index] = []
                        object = { 'predicted_mask': predicted_masks[:, class_index:class_index+1, :, :],
                            'bounding_box': bounding_box_object_1 }
                        connected_components[batch_index][class_index].append(object)
        return connected_components

    def extractConnectedComponentsInMask(self, batch_index, class_index, connected_components, mask):
        bounding_boxes, countour_areas = BoundingBox.fromOpenCVConnectedComponentsImage(mask, 25, 256)
        for bounding_box in bounding_boxes:
            if bounding_box.area >= 2:
                mask_cropped = mask[bounding_box.top:bounding_box.bottom+1,bounding_box.left:bounding_box.right+1]
                if self.visual_logging:
                    cv2.imshow(f'Mask {self.config.classes[class_index]}', cv2.resize(mask_cropped, (mask_cropped.shape[1], mask_cropped.shape[0])) )
                    cv2.waitKey(0)
                if not batch_index in connected_components:
                    connected_components[batch_index] = {}
                if not class_index in connected_components[batch_index]:
                    connected_components[batch_index][class_index] = []
                unique_bounding_box = True
                for connected_component_in_class in connected_components[batch_index][class_index]:
                    previous_bounding_box = connected_component_in_class['bounding_box']
                    if previous_bounding_box.areBoundsAproximatelySimilar(bounding_box):
                        unique_bounding_box = False
                        break
                if unique_bounding_box:
                    object = { 'mask': mask_cropped, 'bounding_box' : bounding_box }
                    connected_components[batch_index][class_index].append(object)

    def extractConnectedComponents(self, classifier_predictions, masks):
        connected_components = {}
        image_utils = ImageUtils()
        for batch_index in range(masks.size()[0]):
            for class_index in range(masks.size()[1]):
                probability = classifier_predictions[batch_index][class_index].data[0]
                if probability > self.threshold:
                    mask = image_utils.toNumpy(masks[batch_index][class_index].data)
                    if self.visual_logging:
                        cv2.imshow(f'Mask {self.config.classes[class_index]}', cv2.resize(mask, (mask.shape[1], mask.shape[0])) )
                        cv2.waitKey(0)
                    mask = ((mask * 255) / mask.max()).astype(np.uint8)
                    self.extractConnectedComponentsInMask(batch_index, class_index, connected_components, mask)                  
        return connected_components
