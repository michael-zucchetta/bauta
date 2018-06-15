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

    def __init__(self, max_threshold, density_threshold, area_thresold, config, visual_logging):
        self.config = config
        self.max_threshold = max_threshold
        self.density_threshold = density_threshold 
        self.area_thresold = area_thresold
        self.visual_logging = visual_logging
        self.image_utils = ImageUtils()

    def cropRefinerDataset(self, connected_components_predicted, predicted_masks, embeddings_merged, embeddings_2, embeddings_4, embeddings_8, input_image):
        image_utils = ImageUtils()
        connected_components = {}
        for batch_index in connected_components_predicted.keys():
            for class_index in connected_components_predicted[batch_index].keys():
                for predicted_connected_component in connected_components_predicted[batch_index][class_index]:
                    mask = predicted_connected_component['mask']
                    bounding_box_object_16 = predicted_connected_component['bounding_box']
                    bounding_box_object_8 = bounding_box_object_16.resize(predicted_masks.size()[3], predicted_masks.size()[2], embeddings_8.size()[3], embeddings_8.size()[2])
                    bounding_box_object_4 = bounding_box_object_16.resize(predicted_masks.size()[3], predicted_masks.size()[2], embeddings_4.size()[3], embeddings_4.size()[2])
                    bounding_box_object_2 = bounding_box_object_16.resize(predicted_masks.size()[3], predicted_masks.size()[2], embeddings_2.size()[3], embeddings_2.size()[2])
                    bounding_box_object_1 = bounding_box_object_16.resize(predicted_masks.size()[3], predicted_masks.size()[2], input_image.size()[3], input_image.size()[2])
                    if bounding_box_object_16.area > 1:
                        predicted_masks_crop = bounding_box_object_16.cropTensor(predicted_masks, batch_index)
                        predicted_masks_crop = predicted_masks_crop[:, class_index:class_index + 1, :, :]
                        embeddings_crop      = bounding_box_object_16.cropTensor(embeddings_merged, batch_index)
                        embeddings_8_crop    = bounding_box_object_8.cropTensor(embeddings_8, batch_index)                    
                        embeddings_4_crop    = bounding_box_object_4.cropTensor(embeddings_4, batch_index)                        
                        embeddings_2_crop    = bounding_box_object_2.cropTensor(embeddings_2, batch_index)
                        input_image_crop     = bounding_box_object_1.cropTensor(input_image, batch_index)   
                        if self.visual_logging:
                            cv2.imshow(f'Embeddings "{self.config.classes[class_index]}".', self.image_utils.toNumpy(embeddings_merged[:,0:1,:,:].data.squeeze(0).squeeze(0)))
                            cv2.imshow(f'Embeddings 3 "{self.config.classes[class_index]}".', self.image_utils.toNumpy(embeddings_8[:,0:1,:,:].data.squeeze(0).squeeze(0)))
                            cv2.imshow(f'Embeddings 2 "{self.config.classes[class_index]}".', self.image_utils.toNumpy(embeddings_4[:,0:1,:,:].data.squeeze(0).squeeze(0)))
                            cv2.imshow(f'Embeddings 1 "{self.config.classes[class_index]}".', self.image_utils.toNumpy(embeddings_2[:,0:1,:,:].data.squeeze(0).squeeze(0)))
                            cv2.imshow(f'Crop Embeddings "{self.config.classes[class_index]}".', self.image_utils.toNumpy(embeddings_crop[:,0:1,:,:].data.squeeze(0).squeeze(0)))
                            cv2.imshow(f'Crop Embeddings 3"{self.config.classes[class_index]}".', self.image_utils.toNumpy(embeddings_8_crop[:,0:1,:,:].data.squeeze(0).squeeze(0)))
                            cv2.imshow(f'Crop Embeddings 2"{self.config.classes[class_index]}".', self.image_utils.toNumpy(embeddings_4_crop[:,0:1,:,:].data.squeeze(0).squeeze(0)))
                            cv2.imshow(f'Crop Embeddings 1"{self.config.classes[class_index]}".', self.image_utils.toNumpy(embeddings_2_crop[:,0:1,:,:].data.squeeze(0).squeeze(0)))
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        if not batch_index in connected_components:
                            connected_components[batch_index] = {}
                        if not class_index in connected_components[batch_index]:
                            connected_components[batch_index][class_index] = []
                        object = { 'predicted_mask': predicted_masks_crop,
                            'input_image': input_image_crop,
                            'embeddings': [input_image_crop.size(), predicted_masks_crop, embeddings_crop, embeddings_8_crop, embeddings_4_crop, embeddings_2_crop],
                            'bounding_box': bounding_box_object_1 }
                        connected_components[batch_index][class_index].append(object)
        return connected_components

    def extractConnectedComponentsInMask(mask):
        if mask.sum() > 10.00:
            bounding_boxes, countour_areas = BoundingBox.fromOpenCVConnectedComponentsImage(mask, 25, 256)
            return bounding_boxes
        return []

    def extractConnectedComponents(self, masks):
        connected_components = {}
        image_utils = ImageUtils()
        batches = masks.size()[0]
        classes = masks.size()[1]
        height = masks.size()[2]
        width = masks.size()[3]
        mask_area = (width + 1) * (height + 1)
        for batch_index in range(batches):
            for class_index in range(classes):
                mask = masks[batch_index][class_index]
                mask = image_utils.toNumpy(mask.data)
                if self.visual_logging:
                    cv2.imshow(f'Mask {self.config.classes[class_index]}', cv2.resize(mask, (mask.shape[1], mask.shape[0])) )
                    cv2.waitKey(0)
                maximum = mask.max()
                if maximum > self.max_threshold:
                    mask = (mask * 255) / maximum
                    mask = mask.astype(np.uint8)
                    bounding_boxes = InferenceUtils.extractConnectedComponentsInMask(mask) #TODO: threshold
                    for bounding_box in bounding_boxes:
                        if bounding_box.area >= 2:
                            mask_cropped = mask[bounding_box.top:bounding_box.bottom+1,bounding_box.left:bounding_box.right+1]
                            mask_density = (mask_cropped * maximum).mean()
                            if self.visual_logging:
                                cv2.imshow(f'Mask {self.config.classes[class_index]}', cv2.resize(mask_cropped, (mask_cropped.shape[1], mask_cropped.shape[0])) )
                                cv2.waitKey(0)
                            if mask_density > self.density_threshold:
                                if (bounding_box.area / mask_area) >= self.area_thresold:
                                    if not batch_index in connected_components:
                                        connected_components[batch_index] = {}
                                    if not class_index in connected_components[batch_index]:
                                        connected_components[batch_index][class_index] = []
                                    object = { 'mask': mask_cropped, 'bounding_box' : bounding_box }
                                    connected_components[batch_index][class_index].append(object)
        return connected_components
