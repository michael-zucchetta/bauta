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

    def cropRefinerDataset(connected_components_inside_IoU, embeddings, refiner_target_masks, refiner_input_image):
        image_utils = ImageUtils()
        connected_components = {}
        for batch_index in connected_components_inside_IoU.keys():
            for class_index in connected_components_inside_IoU[batch_index].keys():
                for predicted_connected_component in connected_components_inside_IoU[batch_index][class_index]:
                    mask = predicted_connected_component['mask']
                    bounding_box = predicted_connected_component['bounding_box']
                    cropped_embeddings = Variable(embeddings[batch_index:batch_index+1,:,bounding_box.top:bounding_box.bottom+1,bounding_box.left:bounding_box.right+1].data)
                    bounding_box_scaled = bounding_box.resize(embeddings.size()[3], embeddings.size()[2], refiner_input_image.size()[3], refiner_input_image.size()[2])
                    croped_input_image = Variable(refiner_input_image[batch_index:batch_index+1,:,bounding_box_scaled.top:bounding_box_scaled.bottom+1,bounding_box_scaled.left:bounding_box_scaled.right+1])
                    if refiner_target_masks is not None:
                        cropped_refiner_target_masks = refiner_target_masks[batch_index:batch_index+1,class_index:class_index+1,bounding_box_scaled.top:bounding_box_scaled.bottom+1,bounding_box_scaled.left:bounding_box_scaled.right+1]
                        cropped_refiner_target_masks = Variable(cropped_refiner_target_masks)
                    else:
                        cropped_refiner_target_masks = None
                    mask = Variable(torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).squeeze(4) / 255)
                    if not batch_index in connected_components:
                        connected_components[batch_index] = {}
                    if not class_index in connected_components[batch_index]:
                        connected_components[batch_index][class_index] = []
                    #cv2.imshow(f'mask {class_index}', image_utils.toNumpy(mask.data.squeeze(0)))
                    #cv2.imshow(f'croped_input_image {class_index}', image_utils.toNumpy(croped_input_image.data.squeeze(0)))
                    #cv2.waitKey(0)
                    object = { 'predicted_mask': mask,
                        'input_image': croped_input_image,
                        'target_mask': cropped_refiner_target_masks,
                        'embeddings': cropped_embeddings,
                        'bounding_box_scaled': bounding_box_scaled }
                    connected_components[batch_index][class_index].append(object)
        return connected_components

    def extractConnectedComponentsInMask(mask):
        if mask.sum() > 10.00:
            bounding_boxes, countour_areas = BoundingBox.fromOpenCVConnectedComponentsImage(mask, 25, 256)
            return bounding_boxes
        return []

    def extractConnectedComponents(masks):
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
                maximum = mask.max()
                if maximum > 0.60:
                    mask = (mask * 255) / maximum
                    mask = mask.astype(np.uint8)
                    bounding_boxes = InferenceUtils.extractConnectedComponentsInMask(mask)
                    for bounding_box in bounding_boxes:
                        if bounding_box.area >= 2:
                            mask_cropped = mask[bounding_box.top:bounding_box.bottom+1,bounding_box.left:bounding_box.right+1]
                            mask_density = (mask_cropped * maximum).mean()
                            if mask_density > 20:
                                if (bounding_box.area / mask_area) >= 0.05:
                                    if not batch_index in connected_components:
                                        connected_components[batch_index] = {}
                                    if not class_index in connected_components[batch_index]:
                                        connected_components[batch_index][class_index] = []
                                    object = { 'mask': mask_cropped, 'bounding_box' : bounding_box }
                                    #cv2.imshow(f'Mask {class_index}', cv2.resize(mask_cropped, (mask_cropped.shape[1] * 15, mask_cropped.shape[0] * 15)) )
                                    #cv2.waitKey(0)
                                    connected_components[batch_index][class_index].append(object)
        return connected_components
