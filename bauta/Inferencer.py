import cv2
import numpy as np
import os, random, math
import sys
import click

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, utils

from bauta.DataAugmentationDataset import DataAugmentationDataset
from bauta.utils.InferenceUtils import InferenceUtils
from bauta.utils.EnvironmentUtils import EnvironmentUtils
from bauta.utils.SystemUtils import SystemUtils
from bauta.utils.CudaUtils import CudaUtils
from bauta.utils.ImageUtils import ImageUtils
from bauta.DatasetConfiguration import DatasetConfiguration
from bauta.InferenceResult import InferenceResult

class Inferencer():

    def __init__(self, data_path, visual_logging, gpu):
        self.visual_logging = visual_logging
        self.gpu = gpu
        self.data_path = data_path
        self.cuda_utils = CudaUtils()
        self.image_utils = ImageUtils()
        self.config = DatasetConfiguration(False, data_path)


    def extractObjects(self, inference_results):
        new_inference_results = []
        for inference_result in inference_results:
            mask = self.image_utils.toNumpy(inference_result.mask.squeeze().data)
            original_image = self.image_utils.toNumpy(inference_result.image.squeeze().data)
            #cv2.imshow(f'Image', original_image)
            #cv2.imshow(f'Refined Mask', mask)
            #cv2.waitKey(0)
            mask = (mask * 255).astype(np.uint8)[:,:,0]
            blue_channel, green_channel, red_channel  = cv2.split(original_image)            
            image_cropped_with_alpha_channel = cv2.merge(((blue_channel * 255).astype(np.uint8), \
                (green_channel * 255).astype(np.uint8), (red_channel * 255).astype(np.uint8), mask))
            #print(inference_result.class_label, inference_result.bounding_box)
            #cv2.imshow(f'Image {inference_result.class_label}', image_cropped_with_alpha_channel)
            #cv2.imshow(f'Image Masked', image_cropped_with_alpha_channel)
            #cv2.waitKey(0)
            inference_result = InferenceResult(\
                class_label = inference_result.class_label,
                bounding_box  = inference_result.bounding_box,
                mask = inference_result.mask,
                image = image_cropped_with_alpha_channel)
            new_inference_results.append(inference_result)
        return new_inference_results

    def refine(self, refiner_dataset, model):
        inference_results = []
        for batch_index in refiner_dataset.keys():
            for class_index in refiner_dataset[batch_index].keys():
                for connected_component in refiner_dataset[batch_index][class_index]:
                    input_image, predicted_mask, embeddings= \
                        self.cuda_utils.cudify([connected_component['input_image'], connected_component['predicted_mask'], connected_component['embeddings']], self.gpu)
                    #cv2.imshow(f'Mask {self.config.classes[class_index]}', self.image_utils.toNumpy(predicted_mask.squeeze().data))
                    #cv2.waitKey(0)
                    predicted_refined_mask  = model.mask_refiners([input_image, embeddings, class_index, predicted_mask])
                    #cv2.imshow(f'Refined Mask {self.config.classes[class_index]}', self.image_utils.toNumpy(predicted_refined_mask.squeeze().data))
                    #cv2.waitKey(0)
                    inference_result = InferenceResult(\
                        class_label = self.config.classes[class_index],
                        bounding_box = connected_component['bounding_box_scaled'],
                        mask = predicted_refined_mask,
                        image = input_image)
                    inference_results.append(inference_result)
        return inference_results

    def inferenceOnImage(self, model, input_image):
        input_image, new_height, new_width = self.image_utils.paddingScale(input_image)
        input_image_preprocessed = Variable(DataAugmentationDataset.preprocessInputImage(input_image))
        input_image_preprocessed = self.cuda_utils.cudify([input_image_preprocessed.unsqueeze(0)], self.gpu)[0]
        print(input_image_preprocessed.size())
        predicted_masks, embeddings  = model.forward(input_image_preprocessed)
        connected_components_predicted = InferenceUtils.extractConnectedComponents(predicted_masks)
        refiner_dataset = \
            InferenceUtils.cropRefinerDataset(connected_components_predicted, embeddings, None, transforms.ToTensor()(input_image).unsqueeze(0))
        inference_results = self.refine(refiner_dataset, model)
        objects = self.extractObjects(inference_results)
        return objects
