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

    def refine(self, refiner_dataset, mask_refiner):
        inference_results = []
        for batch_index in refiner_dataset.keys():
            for class_index in refiner_dataset[batch_index].keys():
                for connected_component in refiner_dataset[batch_index][class_index]:
                    embeddings = self.cuda_utils.cudify(connected_component['embeddings'], self.gpu)
                    predicted_refined_mask = mask_refiner(embeddings)
                    if self.visual_logging:
                        cv2.imshow(f'Mask {self.config.classes[class_index]}', self.image_utils.toNumpy(predicted_refined_mask.squeeze().data))
                        cv2.waitKey(0)
                    #cv2.imshow(f'mask {class_index}', image_utils.toNumpy(mask.data.squeeze(0).squeeze(0)))
                    #cv2.imshow(f'Image', image_utils.toNumpy(input_image.data.squeeze(0).squeeze(0)))
                    #cv2.waitKey(0)
                    #cv2.imshow(f'Refined Mask {self.config.classes[class_index]}', self.image_utils.toNumpy(predicted_refined_mask.squeeze().data))
                    #cv2.waitKey(0)
                    inference_result = InferenceResult(\
                        class_label = self.config.classes[class_index],
                        bounding_box = connected_component['bounding_box'],
                        mask = predicted_refined_mask,
                        image = connected_component['input_image'])
                    inference_results.append(inference_result)
        return inference_results

    def inferenceOnImage(self, model, input_image):
        input_image, new_height, new_width = self.image_utils.paddingScale(input_image)
        if self.visual_logging:
            print(input_image.shape)
            cv2.imshow(f'Input padding scale', input_image)
            cv2.waitKey(0)
        input_image = Variable(transforms.ToTensor()(input_image))
        input_image = self.cuda_utils.cudify([input_image.unsqueeze(0)], self.gpu)[0]
        predicted_masks, embeddings_merged, embeddings_2, embeddings_4, embeddings_8 = model.forward(input_image)
        inference_utils = InferenceUtils(self.config, self.visual_logging)
        connected_components_predicted = inference_utils.extractConnectedComponents(predicted_masks)
        refiner_dataset = \
            inference_utils.cropRefinerDataset(connected_components_predicted, predicted_masks, embeddings_merged, embeddings_2, embeddings_4, embeddings_8, input_image)
        inference_results = self.refine(refiner_dataset, model.mask_refiner)
        objects = self.extractObjects(inference_results)
        return objects
