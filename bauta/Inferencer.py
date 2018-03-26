import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from bauta.DressDetectorDataset import DressDetectorDataset
from bauta.InferenceUtils import InferenceUtils
import sys
from bauta.Environment import Environment
from bauta.ImageUtils import ImageUtils
import os, random, math
import click
import torch.nn.functional as F

class MaskDetectorInferencer():

    def __init__(self, file_name, show_results, save_result, result_folder, folder_name):
        self.file_name = file_name
        self.show_results = show_results
        self.save_result = save_result
        self.result_folder = result_folder
        self.folder_name = folder_name
        self.environment = Environment()
        self.inference_utils = InferenceUtils()
        self.image_utils = ImageUtils()

    def randomizeSaturationAndGain(masked_image):
        masked_image_bgr = masked_image[:,:,0:3]
        masked_image_hsv = cv2.cvtColor(masked_image_bgr,cv2.COLOR_BGR2HSV)
        saturation_gain = random.random() * 2.0
        masked_image_hsv[:,:,1] = masked_image_hsv[:,:,1] * saturation_gain
        masked_image_hsv[:,:,1] = ((masked_image_hsv[:,:,1] > 255) * 255) + ((masked_image_hsv[:,:,1] <= 255) * masked_image_hsv[:,:,1])
        value_gain = random.random() * 2.0
        masked_image_hsv[:,:,2] = masked_image_hsv[:,:,2] * value_gain
        masked_image_hsv[:,:,2] = ((masked_image_hsv[:,:,2] > 255) * 255) + ((masked_image_hsv[:,:,2] <= 255) * masked_image_hsv[:,:,2])
        masked_image_bgr = cv2.cvtColor(masked_image_hsv,cv2.COLOR_HSV2BGR)
        for color_channel in range(0, 3):
            masked_image[:,:,color_channel] = masked_image_bgr[:,:,color_channel]
        return masked_image

    def randomTranslation(masked_image):
        image_utils = ImageUtils()
        image_utils.composeImages(masked_image, masked_image, int(np.random.uniform(-10, 10)), int(np.random.uniform(-10, 10)), False, True)
        return masked_image

    def displayResults(bounding_boxes, probability_matrix, input_image):
        inference_utils = InferenceUtils()
        for bounding_box in bounding_boxes:
            print(bounding_box)
        images = inference_utils.generateIndividualDressesWithWhiteBackground(bounding_boxes, probability_matrix, input_image)
        folder_path = f"{result_folder}/{image_name}"
        if save_result:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            cv2.imwrite(f"{folder_path}/{image_name}", input_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        index = 1
        if len(images) > 0:
            print("Dress found in image")
            if len(images[0]) > 0:
                for image_index in range(len(images[0])):
                    if save_result:
                        cv2.imwrite(f"{folder_path}/{image_index}.png", images[0][image_index], [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    if show_results:
                        cv2.imshow(f'Predicted Dress {image_index}', images[0][image_index])
        else:
            print("No dresses found in image")

    def inferenceOnImage(self, mask_detector):
        image_utils = ImageUtils()
        if input_image is None:
            sys.stderr.write(f"Error reading image\n")
            sys.exit(-1)
        cv2.imshow(f'input_image', input_image)
        probability_matrix, input_image_bounding_box = inference_utils.dressBinaryMatrix(mask_detector, input_image)
        if show_results:
            cv2.imshow(f'input_image_bounding_box', input_image_bounding_box)
        image_scaled = input_image_bounding_box
        if show_results:
            cv2.imshow('probability_matrix without threshold', probability_matrix)
        probability_matrix = inference_utils.scaleTo255AndThresholdAboveStandardDeviation(probability_matrix, 1)
        if show_results:
            cv2.imshow('image_scaled', image_scaled)
            cv2.imshow('probability_matrix with threshold', probability_matrix )
        bounding_boxes, dress_location_matrix = inference_utils.dressSegments(probability_matrix, show_results)
        return bounding_boxes, probability_matrix, image_scaled

    def inferenceOnFile(self, mask_detector):
        with open(file_name, 'rb') as image_file:
            input_image = cv2.imread(file_name)
            bounding_boxes, probability_matrix, image_scaled = inferenceOnImage(mask_detector, input_image, show_results)
            displayResults(bounding_boxes, probability_matrix, image_scaled, show_results, save_result, result_folder, image_name)
            if show_results:
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def inference(self):
        mask_detector = self.environment.loadModel(environment.best_model_file)
        if not folder_name:
            inferenceOnFile(mask_detector, file_name, show_results, save_result, result_folder, file_name)
        else:
            file_names = list(filter(lambda filename: filename.endswith('.png') or filename.endswith('.jpg'), os.listdir(folder_name)))
            for file_name in file_names:
                full_file_name = os.path.join(folder_name, file_name)
                inferenceOnFile(mask_detector, full_file_name, show_results, save_result, result_folder, file_name)
