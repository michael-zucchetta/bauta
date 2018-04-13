import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import math
import cv2
from multiprocessing import Process, Queue
import os
import datetime
import numpy as np

from roi_align.roi_align import RoIAlign

from bauta.DataAugmentationDataset import DataAugmentationDataset
from bauta.BoundingBoxExtractor import BoundingBoxExtractor
from bauta.Model import Model
from bauta.DatasetConfiguration import DatasetConfiguration
from bauta.utils.EnvironmentUtils import EnvironmentUtils
from bauta.utils.ImageUtils import ImageUtils
from bauta.Constants import constants
from bauta.utils.CudaUtils import CudaUtils
from bauta.utils.RoIAlignUtils import RoIAlignUtils
from bauta.utils.SystemUtils import SystemUtils

class Trainer():

    def __init__(self, data_path, visual_logging, reset_model, num_epochs, batch_size, learning_rate, gpu,\
        loss_scaled_weight, loss_unscaled_weight, loss_objects_found_weight, only_masks):
        super(Trainer, self).__init__()
        self.only_masks = only_masks
        self.config = DatasetConfiguration(True, data_path)
        self.data_path = data_path
        self.visual_logging = visual_logging
        self.reset_model = reset_model
        self.loss_scaled_weight = loss_scaled_weight
        self.loss_unscaled_weight = loss_unscaled_weight
        self.loss_objects_found_weight = loss_objects_found_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gpu = gpu
        self.system_utils = SystemUtils()
        self.logger = self.system_utils.getLogger(self)
        self.image_utils = ImageUtils()
        self.environment = EnvironmentUtils(self.data_path)
        self.bounding_box_extractor = BoundingBoxExtractor(constants.input_width, constants.input_height, 8)
        self.mask_detector_model = self.loadModel()
        self.objects_found_loss = nn.L1Loss()
        self.optimizer = torch.optim.SGD([
                    {'params': self.mask_detector_model.parameters()}
                ], lr=self.learning_rate)
        self.cuda_utils = CudaUtils()
        self.mask_detector_model, self.bounding_box_extractor, self.objects_found_loss = \
            self.cuda_utils.cudify([self.mask_detector_model, self.bounding_box_extractor, self.objects_found_loss], self.gpu)

    def getWorkers(self):
        if self.visual_logging:
            return 0
        else:
            return 4

    def computeLoss(self, network_output, target_mask, target_objects_in_image):
        object_found, mask_scaled, mask, roi_align, bounding_boxes = network_output
        loss_objects_found = self.objects_found_loss(object_found.float(), target_objects_in_image)
        target_mask_scaled = nn.MaxPool2d(8, 8, return_indices=False)(target_mask)
        loss_scaled = self.focalLoss(mask_scaled, target_mask_scaled)
        loss = ( self.loss_scaled_weight * loss_scaled ) + \
            ( self.loss_objects_found_weight * loss_objects_found )
        if not self.only_masks:
            target_mask_filtered, target_bounding_boxes_filtered = RoIAlignUtils.applyRoiAlignOneToOne(roi_align, target_mask, bounding_boxes, object_found)
            loss_unscaled = self.focalLoss(mask, target_mask_filtered)
            loss = loss + self.loss_unscaled_weight * loss_unscaled
            if self.visual_logging:
                self.visualLoggingOutput(network_output, target_mask_scaled, target_mask_filtered)
            return loss, loss_scaled, loss_objects_found, loss_unscaled
        else:
            if self.visual_logging:
                self.visualLoggingOutput(network_output, target_mask_scaled)
            return loss, loss_scaled, loss_objects_found

    def testLoss(self):
        current_test_loss = None
        dataset_test = DataAugmentationDataset(False,  self.data_path, self.visual_logging)
        test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.getWorkers())
        average_current_test_loss = 0.0
        iterations = 0.0
        for i, (input_images, target_mask, target_objects_in_image) in enumerate(test_loader):
            input_images, target_mask, target_objects_in_image = self.cuda_utils.toVariable(self.cuda_utils.cudify([input_images, target_mask, target_objects_in_image], self.gpu))
            if self.visual_logging:
               self.visualLoggingDataset(input_images, target_mask)
            network_output = self.mask_detector_model.forward([input_images, self.only_masks])
            losses = self.computeLoss(network_output, target_mask, target_objects_in_image)
            average_current_test_loss = average_current_test_loss +  losses[0].data[0]
            iterations = iterations + 1.0
        average_current_test_loss = average_current_test_loss / iterations
        return average_current_test_loss

    def log(self, text):
        self.logger.info(f"{datetime.datetime.utcnow()} -- {text}")

    def testAndSaveIfImproved(self, best_test_loss):
        average_current_test_loss = self.testLoss()
        if average_current_test_loss < best_test_loss:
            self.log(f"Model Improved. Previous Best Test Loss {best_test_loss:{1}.{4}} | Current Best Test Loss  {average_current_test_loss:{1}.{4}} | Improvement Change: {(100.0 * (best_test_loss - average_current_test_loss) / average_current_test_loss):{1}.{4}} %")
            best_test_loss = average_current_test_loss
            self.log(f"Saving model...")
            self.environment.saveModel(self.mask_detector_model, self.environment.best_model_file)
            self.log(f"...model saved")
        else:
            self.log(f"Model did *NOT* Improve. Current Best Test Loss {best_test_loss:{1}.{4}} | Current Test Loss {average_current_test_loss:{1}.{4}} | Improvement Change: {(100.0 * (best_test_loss - average_current_test_loss) / average_current_test_loss):{1}.{4}} %")
        return best_test_loss

    def loadModel(self):
        mask_detector_model = None
        if not self.reset_model:
            mask_detector_model = self.environment.loadModel(self.environment.best_model_file)
        if self.reset_model or mask_detector_model is None:
            # incase no model is stored or in case the user wants to reset it
            mask_detector_model = Model(len(self.config.classes), 4, 7, 8)
        self.log(mask_detector_model)
        return mask_detector_model

    def focalLoss(self, mask, target_mask):
        mask_t = torch.mul(mask, target_mask) + torch.mul(mask - 1, target_mask - 1)
        focal_loss = -torch.mul(torch.log(mask_t + 1e-20), (-mask_t + 1).pow(2))
        return focal_loss.mean()

    def visualLoggingDataset(self, input_images, target_mask):
        for current_index in range(input_images.size()[0]):
            cv2.imshow(f'Trainer -- Input Image {current_index}', self.image_utils.toNumpy(input_images.data[current_index]))
            for current_class_index in range(target_mask[current_index].size()[0]):
                cv2.imshow(f'Trainer -- Target Mask {current_index} for class {self.config.classes[current_class_index]}', self.image_utils.toNumpy(target_mask.data[current_index][current_class_index]))

    def visualLoggingOutput(self, network_output, target_mask_scaled, target_mask_roi):
        object_found, mask_scaled, mask, roi_align, bounding_boxes = network_output
        current_found_index = 0
        for current_index in range(mask_scaled.size()[0]):
            for current_class in range(len(self.config.classes)):
                cv2.imshow(f'Target Mask Scaled {current_index} for "{self.config.classes[current_class]}".', self.image_utils.toNumpy(target_mask_scaled.data[current_index][current_class]))
                cv2.imshow(f'Output Mask Scaled {current_index} for "{self.config.classes[current_class]}".', self.image_utils.toNumpy(mask_scaled.data[current_index][current_class]))
                if object_found[current_index][current_class].data[0] == 1:
                    cv2.imshow(f'Output Found Mask {current_index} for "{self.config.classes[current_class]}".', self.image_utils.toNumpy(mask.data[current_found_index]))
                    current_found_index = current_found_index + 1
        cv2.waitKey(0)

    def logLoss(self, losses, epoch, train_dataset_index, dataset_train):
        if not self.only_masks:
            loss, loss_scaled, loss_objects_found, loss_unscaled = [loss.data[0] for loss in losses]
            self.log(f'Epoch [{epoch+1}/{self.num_epochs}] -- Iter [{train_dataset_index+1}/{math.ceil(len(dataset_train)/self.batch_size)}] -- Total Loss: {loss:{1}.{4}} -- Objects Found Loss: {loss_objects_found:{0}.{4}} -- Focal Loss Unscaled: {loss_unscaled:{1}.{4}} -- Focal Loss Scaled {loss_scaled:{1}.{4}}')
        else:
            loss, loss_scaled, loss_objects_found = [loss.data[0] for loss in losses]
            self.log(f'Epoch [{epoch+1}/{self.num_epochs}] -- Iter [{train_dataset_index+1}/{math.ceil(len(dataset_train)/self.batch_size)}] -- Total Loss: {loss:{1}.{4}} -- Objects Found Loss: {loss_objects_found:{0}.{4}}  -- Focal Loss Scaled {loss_scaled:{1}.{4}}')


    def train(self):
        best_test_loss = self.testLoss()
        self.log(f"Initial Test Loss {best_test_loss:{1}.{4}} ")
        for epoch in range(self.num_epochs):
            self.log(f"Epoch {epoch}")
            dataset_train = DataAugmentationDataset(True, self.data_path, self.visual_logging)
            train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=self.getWorkers())
            train_dataset_index = 0
            for i, (input_images, target_mask, target_objects_in_image) in enumerate(train_loader):
                sys.stdout.flush()
                input_images, target_mask, target_objects_in_image = self.cuda_utils.toVariable(self.cuda_utils.cudify([input_images, target_mask, target_objects_in_image], self.gpu))
                if self.visual_logging:
                    self.visualLoggingDataset(input_images, target_mask)
                network_output  = self.mask_detector_model.forward([input_images, self.only_masks])
                self.optimizer.zero_grad()
                losses = self.computeLoss(network_output, target_mask, target_objects_in_image)
                losses[0].backward()
                self.optimizer.step()
                self.logLoss(losses, epoch, train_dataset_index, dataset_train)
                train_dataset_index = train_dataset_index + 1
            self.environment.saveModel(self.mask_detector_model, os.path.join(self.environment.models_path, f"{(epoch + 1)}.backup"))
            best_test_loss = self.testAndSaveIfImproved(best_test_loss)
