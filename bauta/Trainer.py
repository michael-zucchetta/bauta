
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
from bauta.Environment import Environment
from bauta.Model import Model
from bauta.DatasetConfiguration import DatasetConfiguration

class Trainer():


    def __init__(self, data_path, visual_logging, reset_model, num_epochs, batch_size, learning_rate, gpu, use_bounding_box):
        super(Trainer, self).__init__()
        self.config = DatasetConfiguration(True, data_path)
        self.data_path = data_path
        self.visual_logging = visual_logging
        self.reset_model = reset_model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gpu = gpu
        self.use_bounding_box = use_bounding_box
        self.environment = Environment(self.data_path)

    def getWorkers(self):
        if self.visual_logging:
            return 0
        else:
            return 4

    def testLoss(self, mask_detector_model):
        bounding_box_criterion_unscaled = nn.L1Loss()
        current_test_loss = None
        dataset_test = DataAugmentationDataset(False,  self.data_path, self.visual_logging)
        test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.getWorkers())
        average_current_test_loss = 0.0
        iterations = 0.0
        for i, (images, target_mask) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = images.cuda(self.gpu)
                target_mask = target_mask.cuda(self.gpu)
            images = Variable(images)
            target_mask = Variable(target_mask)
            mask_scaled, mask, roi_align_scaled, roi_align, bounding_boxes, bounding_boxes_scaled, boxes_index = mask_detector_model.forward([images, self.use_bounding_box])
            target_mask = roi_align(target_mask, bounding_boxes, boxes_index)
            loss_unscaled = bounding_box_criterion_unscaled(mask, target_mask)
            average_current_test_loss = average_current_test_loss +  loss_unscaled.data[0]
            iterations = iterations + 1.0
        average_current_test_loss = average_current_test_loss / iterations
        return average_current_test_loss

    def log(self, text):
        print(f"{datetime.datetime.utcnow()} -- MaskDetectorTrain ==> {text}")

    def testAndSaveIfImproved(self, best_test_loss, mask_detector_model):
        average_current_test_loss = self.testLoss(mask_detector_model)
        if average_current_test_loss < best_test_loss:
            self.log(f"Model Improved. Previous Best Test Loss {best_test_loss:{1}.{4}} | Current Best Test Loss  {average_current_test_loss:{1}.{4}} | Improvement Change: {(100.0 * (best_test_loss - average_current_test_loss) / average_current_test_loss):{1}.{4}} %")
            best_test_loss = average_current_test_loss
            self.log(f"Saving model...")
            self.environment.saveModel(mask_detector_model, self.environment.best_model_file)
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

    def train(self):
        bounding_box_extractor = BoundingBoxExtractor(self.environment.input_width, self.environment.input_height, 8)
        area = self.environment.input_width * self.environment.input_height
        mask_detector_model = self.loadModel()
        bounding_box_criterion_scaled = nn.L1Loss()
        bounding_box_criterion_unscaled = nn.L1Loss()
        optimizer = torch.optim.SGD([
                    {'params': mask_detector_model.parameters()}
                ], lr=self.learning_rate)
        cuda_enabled = torch.cuda.is_available()
        if cuda_enabled:
            bounding_box_criterion_scaled = bounding_box_criterion_scaled.cuda(self.gpu)
            bounding_box_criterion_unscaled = bounding_box_criterion_unscaled.cuda(self.gpu)
            mask_detector_model =  mask_detector_model.cuda(self.gpu)
            bounding_box_extractor = bounding_box_extractor.cuda(self.gpu)
        best_test_loss = self.testLoss(mask_detector_model)
        bounding_box_loss = 0
        self.log(f"Initial Test Loss {best_test_loss:{1}.{4}} ")
        for epoch in range(self.num_epochs):
            self.log(f"Epoch {epoch}")
            dataset_train = DataAugmentationDataset(True, self.data_path, self.visual_logging)
            train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=self.getWorkers())
            train_dataset_index = 0
            for i, (images, target_mask) in enumerate(train_loader):
                sys.stdout.flush()
                if cuda_enabled:
                    images = images.cuda(self.gpu)
                    target_mask = target_mask.cuda(self.gpu)
                images = Variable(images)
                target_mask = Variable(target_mask)
                if self.visual_logging:
                    cv2.imshow(f'input image', self.image_utils.toNumpy(images.data[0]))
                    cv2.imshow(f'target_mask', self.image_utils.toNumpy(target_mask.data[0]))
                mask_scaled, mask, roi_align_scaled, roi_align, bounding_boxes, bounding_boxes_scaled, boxes_index  = mask_detector_model.forward([images, self.use_bounding_box])
                target_mask_scaled = nn.MaxPool2d(8, 8, return_indices=False)(target_mask)
                target_mask = roi_align(target_mask, bounding_boxes, boxes_index)
                if self.visual_logging:
                    cv2.imshow(f'target_mask_scaled', self.image_utils.toNumpy(target_mask_scaled.data[0]))
                    cv2.imshow(f'target_mask_roi', self.image_utils.toNumpy(target_mask.data[0]))
                    cv2.waitKey(0)

                optimizer.zero_grad()
                if self.use_bounding_box:
                    bounding_boxes_scaled_target, bounding_boxes_target = bounding_box_extractor(target_mask_scaled)
                    bounding_box_loss = 1e-3 * (bounding_box_criterion_scaled(bounding_boxes, bounding_boxes_target) + bounding_box_criterion_scaled(bounding_boxes_scaled, bounding_boxes_scaled_target))
                loss_scaled = self.focalLoss(mask_scaled, target_mask_scaled)
                loss_unscaled = self.focalLoss(mask, target_mask)
                loss = ( 0.1 * loss_scaled ) + ( 0.7 * loss_unscaled ) + (0.2 * bounding_box_loss)
                loss.backward()
                optimizer.step()
                loss = loss.data[0]
                loss_scaled = loss_scaled.data[0]
                loss_unscaled = loss_unscaled.data[0]
                if self.use_bounding_box:
                    bounding_box_loss = bounding_box_loss.data[0]
                    self.log(f'Epoch [{epoch+1}/{self.num_epochs}] -- Iter [{train_dataset_index+1}/{math.ceil(len(dataset_train)/self.batch_size)}] -- Focal Loss: {loss:{1}.{4}}  -- BoundingBox Loss {bounding_box_loss:{1}.{4}} Focal Loss Unscaled: {loss_unscaled:{1}.{4}} -- Focal Loss Scaled {loss_scaled:{1}.{4}}')
                else:
                    self.log(f'Epoch [{epoch+1}/{self.num_epochs}] -- Iter [{train_dataset_index+1}/{math.ceil(len(dataset_train)/self.batch_size)}] -- Focal Loss: {loss:{1}.{4}}  --- Focal Loss Scaled {loss_scaled:{1}.{4}}')
                train_dataset_index = train_dataset_index + 1
            self.environment.saveModel(mask_detector_model, os.path.join(self.environment.models_path, f"{(epoch + 1)}.backup"))
            best_test_loss = self.testAndSaveIfImproved(best_test_loss, mask_detector_model)
