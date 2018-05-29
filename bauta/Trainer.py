import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils

import sys
import math
import cv2
from multiprocessing import Process, Queue
import os
import datetime
import numpy as np

from bauta.DataAugmentationDataset import DataAugmentationDataset
from bauta.model.Model import Model
from bauta.DatasetConfiguration import DatasetConfiguration
from bauta.utils.EnvironmentUtils import EnvironmentUtils
from bauta.utils.ImageUtils import ImageUtils
from bauta.Constants import constants
from bauta.utils.CudaUtils import CudaUtils
from bauta.utils.SystemUtils import SystemUtils
from bauta.utils.InferenceUtils import InferenceUtils
from bauta.BoundingBox import BoundingBox

class Trainer():

    def __init__(self, data_path, visual_logging, reset_model, num_epochs, batch_size, learning_rate, momentum, gpu,\
        test_samples):
        super(Trainer, self).__init__()
        self.config = DatasetConfiguration(True, data_path)
        self.data_path = data_path
        self.visual_logging = visual_logging
        self.reset_model = reset_model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.test_loss_history = []
        self.gpu = gpu
        self.momentum = momentum
        self.test_samples = test_samples
        self.system_utils = SystemUtils()
        self.logger = self.system_utils.getLogger(self)
        self.image_utils = ImageUtils()
        self.environment = EnvironmentUtils(self.data_path)
        self.cuda_utils = CudaUtils()

    def getWorkers(self):
        if self.visual_logging:
            return 0
        else:
            return 6

    def focalLoss(self, predicted_masks, target_mask, visual_logging=False):
        foreground_probability = torch.mul(predicted_masks, target_mask)
        background_probability = torch.mul(predicted_masks - 1, target_mask - 1)
        probabilities = foreground_probability + background_probability
        modulating_factor = Variable(((-probabilities) + 1.0).abs().data.clone(), requires_grad=False)
        if visual_logging:
            self.logBatch(target_mask, "target_mask")
            self.logBatch(predicted_masks, "classes")
            self.logBatch(-probabilities+1, "Loss")
        log_likelyhood_loss = -torch.log(torch.clamp(probabilities, 0.001, 1.0))
        focal_loss = torch.mul(log_likelyhood_loss, modulating_factor)
        return focal_loss.mean()

    def testLoss(self):
        current_test_loss = None
        dataset_test = DataAugmentationDataset(False, self.data_path, self.visual_logging, self.test_samples)
        test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.getWorkers())
        average_current_test_loss = 0.0
        iterations = 0.0
        for i, (refiner_input_image, refiner_target_masks, input_images, target_mask, target_objects_in_image, bounding_boxes) in enumerate(test_loader):
            input_images, target_mask, target_objects_in_image = self.cuda_utils.toVariable(self.cuda_utils.cudify([input_images, target_mask, target_objects_in_image], self.gpu))
            if self.visual_logging:
               self.visualLoggingDataset(input_images, target_mask)
            predicted_masks, embeddings = self.model.forward(input_images)
            target_mask = nn.AvgPool2d(15)(target_mask)
            loss = self.focalLoss(predicted_masks, target_mask)
            average_current_test_loss = average_current_test_loss + loss[0].data[0]
            iterations = iterations + 1.0
        average_current_test_loss = average_current_test_loss / iterations
        self.test_loss_history.append(average_current_test_loss)
        return average_current_test_loss

    def log(self, text):
        self.logger.info(f"{datetime.datetime.utcnow()} -- {text}")

    def testAndSaveIfImproved(self, best_test_loss, total_loss):
        average_current_test_loss = float(total_loss.mean())#self.testLoss()
        print(f'AAAABBBB {type(average_current_test_loss)} vs {type(total_loss)} vs {type(best_test_loss)}')
        if average_current_test_loss < best_test_loss:
            self.log(f"Model Improved. Previous Best Test Loss {best_test_loss:{1}.{4}} | Current Best Test Loss  {average_current_test_loss:{1}.{4}} | Improvement Change: {(100.0 * (best_test_loss - average_current_test_loss) / average_current_test_loss):{1}.{4}} %")
            best_test_loss = average_current_test_loss
            self.log(f"Saving model...")
            self.environment.saveModel(self.model, self.environment.best_model_file)
            self.log(f"...model saved")
        else:
            self.log(f"Model did *NOT* Improve. Current Best Test Loss {best_test_loss:{1}.{4}} | Current Test Loss {average_current_test_loss:{1}.{4}} | Improvement Change: {(100.0 * (best_test_loss - average_current_test_loss) / average_current_test_loss):{1}.{4}} %")
        return best_test_loss

    def loadModel(self):
        model = None
        if not self.reset_model:
            model = self.environment.loadModel(self.environment.best_model_file)
            #model = Model(len(self.config.classes), 32, 5, 15)
            #model.backbone = old_model.backbone
            #model.mask_detectors = old_model.mask_detectors
        if self.reset_model or model is None:
            # incase no model is stored or in case the user wants to reset it
            model = Model(len(self.config.classes), 32, 5, 15)
        self.log(model)
        return model

    def logBatch(self, target_mask, title):
        if self.visual_logging:
            for current_index in range(target_mask.size()[0]):
                for current_class in range(len(self.config.classes)):
                    cv2.imshow(f'{title} {current_index}/"{self.config.classes[current_class]}".', self.image_utils.toNumpy(target_mask.data[current_index][current_class]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def visualLoggingDataset(self, input_images, target_mask):
        if self.visual_logging:
            for current_index in range(input_images.size()[0]):
                cv2.imshow(f'Input {current_index}', self.image_utils.toNumpy(input_images.data[current_index]))
                for current_class_index in range(target_mask[current_index].size()[0]):
                    cv2.imshow(f'Target {current_index}/{self.config.classes[current_class_index]}', self.image_utils.toNumpy(target_mask.data[current_index][current_class_index]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def visualLoggingOutput(self, network_output, target_mask_scaled):
        if self.visual_logging:
            classes, object_found, mask_scaled, mask, roi_align, bounding_boxes = network_output
            current_found_index = 0
            for current_index in range(mask_scaled.size()[0]):
                for current_class in range(len(self.config.classes)):
                    cv2.imshow(f'Target {current_index}/"{self.config.classes[current_class]}".', self.image_utils.toNumpy(target_mask_scaled.data[current_index][current_class]))
                    cv2.imshow(f'Output {current_index}/"{self.config.classes[current_class]}".', self.image_utils.toNumpy(mask_scaled.data[current_index][current_class]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def logLoss(self, loss, loss_refiner, epoch, train_dataset_index, dataset_train):
        if loss_refiner is None :
            refiner_loss_message = 'Mask detector not accurate enough to train samples in batch.'
        else:
            refiner_loss_message = f'{loss_refiner:{1}.{4}}'
        self.log(f'Epoch [{epoch+1}/{self.num_epochs}] -- Iter [{train_dataset_index+1}/{math.ceil(len(dataset_train)/self.batch_size)}] -- Mask Loss: {loss:{1}.{4}} -- Refiner Loss: {refiner_loss_message}')

    def buildOptimizer(self):
        optimizer = torch.optim.SGD(params = self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        return optimizer

    def buildRefinerOptimizer(self):
        optimizer = torch.optim.SGD(params = self.model.mask_refiners.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        return optimizer

    def logRefiner(self, refiner_input_image, target_mask, predicted_mask, predicted_refined_mask, class_index):
        if self.visual_logging:
            cv2.imshow(f'Refiner Input.', self.image_utils.toNumpy(refiner_input_image.squeeze()))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.visual_logging:
            cv2.imshow(f'Refiner Target Mask "{self.config.classes[class_index]}".', self.image_utils.toNumpy(target_mask.data.squeeze()))
            cv2.imshow(f'Refiner Predicted Mask "{self.config.classes[class_index]}".', self.image_utils.toNumpy(torch.nn.Upsample(size=(target_mask.size()[2], target_mask.size()[3]), mode='bilinear')(predicted_mask).data.squeeze()))
            #cv2.imshow(f'Refiner "{self.config.classes[class_index]}".', self.image_utils.toNumpy(input_image.data.squeeze()))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.visual_logging:
            cv2.imshow(f'Refiner Predicted Mask "{self.config.classes[class_index]}".', self.image_utils.toNumpy(predicted_refined_mask.data.squeeze()))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def trainRefiner(self, predicted_masks, target_masks, embeddings, refiner_input_images, refiner_target_masks, bounding_boxes):
        mean_refiner_loss = 0.0
        mean_refiner_count = 0
        embeddings = Variable(embeddings.data)
        predicted_masks = Variable(predicted_masks.data)
        for batch_index in range(bounding_boxes.size()[0]):
            refiner_input_image = refiner_input_images[batch_index:batch_index+1, :, :, :]
            target_masks_batch = target_masks[batch_index:batch_index+1, :, :, :]
            refiner_target_masks_batch = refiner_target_masks[batch_index:batch_index+1, :, :, :]
            for object_index in range(bounding_boxes.size()[1]):
                bounding_box = bounding_boxes[batch_index, object_index, :]
                if(bounding_box[1:5].sum() > 1.0):
                    class_index = int(bounding_box[0])
                    left = bounding_box[1]
                    top = bounding_box[2]
                    right = bounding_box[3]
                    bottom = bounding_box[4]
                    bounding_box_object = BoundingBox(top, left, bottom, right)
                    bounding_box_object_scaled = bounding_box_object.resize(512, 512, 16, 16)
                    if bounding_box_object_scaled.area > 1:
                        refiner_input_image_cropped = refiner_input_image[:, :,bounding_box_object.top:bounding_box_object.bottom+1,bounding_box_object.left:bounding_box_object.right+1]
                        embeddings_cropped = embeddings[batch_index:batch_index+1, :,bounding_box_object_scaled.top:bounding_box_object_scaled.bottom+1,bounding_box_object_scaled.left:bounding_box_object_scaled.right+1]
                        target_mask = target_masks_batch[:, class_index:class_index+1,bounding_box_object_scaled.top:bounding_box_object_scaled.bottom+1,bounding_box_object_scaled.left:bounding_box_object_scaled.right+1]
                        refiner_target_mask = refiner_target_masks_batch[:, class_index:class_index+1,bounding_box_object.top:bounding_box_object.bottom+1,bounding_box_object.left:bounding_box_object.right+1]
                        predicted_mask_cropped = predicted_masks[batch_index:batch_index+1, class_index:class_index+1,bounding_box_object_scaled.top:bounding_box_object_scaled.bottom+1,bounding_box_object_scaled.left:bounding_box_object_scaled.right+1]
                        predicted_refined_mask  = self.model.mask_refiners([refiner_input_image_cropped, embeddings_cropped, class_index, predicted_masks[batch_index:batch_index+1, class_index:class_index + 1, bounding_box_object_scaled.top:bounding_box_object_scaled.bottom+1,bounding_box_object_scaled.left:bounding_box_object_scaled.right+1]])#predicted_mask_cropped])
                        self.logRefiner(refiner_input_image_cropped, target_mask, predicted_mask_cropped, predicted_refined_mask, class_index)
                        self.refiner_optimizer.zero_grad()
                        loss = self.focalLoss(predicted_refined_mask, refiner_target_mask)
                        loss.backward()
                        mean_refiner_loss = mean_refiner_loss + loss[0].mean().data[0]
                        mean_refiner_count = mean_refiner_count + 1
                        self.refiner_optimizer.step()
        if mean_refiner_count is 0:
            return None
        else:
            return mean_refiner_loss / mean_refiner_count

    def train(self):
        self.model = self.cuda_utils.cudify([self.loadModel()], self.gpu)[0]
        best_test_loss = self.testLoss()
        self.log(f"Initial Test Loss {best_test_loss:{1}.{4}} ")
        optimizer = self.buildOptimizer()
        self.refiner_optimizer = self.buildRefinerOptimizer()
        for epoch in range(self.num_epochs):
            self.log(f"Epoch {epoch}")
            dataset_train = DataAugmentationDataset(True, self.data_path, self.visual_logging)
            train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=self.getWorkers())
            for train_dataset_index, (refiner_input_image, refiner_target_masks, input_images, target_mask, target_objects_in_image, bounding_boxes) in enumerate(train_loader):
                sys.stdout.flush()
                input_images, target_mask, target_objects_in_image, refiner_target_masks = self.cuda_utils.toVariable(self.cuda_utils.cudify([input_images, target_mask, target_objects_in_image, refiner_target_masks], self.gpu))
                self.visualLoggingDataset(input_images, target_mask)
                predicted_masks, embeddings  = self.model.forward(input_images)
                optimizer.zero_grad()
                target_mask = nn.AvgPool2d(15)(target_mask)
                loss = self.focalLoss(predicted_masks, target_mask)
                loss.backward()
                optimizer.step()
                loss_refiner = self.trainRefiner(predicted_masks, target_mask, embeddings, refiner_input_image, refiner_target_masks, bounding_boxes)
                self.logLoss(loss.data[0], loss_refiner, epoch, train_dataset_index, dataset_train)
                if (train_dataset_index + 1) % 500 == 0: 
                    self.environment.saveModel(self.model, f"{(epoch + 1)}.backup")
                    best_test_loss = self.testAndSaveIfImproved(best_test_loss, loss + loss_refiner)
            self.environment.saveModel(self.model, f"{(epoch + 1)}.backup")
            best_test_loss = self.testAndSaveIfImproved(best_test_loss, loss + loss_refiner)
