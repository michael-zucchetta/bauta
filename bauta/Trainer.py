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
from bauta.model.MaskRefiners import MaskRefiners

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

    def testLoss(self):
        current_test_loss = None
        dataset_test = DataAugmentationDataset(False, self.data_path, self.visual_logging, self.test_samples)
        test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.getWorkers())
        total_loss_average = 0.0
        loss_mask_average = 0.0
        loss_refiner_average = 0.0
        loss_classifier_average = 0.0
        iterations = 0.0
        for i, (input_images, target_mask, bounding_boxes) in enumerate(test_loader):
            input_images, target_mask = self.cuda_utils.toVariable(self.cuda_utils.cudify([input_images, target_mask], self.gpu))
            if self.visual_logging:
               self.visualLoggingDataset(input_images, target_mask)
            total_loss, loss_mask, loss_refiner, loss_classifier =  self.computeLoss(input_images, target_mask, bounding_boxes)
            total_loss_average = total_loss_average + total_loss[0].data[0]
            loss_mask_average = loss_mask_average + loss_mask[0].data[0]
            loss_refiner_average = loss_refiner_average + loss_refiner[0].data[0]
            loss_classifier_average = loss_classifier_average + loss_classifier[0].data[0]
            iterations = iterations + 1.0
        total_loss_average = total_loss_average / iterations
        loss_mask_average = loss_mask_average / iterations
        loss_refiner_average = loss_refiner_average / iterations
        loss_classifier_average = loss_classifier_average / iterations
        self.log(f'Test Loss -- Total Loss: {total_loss_average:{1}.{4}} -- Classifier Loss: {loss_classifier_average:{1}.{4}} -- Mask Loss: {loss_mask_average:{1}.{4}} -- Refined Mask Loss: {loss_refiner_average:{1}.{4}}')
        self.test_loss_history.append(total_loss_average)
        return loss_refiner_average

    def log(self, text):
        self.logger.info(f"{datetime.datetime.utcnow()} -- {text}")

    def testAndSaveIfImproved(self, best_test_loss):
        average_current_test_loss = self.testLoss()
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
        else:
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

    def logLoss(self, total_loss, loss_mask, loss_refiner, loss_classifier, epoch, train_dataset_index, dataset_train):
        self.log(f'Epoch [{epoch+1}/{self.num_epochs}] -- Iter [{train_dataset_index+1}/{math.ceil(len(dataset_train)/self.batch_size)}] --  Total Loss: {total_loss.data[0]:{1}.{4}} -- Classifier Loss: {loss_classifier.data[0]:{1}.{4}} -- Mask Loss: {loss_mask.data[0]:{1}.{4}} -- Refined Mask Loss: {loss_refiner.data[0]:{1}.{4}}')

    def buildOptimizer(self):
        optimizer = torch.optim.SGD([ 
            {'params': self.model.parameters(), 'lr': self.learning_rate}], momentum=self.momentum, nesterov=True)
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

    def balancedLoss(self, predictions, targets):
        foreground = Variable(targets.data, requires_grad=False)
        background = Variable((1.0 - targets).data, requires_grad=False)
        foreground_loss = nn.L1Loss(size_average=True, reduce=False)(foreground * predictions, foreground * targets).mean() / (foreground.mean() + 1e-10)
        background_loss = nn.L1Loss(size_average=True, reduce=False)(background * predictions, background * targets).mean() / (background.mean() + 1e-10)
        if self.visual_logging and len(targets.size()) == 4:
            self.logBatch(foreground, "Tar.Fore.")
            self.logBatch(nn.L1Loss(size_average=False, reduce=False)(foreground * predictions, foreground * targets), "Fore loss")
            self.logBatch(background, "Tar.Back.")
            self.logBatch(nn.L1Loss(size_average=False, reduce=False)(background * predictions, background * targets), "Back loss")
        return (foreground_loss + background_loss) / 2.0

    def computeLoss(self, input_images, target_mask, bounding_boxes):
        predicted_masks, mask_embeddings, embeddings_merged, embeddings_2, embeddings_4, embeddings_8 = self.model.forward(input_images)
        self.logBatch(predicted_masks, "Predict")
        target_mask_scaled_16 = (nn.AvgPool2d(16)(target_mask) > 0.5).float()
        self.logBatch(target_mask_scaled_16, "Target")
        loss_mask = self.balancedLoss(predicted_masks, target_mask_scaled_16)
        classifier_predictions = self.model.classifiers([predicted_masks, embeddings_merged])
        classifier_targets = (target_mask.view(target_mask.size()[0], target_mask.size()[1], -1).sum(2) > 0).float()
        loss_classifier = self.balancedLoss(classifier_predictions, classifier_targets) * 0.1
        self.logBatch(mask_embeddings[:,0,:,:,:].squeeze(1), "Embed")
        predicted_refined_mask = self.model.mask_refiners([input_images.size(), predicted_masks, mask_embeddings, embeddings_merged, embeddings_2, embeddings_4, embeddings_8])        
        self.logBatch(predicted_refined_mask, "Predict")
        target_mask_scaled_2 = (nn.AvgPool2d(2)(target_mask) > 0.5).float()
        self.logBatch(target_mask_scaled_2, "Target")
        loss_refiner = self.balancedLoss(predicted_refined_mask, target_mask_scaled_2)
        total_loss = loss_mask + loss_refiner + loss_classifier
        return total_loss, loss_mask, loss_refiner, loss_classifier

    def train(self):
        self.model = self.cuda_utils.cudify([self.loadModel()], self.gpu)[0]
        best_test_loss = self.testLoss()
        self.log(f"Initial Test Loss {best_test_loss:{1}.{4}} ")
        optimizer = self.buildOptimizer()
        for epoch in range(self.num_epochs):
            self.log(f"Epoch {epoch}")
            dataset_train = DataAugmentationDataset(True, self.data_path, self.visual_logging)
            train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=self.getWorkers())
            for train_dataset_index, (input_images, target_mask, bounding_boxes) in enumerate(train_loader):
                sys.stdout.flush()
                input_images, target_mask = self.cuda_utils.toVariable(self.cuda_utils.cudify([input_images, target_mask], self.gpu))
                self.visualLoggingDataset(input_images, target_mask)
                optimizer.zero_grad()
                total_loss, loss_mask, loss_refiner, loss_classifier =  self.computeLoss(input_images, target_mask, bounding_boxes)
                total_loss.backward()
                optimizer.step()
                self.logLoss(total_loss, loss_mask, loss_refiner, loss_classifier, epoch, train_dataset_index, dataset_train)
                if (train_dataset_index + 1) % 1000 is 0:
                    best_test_loss = self.testAndSaveIfImproved(best_test_loss)

            self.environment.saveModel(self.model, f"{(epoch + 1)}.backup")
            best_test_loss = self.testAndSaveIfImproved(best_test_loss)
