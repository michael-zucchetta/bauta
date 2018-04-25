import unittest
import sys, os
import random
import shutil
import itertools

import torch

from bauta.DatasetGenerator import DatasetGenerator
from bauta.utils.SystemUtils import SystemUtils
from bauta.Trainer import Trainer

class TestTrainer(unittest.TestCase):

    def files(datasets_with_attributes):
        dataset = [element['dataset'] for element in datasets_with_attributes]
        if len(dataset) > 0 and len(dataset[0]) > 0:
            return {element[1] for element in dataset[0][0]}
        else:
            return {}

    def createDataset():
        system_utils = SystemUtils()
        images_path = f'/tmp/{random.randint(0, 100000)}'
        data_path = f'/tmp/{random.randint(0, 100000)}'
        system_utils.makeDirIfNotExists(images_path)
        squares_image_path = f'{images_path}/square.txt'
        with open(f'{images_path}/square.txt','w') as file:
            file.write('./test/data/images/square/square_1.png\n./test/data/images/square/square_2.png')
        circles_image_path = f'{images_path}/circle.txt'
        with open(f'{images_path}/circle.txt','w') as file:
            file.write('./test/data/images/circle/circle_1.png\n./test/data/images/circle/circle_2.png')
        backgrounds_image_path = f'{images_path}/background.txt'
        with open(f'{images_path}/background.txt','w') as file:
            file.write('./test/data/images/background/background_1.png\n./test/data/images/background/background_2.png')
        dataset_generator = DatasetGenerator(data_path)
        datasets_with_attributes = dataset_generator.generateDatasetFromListOfImages(images_path, 0.5, 5)
        with open(f'{data_path}/config.yaml','a') as file:
            file.write("\ndata_sampling:\n")
            file.write("  probability_using_cache: 0.0\n")
        return images_path, data_path

    def removeDataset(images_path, data_path):
        shutil.rmtree(images_path)
        shutil.rmtree(data_path)


    def test_focalLoss(self):
        all_objects_in_image = torch.ones(1, 3)
        targets = torch.zeros(1, 3, 8, 8)
        targets[0, 0, 0:8, 0:7] = 1
        targets[0, 1, 0:8, 7:8] = 1
        targets[0, 2, :, :] = 0
        outputs = torch.zeros(1, 3, 8, 8)
        outputs[0, 0, 0:8, 0:3] = 1 #  ~50% wrong
        outputs[0, 1, 0:8, 0:7] = 1 # ~700% wrong
        outputs[0, 2, 0:8, 0:8] = 0 # 100% correct
        loss = Trainer.focalLoss(outputs, targets, all_objects_in_image)
        self.assertTrue(loss[0] / 50 < 1.1)
        self.assertTrue(loss[1] / 700 < 1.1)
        self.assertTrue(loss[2]  < 1e-3)

    def test_train_within_epoch_improves_loss(self):
        images_path, data_path = TestTrainer.createDataset()
        trainer = Trainer(data_path, visual_logging=False, reset_model=True, num_epochs=4, batch_size=1, learning_rate=0.01, momentum=0.1, gpu=0, \
            loss_scaled_weight=0.5, loss_unscaled_weight=0.5, only_masks=True)
        trainer.train()
        TestTrainer.removeDataset(images_path, data_path)
        best_test_loss = min(trainer.test_loss_history)
        self.assertTrue((trainer.test_loss_history[0] - best_test_loss) / trainer.test_loss_history[0] > 0.01)


if __name__ == '__main__':
    unittest.main()
