import unittest
import sys, os
import random
import shutil
import itertools

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
        return images_path, data_path

    def removeDataset(images_path, data_path):
        shutil.rmtree(images_path)
        shutil.rmtree(data_path)

    def test_train_within_epoch_improves_loss(self):
        images_path, data_path = TestTrainer.createDataset()
        trainer = Trainer(data_path, visual_logging=False, reset_model=True, num_epochs=1, batch_size=1, learning_rate=0.1, gpu=0, \
            loss_scaled_weight=0.5, loss_unscaled_weight=0.5, only_masks=False)
        trainer.train()
        TestTrainer.removeDataset(images_path, data_path)
        self.assertTrue((trainer.test_loss_history[0] - trainer.test_loss_history[1]) / trainer.test_loss_history[0] > 0.01)


if __name__ == '__main__':
    unittest.main()
