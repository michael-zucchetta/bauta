import unittest
import sys, os
import random
import shutil
import itertools

from bauta.DatasetGenerator import DatasetGenerator
from bauta.utils.SystemUtils import SystemUtils

class TestDatasetGenerator(unittest.TestCase):

    def files(datasets_with_attributes):
        dataset = [element['dataset'] for element in datasets_with_attributes]
        if len(dataset) > 0 and len(dataset[0]) > 0:
            return {element[1] for element in dataset[0][0]}
        else:
            return {}

    def generateDatasetFromListOfImages_one_file_per_class(self):
        system_utils = SystemUtils()
        images_path = f'/tmp/{random.randint(0, 100000)}'
        data_path = f'/tmp/{random.randint(0, 100000)}'
        system_utils.makeDirIfNotExists(images_path)
        squares_image_path = f'{images_path}/square.txt'
        with open(f'{images_path}/square.txt','w') as file:
            file.write('./test/data/images/square/square_1.png')
        circles_image_path = f'{images_path}/circle.txt'
        with open(f'{images_path}/circle.txt','w') as file:
            file.write('./test/data/images/circle/circle_1.png')
        backgrounds_image_path = f'{images_path}/background.txt'
        with open(f'{images_path}/background.txt','w') as file:
            file.write('./test/data/images/background/background_1.png')
        dataset_generator = DatasetGenerator(data_path)
        datasets_with_attributes = dataset_generator.generateDatasetFromListOfImages(images_path, 0.25, 5)
        self.assertEqual({'./test/data/images/square/square_1.png'}, \
            TestDatasetGenerator.files(list(filter(lambda element: element['is_train'] and element['class'] == 'square', datasets_with_attributes))))
        self.assertEqual({'./test/data/images/circle/circle_1.png'}, \
            TestDatasetGenerator.files(list(filter(lambda element: element['is_train'] and element['class'] == 'circle', datasets_with_attributes))))
        self.assertEqual({'./test/data/images/background/background_1.png'}, \
            TestDatasetGenerator.files(list(filter(lambda element: element['is_train'] and element['class'] == 'background', datasets_with_attributes))))
        self.assertEqual({}, \
            TestDatasetGenerator.files(list(filter(lambda element: not element['is_train'] and element['class'] == 'square', datasets_with_attributes))))
        self.assertEqual({}, \
            TestDatasetGenerator.files(list(filter(lambda element: not element['is_train'] and element['class'] == 'circle', datasets_with_attributes))))
        self.assertEqual({}, \
            TestDatasetGenerator.files(list(filter(lambda element: not element['is_train'] and element['class'] == 'background', datasets_with_attributes))))

        shutil.rmtree(images_path)
        shutil.rmtree(data_path)

   

if __name__ == '__main__':
    unittest.main()
