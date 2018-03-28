import argparse

from bauta.DatasetRetriever import DatasetRetriever

if __name__ == '__main__':
    '''Usage:
        python datasets_generator.py --data_path ../datasets--datasets_path ../CSVs

        The csv file is simply a file with a list of image ids with image URLs or paths (preferabily separated by a comma or a semicolon).
        It can be a list of image URLs, in such case, the images will be stored with their index
    '''
    argument_parser = argparse.ArgumentParser(description='Download a number of image lists from files')
    argument_parser.add_argument('--data_path', help='Path where the dataset will be stored (along other assets, see Readme.md)')
    argument_parser.add_argument('--datasets_path', type=str, help='The path containing the datasets, every file will be considered as a csv, and its name will be the class name of the dataset')
    argument_parser.add_argument('--is_background', type=bool, help='If the specified datasets_path is path of csv files (or single file) for backgrounds or not', default=False) 
    argument_parser.add_argument('--split_test_proportion', help='How much of each dataset should be test set. Default = 0.3', default=0.3)
    argument_parser.add_argument('--download_batch_size', help='How many images per dataset file are going to be downloaded at once', default=5)
    arguments = argument_parser.parse_args()
    DatasetRetriever(**arguments.__dict__)
