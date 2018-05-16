
from bauta.Trainer import Trainer
import click
import os

@click.command()
@click.option('--data_path', default=f'{os.getcwd()}', help='Data path.')
@click.option('--visual_logging', default=False, help='Display additional logging using images (only using desktop). Do not use it in a server, it requires a desktop environment.')
@click.option('--reset_model', default=False, help='Reset model (start from scratch).')
@click.option('--num_epochs', default=10000, help='Number of epochs.')
@click.option('--batch_size', default=16, help='Batch size.')
@click.option('--learning_rate', default=0.0001, help='Learning rate')
@click.option('--momentum', default=0.9, help='Momentum')
@click.option('--gpu', default=0, help='GPU index')
@click.option('--test_samples', default=-1, help='Number of test samples, -1 means using all the test dataset, any positive number means using that amount of samples.')
def train(data_path, visual_logging, reset_model, num_epochs, batch_size, learning_rate, momentum, gpu, test_samples):
    if test_samples < 0:
        test_samples = None
    if not reset_model:
        reset_model_classes = None
    trainer = Trainer(data_path, visual_logging, reset_model, num_epochs, batch_size, learning_rate, momentum, gpu, \
        test_samples)
    trainer.train()

if __name__ == '__main__':
    train()
