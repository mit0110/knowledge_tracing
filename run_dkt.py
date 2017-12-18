import argparse
import os
import json
import utils

from models import dkt
from quick_experiment import dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dirname', type=str, default=None,
                        help='Path to directory to store tensorboard info')
    parser.add_argument('--filename', type=str,
                        help='The path to the pickled file with the processed'
                             'sequences.')
    parser.add_argument('--test_predictions_filename', type=str,
                        help='The path to the file to store the predictions')
    parser.add_argument('--training_epochs', type=int, default=500,
                        help='Number of epochs to run.')
    parser.add_argument('--hidden_layer_size', type=int, default=100,
                        help='Number of cells in the recurrent layer.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number if instances to process at the same time.')
    parser.add_argument('--log_values', type=int, default=50,
                        help='How many training epochs to wait before logging'
                             'the accuracy in validation.')
    parser.add_argument('--max_num_steps', type=int, default=100,
                        help='Number of time steps to unroll the network.')


    return parser.parse_args()


class DKTDataset(dataset.LabeledSequenceDataset):

    @property
    def labels_type(self):
        return self._labels[0].dtype

    def classes_num(self, _=None):
        """Number of problems in the dataset"""
        assert self.feature_vector_size % 2 == 0
        return (self.feature_vector_size / 2) + 1


def read_configuration(args):
    config = {
        'hidden_layer_size': args.hidden_layer_size,
        'batch_size': args.batch_size,
        'logs_dirname': args.logs_dirname,
        'log_values': args.log_values,
        'max_num_steps': args.max_num_steps,
    }
    dataset_config = {'train': 0.7, 'test': 0.2, 'validation': 0.1}
    return config, dataset_config


def main():
    args = parse_arguments()
    assistment_dataset = DKTDataset()
    print('Reading dataset')
    sequences, labels = utils.pickle_from_file(args.filename)
    experiment_config, partitions = read_configuration(args)
    print('Creating samples')
    assistment_dataset.create_samples(
        sequences, labels, partition_sizes=partitions, samples_num=1,
        sort_by_length=True)

    assistment_dataset.set_current_sample(0)

    print('Dataset Configuration')
    print(partitions)
    print('Experiment Configuration')
    print(experiment_config)
    model = dkt.DktLSTMModel(assistment_dataset, **experiment_config)
    model.fit(partition_name='train', close_session=False,
              training_epochs=args.training_epochs)
    predicted_labels = model.predict('test')
    utils.pickle_to_file(
        predicted_labels,
        os.path.join(args.test_predictions_filename, 'predictions.p'))
    utils.pickle_to_file(
        (model.training_performance, model.validation_performance),
        os.path.join(args.test_predictions_filename, 'performances.p'))


if __name__ == '__main__':
    main()

