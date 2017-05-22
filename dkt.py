import argparse
import os
import dkt_model
import json
import utils

from quick_experiment import dataset
from quick_experiment.models import lstm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dirname', type=str, default=None,
                        help='Path to directory to store tensorboard info')
    parser.add_argument('--filename', type=str,
                        help='The path to the pickled file with the processed'
                             'sequences.')
    parser.add_argument('--test_predictions_filename', type=str,
                        help='The path to the file to store the predictions')
    parser.add_argument('--configuration_filename', type=str, default=None,
                        help='Filename with json configuration dict.')
    return parser.parse_args()


class DKTDataset(dataset.LabeledSequenceDataset):

    @property
    def labels_type(self):
        return self._labels[0].dtype

    def classes_num(self, _=None):
        """The number of problems in the dataset"""
        assert self.feature_vector_size % 2 == 0
        return (self.feature_vector_size / 2) + 1


def read_configuration(args):
    if args.configuration_filename is None:
        return {
            'hidden_layer_size': 200, 'batch_size': 50,
            'logs_dirname': args.logs_dirname,
            'log_values': 50, 'training_epochs': 500, 'max_num_steps': 100
        }, {'train': 0.7, 'test': 0.2, 'validation': 0.1}
    with open(args.configuration_filename) as json_file:
        config = json.load(json_file)
    config['logs_dirname'] = args.logs_dirname
    dataset_config = config.pop('dataset_config')
    return config, dataset_config


def main():
    args = parse_arguments()
    assistment_dataset = DKTDataset()
    sequences, labels = utils.pickle_from_file(args.filename)
    experiment_config, partitions = read_configuration(args)
    print 'Creating samples'
    assistment_dataset.create_samples(
        sequences, labels, partition_sizes=partitions, samples_num=1,
        sort_by_length=True)

    assistment_dataset.set_current_sample(0)

    print 'Dataset Configuration'
    print partitions
    print 'Experiment Configuration'
    print experiment_config
    model = dkt_model.DktLSTMModel(assistment_dataset, **experiment_config)
    model.fit(partition_name='train', close_session=False)
    predicted_labels = model.predict('test')
    utils.pickle_to_file(
        predicted_labels,
        os.path.join(args.test_predictions_filename, 'predictions.p'))



if __name__ == '__main__':
    main()

