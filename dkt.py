import argparse
import numpy
import utils

from quick_experiment import dataset
from quick_experiment.models import dkt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str,
                        help='The path to the pickled file with the processed'
                             'sequences.')
    return parser.parse_args()


class DKTDataset(dataset.LabeledSequenceDataset):

    @property
    def labels_type(self):
        return self._labels[0].dtype

    def classes_num(self, _=None):
        """The number of problems in the dataset"""
        return self.feature_vector_size

    def _pad_batch(self, batch_instances, batch_labels,
                   max_sequence_length=None):
        lengths = self._get_sequence_lengths(batch_instances)
        padded_batch = numpy.zeros((batch_instances.shape[0],
                                    max_sequence_length,
                                    self.feature_vector_size))
        padded_labels = numpy.zeros(
            (batch_instances.shape[0], max_sequence_length, 2))
        for index, sequence in enumerate(batch_instances):
            if lengths[index] <= max_sequence_length:
                padded_batch[index, :lengths[index]] = sequence
                padded_labels[index, :lengths[index]] = batch_labels[index]
            else:
                padded_batch[index] = sequence[-max_sequence_length:]
                padded_labels[index] = batch_labels[
                    index][-max_sequence_length:]
        return padded_batch, padded_labels, lengths


def main():
    args = parse_arguments()
    assistment_dataset = DKTDataset()
    sequences, labels = utils.pickle_from_file(args.filename)
    partitions = {'train': 0.7, 'test': 0.2, 'validation': 0.1}
    assistment_dataset.create_samples(
        numpy.array([x.tocsr() for x in sequences]),
        numpy.array([numpy.array(zip(*x)) for x in labels]),
        partition_sizes=partitions, samples_num=1)

    assistment_dataset.set_current_sample(0)

    experiment_config = {
        'hidden_layer_size': 20, 'batch_size': 500, 'logs_dirname': None,
        'log_values': 100, 'training_epochs': 1000, 'max_num_steps': 10
    }
    model = dkt.DKTModel(assistment_dataset, **experiment_config)
    model.fit(partition_name='train', close_session=False)


if __name__ == '__main__':
    main()

