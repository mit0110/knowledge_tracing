"""Counts the trainable parameter of a model."""

import os
import dkt_model
import numpy
import quick_experiment.utils

from quick_experiment import dataset

SKILLS = 140

CONFIG = {
    'hidden_layer_size': 200, 'batch_size': 100, 'training_epochs': 0,
    'max_num_steps': 100
}


class MockDKTDataset(dataset.LabeledSequenceDataset):
    @property
    def labels_type(self):
        return numpy.float32

    def classes_num(self, _=None):
        """The number of problems in the dataset"""
        return SKILLS + 1

    @property
    def feature_vector_size(self):
        return SKILLS * 2


def main():
    assistment_dataset = MockDKTDataset()

    model = dkt_model.DktLSTMModel(assistment_dataset, **CONFIG)
    model.fit(partition_name='train')
    model.count_trainable_parameters()


if __name__ == '__main__':
    main()

