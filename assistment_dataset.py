"""Classes for the assistment dataset."""

import numpy
from quick_experiment import dataset


class AssistmentDataset(dataset.EmbeddedSequenceDataset):

    def __init__(self, embedding_model=None):
        super(AssistmentDataset, self).__init__()
        self._embedding_model = embedding_model

    def create_samples(self, instances, labels, samples_num, partition_sizes,
                       use_numeric_labels=False, sort_by_length=False):
        """Creates samples with a random partition generator.

        Args:
            instances (:obj: iterable): instances to divide in samples.
            labels (:obj: iterable): labels to divide in samples.
            samples_num (int): the number of samples to create.
            partition_sizes (dict): a map from the partition names to their
                proportional sizes. The sum of all values must be less or equal
                to one.
            use_numeric_labels (bool): if True, the labels are converted to
                a continuous range of integers.
            sort_by_length (bool): If True, instances are sorted according the
                lenght of the sequence.
        """
        super(AssistmentDataset, self).create_samples(
            instances, labels, samples_num, partition_sizes, use_numeric_labels)
        self._fit_embedding_vocabulary()

    def _fit_embedding_vocabulary(self):
        if self._embedding_model is None:
            return
        word2index = {
            word: index
            for index, word in enumerate(self._embedding_model.wv.index2word)
        }
        # We have to add one to the result because the 0 embedding is for the
        # padded element of the sequence.
        map_function = numpy.vectorize(lambda x: (word2index.get(
            str(numpy.abs(x)), len(word2index)) + 1) * numpy.sign(x))
        self._instances = numpy.array([
            map_function(sequence) for sequence in self._instances
        ])

