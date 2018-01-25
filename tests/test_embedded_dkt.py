import numpy
import random
import tensorflow as tf
import unittest

from models import embedded_dkt
from quick_experiment import dataset


class EmbeddedSeqLSTMModelTest(unittest.TestCase):
    """Tests for building and running a SeqPredictionModel instance"""

    MODEL = embedded_dkt.EmbeddedSeqLSTMModel

    def _get_random_sequence(self):
        return numpy.array([
            (x % self.feature_size) * numpy.sign(x - self.feature_size)
            for x in range(1, random.randint(3, 2*self.max_num_steps))])

    def setUp(self):
        num_examples = 50
        self.max_num_steps = 10
        self.feature_size = 5
        self.batch_size = 5  # Validation size
        # The matrix is an array of sequences of varying sizes. Each
        # sequence is an array of two elements.
        sequences = [self._get_random_sequence() for _ in range(num_examples)]
        self.matrix = numpy.array([x[:-1] for x in sequences])
        self.labels = numpy.array([x[1:] for x in sequences])
        self.partition_sizes = {
            'train': 0.65, 'test': 0.25, 'validation': 0.1
        }
        self.dataset = dataset.EmbeddedSequenceDataset()
        self.dataset.create_samples(self.matrix, self.labels, 1,
                                    self.partition_sizes, sort_by_length=True)
        self.dataset.set_current_sample(0)
        self.model_arguments = {
            'hidden_layer_size': 40, 'batch_size': self.batch_size,
            'logs_dirname': None,
            'log_values': 10,
            'max_num_steps': self.max_num_steps}
        # Check build does not raise errors
        tf.reset_default_graph()
        self.model = self.MODEL(
            self.dataset, **self.model_arguments)

    def test_build_network(self):
        """Test if the SeqLSTMModels is correctly built."""
        self.model.fit(close_session=True, training_epochs=50)

    def test_single_distributed_layer(self):
        """Test the model uses the same weights for the time distributed layer.
        """
        self.model.fit(training_epochs=50)
        with self.model.graph.as_default():
            for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope='softmax_layer'):
                self.assertNotIn('softmax_layer-1', variable.name,
                                 msg='More than one layer created.')

    def test_fit_loss(self):
        # Emulate the first part of the fit call
        with tf.Graph().as_default():
            self.model._build_inputs()
            # Build a Graph that computes predictions from the inference model.
            logits = self.model._build_layers()
            # Add to the Graph the Ops for loss calculation.
            loss = self.model._build_loss(logits)
            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = self.model._build_train_operation(loss)

            init = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            self.model.sess = tf.Session()
            self.model.sess.run([init, init_local])
            for epoch in range(10):
                loss_value = self.model.run_train_op(epoch, loss, 'train',
                                                     train_op)
                self.assertFalse(numpy.isnan(loss_value),
                                 msg='The loss value is nan.')

    @staticmethod
    def _get_correctly_predicted(labels, lengths, logit_labels):
        total_correct = 0
        for sequence_index, sequence in enumerate(labels):
            for label_index, true_label_vector in enumerate(
                    sequence[:lengths[sequence_index]]):
                true_label = numpy.argmax(true_label_vector)
                if true_label == logit_labels[sequence_index, label_index]:
                    total_correct += 1
        return total_correct

    def test_build_evaluation(self):
        self.model.fit(training_epochs=50)
        metric = self.model.evaluate('test')
        self.assertLessEqual(0, metric)
        self.assertGreaterEqual(1, metric)

    def test_build_predictions(self):
        self.model._build_inputs()
        logits = self.model._build_layers()
        predictions = self.model._build_predictions(logits)
        self.assertEqual(logits.shape[1], predictions.shape[1])
        self.assertEqual(predictions.shape[1], self.max_num_steps)
        self.assertEqual(2, len(predictions.shape.dims))

    def test_predict(self):
        """Test the prediction for each sequence element is the probability
        of the next element in sequence, for all possible elements."""
        self.model.fit(close_session=False, training_epochs=50)
        true, predictions = self.model.predict('test')
        self.assertIsInstance(predictions, numpy.ndarray)
        for true_sequence, predicted_sequence in zip(true, predictions):
            self.assertEqual(true_sequence.shape[0],
                             predicted_sequence.shape[0])

    def test_evaluate(self):
        """Test if the model returns a valid accuracy value."""
        self.model.fit(close_session=False, training_epochs=50)
        metric = self.model.evaluate('test')
        self.assertLessEqual(0, metric)
        self.assertGreaterEqual(1, metric)

    def test_fill_feed_dict(self):
        for instance in self.dataset._instances:
            self.assertLessEqual(instance.shape[0], 2*self.max_num_steps)
        self.model.build_all()
        batch_iterator = self.model._fill_feed_dict(partition_name='train')
        instances = next(batch_iterator)[self.model.instances_placeholder]
        self.assertEqual(instances.shape, (self.batch_size, self.max_num_steps))
        # As the maximum sequence lenght is 2, this should run exactly two times
        instances = next(batch_iterator)[self.model.instances_placeholder]
        self.assertEqual(instances.shape, (self.batch_size, self.max_num_steps))
        with self.assertRaises(StopIteration):
            next(batch_iterator)

    def test_no_prev_state(self):
        self.model.use_prev_state = False
        self.model.fit(close_session=True, training_epochs=50)


class CoEmbeddedSeqLSTMModelTest(EmbeddedSeqLSTMModelTest):
    """Tests for building and running a SeqPredictionModel instance"""

    MODEL = embedded_dkt.CoEmbeddedSeqLSTMModel


if __name__ == '__main__':
    unittest.main()
