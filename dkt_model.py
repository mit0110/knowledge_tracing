"""Definition of the DKT LSTM network"""
import numpy
import tensorflow as tf

from quick_experiment.models import seq_lstm
from sklearn import metrics


class DktLSTMModel(seq_lstm.SeqLSTMModel):

    def _build_loss(self, logits):
        """Calculates the avg binary cross entropy using the sigmoid function.

        Args:
            logits: Tensor - [batch_size, max_num_steps, classes_num]
        """
        mask = tf.sequence_mask(self.lengths_placeholder, self.max_num_steps)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(self.labels_placeholder, logits.dtype))
        # loss has shape [batch_size, max_num_steps, classes_num]
        loss = tf.div(
            tf.reduce_sum(tf.boolean_mask(loss, mask)),
            tf.cast(tf.reduce_sum(self.lengths_placeholder), loss.dtype))
        return loss

    def _build_predictions(self, logits):
        """Return a tensor with the predicted performance of next exercise.

        The prediction for each step is float with the probability of the
        next exercise being correct. To know the true next execise we use
        the information from self.labels_placeholder

        Args:
            logits: Logits tensor, float - [batch_size, max_num_steps,
                num_classes].

        Returns:
            A float64 tensor with the predictions, with shape [batch_size,
            max_num_steps, num_classes].
        """
        predictions = tf.nn.sigmoid(logits)
        return predictions

    @staticmethod
    def _get_short_labels(labels, seq_indices):
        """Returns only the labels or scores in true_indices plus the last one.
        """
        # Remove the prediction for the last class (the EOS symbol) and
        # For the last timestep (which should predict the EOS).
        seq_short_prediction = labels[:-1, :-1][seq_indices]
        assert seq_short_prediction.ndim == 1
        # Assert we took only one element per sequence.
        assert seq_short_prediction.shape[0] == seq_indices.shape[0]
        return numpy.append(seq_short_prediction, [labels[-1, -1]])

    def _get_batch_prediction(self, partition_name):
        true = []
        predictions = []
        true_indices = []
        lengths = numpy.zeros(self.batch_size)
        for feed_dict in self._fill_feed_dict(partition_name, reshuffle=False):
            step_prediction = self.sess.run(self.predictions,
                                            feed_dict=feed_dict)
            true.append(feed_dict[self.labels_placeholder])
            predictions.append(step_prediction)
            # Get the true next exercise in the sequence.
            true_indices.append(
                feed_dict[self.instances_placeholder][:, :, self.dataset.classes_num() - 1].astype(numpy.bool))
            lengths += feed_dict[self.lengths_placeholder]
        # each prediction and true has shape
        # [batch_size, max_num_step, classes_num - 1]
        predictions = numpy.vstack(predictions)
        true = numpy.vstack(true)
        # each true indices has shape
        # [batch_size, max_num_step, classes_num - 1]
        true_indices = numpy.vstack(true_indices)

        short_predictions = []
        short_true = []
        for index, length in enumerate(lengths):
            # Shape [sequence_length, num_classes]
            seq_prediction = predictions[index, :length]
            # Shape [sequence_length - 1, num_classes -1]
            seq_indices = true_indices[index, 1:length]
            short_predictions.append(self._get_short_labels(
                seq_prediction, seq_indices))
            seq_true = true[index, :length]
            short_true.append(self._get_short_labels(seq_true, seq_indices))
        return short_true, short_predictions

    def predict(self, partition_name):
        """Applies the classifier to all elements in partition name.

        Returns:
            A tuple (true, predictions). true has the true labels of the
            predicted elements, predictions has the predicted labels of the
            elements. Each label is a the probability of the next exercise
            being correct.
            Both true and predictions are arrays (sequences) of length
            self.dataset.num_examples(partition_name). The elements of the list
            are the labels of the sequence represented as an array.
        """
        predictions = []
        true = []
        self.dataset.reset_batch()
        with self.graph.as_default():
            while self.dataset.has_next_batch(self.batch_size, partition_name):
                batch_true, batch_prediction = self._get_batch_prediction(
                    partition_name)
                predictions.extend(batch_prediction)
                true.extend(batch_true)

        return numpy.array(true), numpy.array(predictions)

    def _build_evaluation(self, logits):
        """Function not used. We need the full sequence to calculate the
        performance, not step by step.
        """
        return None

    def evaluate_validation(self, unused_arg):
        partition = 'validation'
        true, predictions = self.predict(partition)
        r2s = []
        mse = []
        for sequence_true, sequence_predicted in zip(true, predictions):
            # Calculate the performance per sequence
            r2s.append(metrics.r2_score(sequence_true, sequence_predicted))
            mse.append(metrics.mean_squared_error(sequence_true,
                                                  sequence_predicted))
        return numpy.mean(r2s), numpy.mean(mse)
