"""Definition of the DKT LSTM network"""
import numpy
import tensorflow as tf

from quick_experiment.models import seq_lstm


class DktLSTMModel(seq_lstm.SeqLSTMModel):

    def _build_predictions(self, logits):
        """Return a tensor with the predicted performance of next exercise.

        The prediction for each step is float with the probability of the
        next exercise being correct. To know the true next execise we use
        the information from self.labels_placeholder

        Args:
            logits: Logits tensor, float - [batch_size, max_num_steps,
                NUM_CLASSES (feature_vector_size + 1)].

        Returns:
            A float64 tensor with the predictions, with shape [batch_size,
            max_num_steps].
        """
        # The elements of self.labels_placeholder contain the result of the
        # next exercise in the same position as the exercise id. The result
        # value can only be 0 or 1.
        # We multiply the predictions by the labels placeholder to filter out
        # the predictions for exercises that are not the next one.
        predictions = tf.reduce_max(tf.multiply(
            logits, tf.cast(self.labels_placeholder, logits.dtype)),
            axis=2, name='predictions')
        return predictions

    def _get_step_predictions(self, batch_prediction, batch_true, feed_dict):
        step_prediction = self.sess.run(self.predictions, feed_dict=feed_dict)
        labels = numpy.amax(feed_dict[self.labels_placeholder], axis=2)
        assert labels.shape == step_prediction.shape
        for index, length in enumerate(feed_dict[self.lengths_placeholder]):
            batch_prediction[index] = numpy.append(
                batch_prediction[index], step_prediction[index, :length])
            batch_true[index] = numpy.append(
                batch_true[index], labels[index, :length])

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
        # The logic is changed in _get_step_predictions
        return super(DktLSTMModel, self).predict(partition_name)

    def _build_evaluation(self, logits):
        """Evaluate the quality of the logits at predicting the label.

        Args:
            logits: Logits tensor, float - [batch_size, max_num_steps,
                feature_vector + 1].
        Returns:
            A scalar int32 tensor with the number of examples (out of
            batch_size) that were predicted correctly.
        """
        predictions = self._build_predictions(logits)
        # predictions has shape [batch_size, max_num_steps]
        with tf.name_scope('evaluation_r2'):
            mask = tf.sequence_mask(
                self.lengths_placeholder, maxlen=self.max_num_steps,
                dtype=predictions.dtype)
            # We use the mask to ignore predictions outside the sequence length.
            r2, r2_update = tf.contrib.metrics.streaming_pearson_correlation(
                predictions, tf.cast(tf.reduce_max(
                    self.labels_placeholder, axis=2), predictions.dtype),
                weights=mask)

        return r2, r2_update

    def evaluate_validation(self, correct_predictions):
        partition = 'validation'
        # Reset the metric variables
        stream_vars = [i for i in tf.local_variables()
                       if i.name.split('/')[0] == 'evaluation_r2']
        r2_op, r2_update_op = correct_predictions
        self.dataset.reset_batch()
        r2_value = None
        self.sess.run([tf.variables_initializer(stream_vars)])
        while self.dataset.has_next_batch(self.batch_size, partition):
            for feed_dict in self._fill_feed_dict(partition, reshuffle=False):
                self.sess.run([r2_update_op], feed_dict=feed_dict)
            r2_value = self.sess.run([r2_op])[0]
        return r2_value
