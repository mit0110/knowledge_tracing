"""Definition of the DKT LSTM network"""
import numpy
import tensorflow as tf

from quick_experiment.models import seq_lstm


class DktLSTMModel(seq_lstm.SeqLSTMModel):

    def _build_loss(self, logits):
        """Calculates the avg binary cross entropy using the sigmoid function.

        Args:
            logits: Tensor - [batch_size, max_num_steps, classes_num]
        """
        mask = tf.sequence_mask(self.lengths_placeholder, self.max_num_steps)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=self.labels_placeholder)
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
