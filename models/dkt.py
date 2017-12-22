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
        # Labels can be 1 or -1, we will replace the -1 with p(1)=0.
        labels = tf.cast(self.labels_placeholder, logits.dtype)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.clip_by_value(labels, 0, 1))
        # loss has shape [batch_size, max_num_steps, classes_num]
        # We set to 0 the losses of the predictions of exercises that are not
        # the next one.
        loss = tf.multiply(loss, tf.abs(labels))
        loss = tf.div(
            tf.reduce_sum(tf.boolean_mask(loss, mask)),
            tf.cast(tf.reduce_sum(self.lengths_placeholder), loss.dtype))

        if self.logs_dirname:
            tf.summary.scalar('train_loss', loss)

        return loss

    def _build_predictions(self, logits):
        """Return a tensor with the predicted performance of next exercise.

        The prediction for each step is float with the probability of the
        next exercise being correct. To know the true next exercise we use
        the id of the exercise that is not 0 from self.labels_placeholder

        Args:
            logits: Logits tensor, float - [batch_size, max_num_steps,
                num_classes].

        Returns:
            A float64 tensor with the predictions, with shape [batch_size,
            max_num_steps].
        """
        predictions = tf.nn.sigmoid(logits)
        # We leave only the predictions for the true next exercise.
        # We use the fact that labels_placeholder can be 1 or -1 for the next
        # exercise.
        predictions = tf.multiply(
            predictions,
            tf.cast(tf.abs(self.labels_placeholder), predictions.dtype))
        # We keep only the predictions that are not 0. Should be only one per
        # step because labels_placeholder is a one hot encoding.
        predictions = tf.reduce_max(predictions, axis=2)
        return predictions

    def _get_batch_prediction(self, partition_name):
        true = []
        predictions = []
        lengths = numpy.zeros(self.batch_size)
        for feed_dict in self._fill_feed_dict(partition_name, reshuffle=False):
            feed_dict[self.dropout_placeholder] = 0
            step_prediction = self.sess.run(self.predictions,
                                            feed_dict=feed_dict)
            # each true has shape [batch_size, max_num_step, num_classes]
            # we take the maximum value per time step, which will give us 1
            # if the (only) exercise was solved correctly or 0 if not.
            true.append(numpy.amax(feed_dict[self.labels_placeholder], axis=2))
            # each prediction has shape [batch_size, max_num_step]
            predictions.append(step_prediction)
            lengths += feed_dict[self.lengths_placeholder]
        predictions = numpy.hstack(predictions)
        true = numpy.hstack(true)
        assert predictions.shape[0] == self.batch_size
        assert predictions.shape == true.shape
        assert predictions.shape[1] >= lengths.max()

        short_predictions = []
        short_true = []
        for index, length in enumerate(lengths):
            short_predictions.append(predictions[index, :int(length)])
            short_true.append(true[index, :int(length)])
        return short_true, short_predictions

    def predict(self, partition_name, limit=-1):
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
            while (self.dataset.has_next_batch(self.batch_size, partition_name)
                   and (limit <= 0 or len(predictions) < limit)):
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
            A scalar float32 tensor with the mean squared error.
        """
        predictions = self._build_predictions(logits)
        # predictions has shape [batch_size, max_num_steps]
        with tf.name_scope('evaluation_performance'):
            mask = tf.sequence_mask(
                self.lengths_placeholder, maxlen=self.max_num_steps,
                dtype=predictions.dtype)
            # We use the mask to ignore predictions outside the sequence length.
            labels = tf.cast(tf.reduce_max(
                    self.labels_placeholder, axis=2), predictions.dtype)

            mse, mse_update = tf.contrib.metrics.streaming_mean_squared_error(
                predictions, labels, weights=mask)

        if self.logs_dirname:
            tf.summary.scalar('eval_mse', mse)
            tf.summary.scalar('eval_up_mse', mse_update)

        return mse, mse_update

    def evaluate(self, partition='validation'):
        with self.graph.as_default():
            # Reset the metric variables
            stream_vars = [i for i in tf.local_variables()
                           if i.name.split('/')[0] == 'evaluation_performance']
            mse, mse_update = self.evaluation_op
            self.dataset.reset_batch()
            mse_value = None
            self.sess.run([tf.variables_initializer(stream_vars)])
            while self.dataset.has_next_batch(self.batch_size, partition):
                for feed_dict in self._fill_feed_dict(partition,
                                                      reshuffle=False):
                    feed_dict[self.dropout_placeholder] = 0
                    self.sess.run([mse_update], feed_dict=feed_dict)
                mse_value = self.sess.run([mse])[0]

        return mse_value

