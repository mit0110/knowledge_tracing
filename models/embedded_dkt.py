"""Model for sequence prediction with a LSTM RNN."""
import numpy
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops.math_ops import tanh
from quick_experiment.models import seq_lstm


class EmbeddedSeqLSTMModel(seq_lstm.SeqLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.
    """
    def __init__(self, dataset, name=None, hidden_layer_size=0, batch_size=None,
                 training_epochs=1000, logs_dirname='.', log_values=True,
                 max_num_steps=30, embedding_size=200, dropout_ratio=0.3,
                 **kwargs):
        super(EmbeddedSeqLSTMModel, self).__init__(
            dataset, batch_size=batch_size, training_epochs=training_epochs,
            logs_dirname=logs_dirname, name=name, log_values=log_values,
            dropout_ratio=dropout_ratio, hidden_layer_size=hidden_layer_size,
            max_num_steps=max_num_steps, **kwargs)
        self.embedding_size = embedding_size

    def _build_inputs(self):
        """Generate placeholder variables to represent the input tensors.

        In the case of the embeddings, we don't need to take the full one
        hot encoding but only the index of the input element, so we reduce
        one dimension of the input and labels placeholders."""
        # Placeholder for the inputs in a given iteration.
        self.instances_placeholder = tf.placeholder(
            tf.int32, (None, self.max_num_steps),
            name='sequences_placeholder')

        self.lengths_placeholder = tf.placeholder(
            tf.int32, (None, ), name='lengths_placeholder')

        self.labels_placeholder = tf.placeholder(
            tf.int32, (None, self.max_num_steps),
            name='labels_placeholder')

    def _get_embedding(self, element_ids, element_embeddings,
                       positive_embedding, element_only=False):
        """Returns self.element_embeddings + self.positive_embeddings if
        the element is positive, and self.element_embedding is is negative."""
        embedded_element = tf.nn.embedding_lookup(
            element_embeddings, tf.abs(element_ids), name='embedded_element')
        if element_only:
            return embedded_element
        embedded_outcome = tf.nn.embedding_lookup(
            positive_embedding,
            tf.clip_by_value(element_ids, clip_value_min=0,
                             clip_value_max=self.dataset.feature_vector_size),
            name='embedded_outcome')

        return tf.add_n([embedded_element, embedded_outcome],
                        name='full_embedding')

    def _build_input_layers(self):
        with tf.name_scope('embedding_layer') as scope:
            self.base_embedding = tf.Variable(
                tf.random_uniform([self.dataset.feature_vector_size,
                                   self.embedding_size], 0, 1.0),
                trainable=True, name='base_embedding')
            element_embeddings = tf.concat([
                tf.zeros([1, self.embedding_size]), self.base_embedding], 0)
            self.positive_embedding = tf.Variable(
                tf.random_uniform([self.dataset.feature_vector_size,
                                   self.embedding_size], 0, 1.0),
                trainable=True, name='positive_embedding')
            positive_embedding = tf.concat([tf.zeros([1, self.embedding_size]),
                                            self.positive_embedding], 0)
            input = self._get_embedding(self.instances_placeholder,
                                        element_embeddings,
                                        positive_embedding)
            if self.dropout_ratio != 0:
                return tf.layers.dropout(inputs=input, rate=self.dropout_ratio)
            return input

    def _build_loss(self, logits):
        """Calculates the average binary cross entropy.

        Args:
            logits: Tensor - [batch_size, max_num_steps, classes_num]
        """
        # Convert labels to one hot encoding mask to filter only the true next
        # exercise. We want the first position to represent the id 1,
        # and we want the zero elements to be set as zeros.
        labels_mask = tf.cast(
            tf.one_hot(indices=tf.abs(self.labels_placeholder) - 1,
                       depth=self.dataset.classes_num(), on_value=1,
                       off_value=0, axis=-1),
            logits.dtype)
        true_logits = tf.multiply(logits, labels_mask)
        # Now the only elements active are the ones with the logits for the
        # true next elements only.
        true_logits = tf.reduce_sum(true_logits, axis=2)

        probability_labels = tf.cast(
            tf.clip_by_value(tf.sign(self.labels_placeholder), 0, 1),
            true_logits.dtype)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=true_logits, labels=probability_labels)
        # loss has shape [batch_size, max_num_steps]
        # We set to 0 the losses of the predictions outside the sequence.
        mask = tf.sequence_mask(self.lengths_placeholder, self.max_num_steps)
        loss = tf.boolean_mask(loss, mask)
        # We sum all the losses in all the sequences and then divide by the
        # number of steps in the sequence (which may be zero)
        loss = seq_lstm.safe_div(
            tf.reduce_sum(loss),
            tf.cast(tf.reduce_sum(self.lengths_placeholder), loss.dtype))

        return loss

    def _build_predictions(self, logits):
        """Return a tensor with the predicted probability for each instance.

        The prediction is a vector with the probabilities of approving the
        next exercise.

        Args:
            logits: Logits tensor, float - [batch_size, max_num_steps,
                classes_num].

        Returns:
            A float64 tensor with the predictions, with shape [batch_size,
            max_num_steps].
        """
        # We project logits to the range (0, 1)
        logits = tf.nn.sigmoid(logits)
        labels_mask = tf.cast(
            tf.one_hot(indices=tf.abs(self.labels_placeholder) - 1,
                       depth=self.dataset.classes_num(), on_value=1,
                       off_value=0, axis=-1),
            logits.dtype)
        predictions = tf.multiply(logits, labels_mask)
        predictions = tf.reduce_max(predictions, axis=2)

        return predictions

    def _get_step_predictions(self, batch_prediction, batch_true, feed_dict):
        step_prediction = self.sess.run(self.predictions, feed_dict=feed_dict)
        labels = numpy.clip(numpy.sign(feed_dict[self.labels_placeholder]),
                            0, 1)
        for index, length in enumerate(feed_dict[self.lengths_placeholder]):
            batch_prediction[index] = numpy.append(
                batch_prediction[index], step_prediction[index, :length])
            batch_true[index] = numpy.append(
                batch_true[index], labels[index, :length])

    def _build_evaluation(self, logits):
        """Evaluate the quality of the logits at predicting the label.

        Args:
            logits: Logits tensor, float - [batch_size, max_num_steps,
                embedding_size].
        Returns:
            Two operations, where the first one gives the value of the
            mean squared error in the validation dataset.
        """
        predictions = self._build_predictions(logits)
        # predictions has shape [batch_size, max_num_steps]
        with tf.name_scope('evaluation_performance'):
            mask = tf.sequence_mask(
                self.lengths_placeholder, maxlen=self.max_num_steps,
                dtype=predictions.dtype)
            # We use the mask to ignore predictions outside the sequence length.
            true_labels = tf.clip_by_value(
                tf.cast(tf.sign(self.labels_placeholder), predictions.dtype),
                clip_value_min=0, clip_value_max=1)
            mse, mse_update = tf.contrib.metrics.streaming_mean_squared_error(
                predictions, true_labels, weights=mask)

        if self.logs_dirname is not None:
            tf.summary.scalar('evaluation_mse', mse)

        return mse, mse_update

    def write_embeddings(self, metadata_path):
        with self.graph.as_default():
            config = projector.ProjectorConfig()

            # Add base embedding
            embedding = config.embeddings.add()
            embedding.tensor_name = self.base_embedding.name
            # Link this tensor to its metadata file
            embedding.metadata_path = metadata_path

            # Add positive embedding
            embedding = config.embeddings.add()
            embedding.tensor_name = self.positive_embedding.name
            # Link this tensor to the same metadata file
            embedding.metadata_path = metadata_path

            # Saves a configuration file that TensorBoard will read
            # during startup.
            projector.visualize_embeddings(self.summary_writer, config)


class EmbeddedBasicLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    """BasicLSTMCell to transform the input before running the cell."""

    def __init__(self, num_units, forget_bias=1.0, input_size=None,
                 state_is_tuple=True, activation=tanh, reuse=None,
                 modifier_function=None):
        super(EmbeddedBasicLSTMCell, self).__init__(
            num_units, forget_bias=forget_bias, state_is_tuple=state_is_tuple,
            activation=activation, reuse=reuse)
        self.modifier_function = modifier_function

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).

        Args:
            inputs: `2-D` tensor with shape `[batch_size x input_size]`.
            state: An `LSTMStateTuple` of state tensors, each shaped
                `[batch_size x self.state_size]`, if `state_is_tuple` has been
                set to `True`.  Otherwise, a `Tensor` shaped
                `[batch_size x 2 * self.state_size]`.
        Returns:
            A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        if self._state_is_tuple:
            c, h = state
        else:
            raise ValueError('EmbeddedBasicLSTMCell must use a state tuple')
        # if self.modifier_function is not None:
        inputs = tf.abs(tf.subtract(inputs, h))
        return super(EmbeddedBasicLSTMCell, self).call(inputs, state)


class CoEmbeddedSeqLSTMModel(EmbeddedSeqLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.

    The embedded layer is combined with the hidden state of the recurrent
    network before entering the hidden layer.
    """
    def __init__(self, dataset, name=None, hidden_layer_size=0, batch_size=None,
                 training_epochs=1000, logs_dirname='.', log_values=True,
                 max_num_steps=30, embedding_size=200, dropout_ratio=0.3,
                 **kwargs):
        super(CoEmbeddedSeqLSTMModel, self).__init__(
            dataset, batch_size=batch_size, training_epochs=training_epochs,
            logs_dirname=logs_dirname, name=name, log_values=log_values,
            dropout_ratio=dropout_ratio, hidden_layer_size=hidden_layer_size,
            max_num_steps=max_num_steps, embedding_size=embedding_size,
            **kwargs)

    def _build_rnn_cell(self):
        return EmbeddedBasicLSTMCell(self.hidden_layer_size, forget_bias=1.0)

    def _build_recurrent_layer(self):
        # The recurrent layer
        input = self._build_input_layers()
        rnn_cell = self._build_rnn_cell()
        with tf.name_scope('recurrent_layer') as scope:
            # Get the initial state. States will be a LSTMStateTuples.
            state_variable = self._build_state_variables(rnn_cell)
            # outputs is a Tensor shaped [batch_size, max_time,
            # cell.output_size].
            # State is a Tensor shaped [batch_size, cell.state_size]
            outputs, new_state = tf.nn.dynamic_rnn(
                rnn_cell, inputs=input,
                sequence_length=self.lengths_placeholder, scope=scope,
                initial_state=state_variable)
            # Define the state operations. This wont execute now.
            self.last_state_op = self._get_state_update_op(state_variable,
                                                           new_state)
            self.reset_state_op = self._get_state_update_op(
                state_variable,
                rnn_cell.zero_state(self.batch_size, tf.float32))
        return outputs
