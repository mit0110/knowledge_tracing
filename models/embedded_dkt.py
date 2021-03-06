"""Model for sequence prediction with a LSTM RNN."""
import numpy
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
from quick_experiment.models import seq_lstm


class EmbeddedSeqLSTMModel(seq_lstm.SeqLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.
    """
    def __init__(self, dataset, embedding_size=200, embedding_model=None,
                 finetune_embeddings=True, **kwargs):
        super(EmbeddedSeqLSTMModel, self).__init__(dataset, **kwargs)
        self.embedding_size = embedding_size
        self.embedding_model = embedding_model
        self.finetune_embeddings = finetune_embeddings

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

        self.dropout_placeholder = tf.placeholder_with_default(
            0.0, shape=(), name='dropout_placeholder')

    def _pad_batch(self, input_tensor):
        self.current_batch_size = tf.shape(input_tensor)[0]
        new_instances = tf.subtract(self.batch_size, tf.shape(input_tensor)[0])
        # Pad lenghts
        self.batch_lengths = tf.pad(self.lengths_placeholder,
                                    paddings=[[tf.constant(0), new_instances]],
                                    mode='CONSTANT')
        # Pad instances
        paddings = [[tf.constant(0), new_instances], tf.constant([0, 0])]
        input_tensor = tf.pad(input_tensor, paddings=paddings, mode='CONSTANT')
        # Ensure the correct shape. This is only to avoid an error with the
        # dynamic_rnn, which needs to know the size of the batch.
        return tf.reshape(
            input_tensor, shape=(self.batch_size, self.max_num_steps))

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
        input = self._pad_batch(self.instances_placeholder)
        if self.embedding_model is not None:
            embedding_matrix = self.embedding_model.wv.syn0
            # https://github.com/dennybritz/cnn-text-classification-tf/issues/17
            self.embedding_placeholder = tf.placeholder_with_default(
                embedding_matrix, shape=embedding_matrix.shape,
                name='embedding_placeholder')
            self.base_embedding = tf.Variable(tf.random_uniform(
                embedding_matrix.shape, -1.0, 1.0),
                name='input_embedding_var', trainable=self.finetune_embeddings)
            self.embedding_init = self.base_embedding.assign(
                self.embedding_placeholder)
            # We add the embedding for the zero element, which SHOULD be the
            # padding element, and the embedding for the OOV element.
            element_embeddings = tf.concat([
                tf.zeros([1, self.embedding_size]),
                self.base_embedding,
                tf.random_uniform([1, self.embedding_size], -1.0, 1.0)
            ], 0)
        else:
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
        input = self._get_embedding(input, element_embeddings,
                                    positive_embedding)

        if self.dropout_ratio != 0:
            return tf.layers.dropout(inputs=input,
                                     rate=self.dropout_placeholder)
        return input

    def build_all(self):
        super(EmbeddedSeqLSTMModel, self).build_all()
        if self.embedding_model is not None:
            with self.graph.as_default():
                self.sess.run([self.embedding_init])

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
        predictions = tf.reduce_max(predictions, axis=2, name='predictions')

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

    def _build_evaluation(self, predictions):
        """Evaluate the quality of the logits at predicting the label.

        Args:
            predictions: Logits tensor, float - [current_batch_size,
                                                 max_num_steps].
        Returns:
            Two operations, where the first one gives the value of the
            mean squared error in the validation dataset.
        """
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


class EmbeddedSeqGRUModel(EmbeddedSeqLSTMModel):
    """A Recurrent Neural Network model with GRU cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.

    The embedded layer is combined with the hidden state of the recurrent
    network before entering the hidden layer.
    """
    def _build_rnn_cell(self):
        return tf.contrib.rnn.GRUCell(self.hidden_layer_size)

    def _build_state_variables(self, cell):
        # Get the initial state and make a variable out of it
        # to enable updating its value.
        state = cell.zero_state(self.batch_size, tf.float32)
        return tf.Variable(state, trainable=False)

    @staticmethod
    def _get_state_update_op(state_variables, new_state):
        # Add an operation to update the train states with the last state
        # Assign the new state to the state variables on this layer
        return state_variables.assign(new_state)


class EmbeddedSeqRNNModel(EmbeddedSeqGRUModel):
    """A Recurrent Neural Network model with GRU cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.

    The embedded layer is combined with the hidden state of the recurrent
    network before entering the hidden layer.
    """
    def _build_rnn_cell(self):
        return tf.contrib.rnn.BasicRNNCell(self.hidden_layer_size)


class EmbeddedBasicLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    """BasicLSTMCell to transform the input before running the cell."""

    def __init__(self, num_units, *args, modifier_function=None, **kwargs):
        super(EmbeddedBasicLSTMCell, self).__init__(
            num_units, *args, **kwargs)
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
        if self.modifier_function is not None:
            inputs = self.modifier_function(inputs, h)
        else:
            inputs = tf.abs(tf.subtract(inputs, h))
        return super(EmbeddedBasicLSTMCell, self).call(inputs, state)


class CoEmbeddedSeqLSTMModel(EmbeddedSeqLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.

    The embedded layer is combined with the hidden state of the recurrent
    network before entering the hidden layer.
    """
    def __init__(self, dataset, hidden_layer_size=200, **kwargs):
        kwargs.pop('embedding_size')
        super(CoEmbeddedSeqLSTMModel, self).__init__(
            dataset, hidden_layer_size=hidden_layer_size,
            embedding_size=hidden_layer_size, **kwargs)

    def _build_rnn_cell(self):
        return EmbeddedBasicLSTMCell(self.hidden_layer_size, forget_bias=1.0)


class CoEmbeddedSeqLSTMModel2(CoEmbeddedSeqLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.

    The embedded layer is combined with the hidden state of the recurrent
    network before entering the hidden layer. The embedding_size will be the
    same as the hidden layer size.
    """

    def _build_rnn_cell(self):
        return EmbeddedBasicLSTMCell(
            self.hidden_layer_size, forget_bias=1.0,
            modifier_function=lambda i, h: tf.square(tf.subtract(i, h)))


class CoEmbeddedSeqLSTMModel3(CoEmbeddedSeqLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.

    The embedded layer is combined with the hidden state of the recurrent
    network before entering the hidden layer. The embedding_size will be the
    same as the hidden layer size.
    """

    def _build_rnn_cell(self):
        # We define a new variable for the standard deviation of the normal
        # distribution
        std_var = tf.Variable(1.0, name='normal_std', trainable=True)
        tf.summary.scalar('normal_std', std_var)
        dist = tf.distributions.Normal(loc=0.0, scale=std_var)

        def modifier_function(input, state):
            return dist.prob(tf.subtract(input, state))

        return EmbeddedBasicLSTMCell(
            self.hidden_layer_size, forget_bias=1.0,
            modifier_function=modifier_function)


class CoEmbeddedSeqLSTMModel4(CoEmbeddedSeqLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.

    The embedded layer is combined with the hidden state of the recurrent
    network before entering the hidden layer. The embedding_size will be the
    same as the hidden layer size.
    """

    def _build_rnn_cell(self):
        # We define a new variable for the standard deviation of the normal
        # distribution
        dist = tf.distributions.Normal(loc=0.0, scale=1.0)

        def modifier_function(input, state):
            return dist.prob(tf.subtract(input, state))

        return EmbeddedBasicLSTMCell(
            self.hidden_layer_size, modifier_function=modifier_function)


class CoEmbeddedSeqLSTMModel5(CoEmbeddedSeqLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.

    The embedded layer is combined with the hidden state of the recurrent
    network before entering the hidden layer. The embedding_size will be the
    same as the hidden layer size.
    """

    def _build_rnn_cell(self):
        return EmbeddedBasicLSTMCell(
            self.hidden_layer_size,
            modifier_function=lambda i, h: tf.tanh(tf.subtract(i, h)))


class CoEmbeddedSeqLSTMModel6(CoEmbeddedSeqLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.

    The embedded layer is combined with the hidden state of the recurrent
    network before entering the hidden layer. The embedding_size will be the
    same as the hidden layer size.
    """

    def _build_rnn_cell(self):
        return EmbeddedBasicLSTMCell(
            self.hidden_layer_size,
            modifier_function=lambda i, h: tf.sigmoid(tf.subtract(i, h)))


class EmbeddedBasicGRUCell(tf.contrib.rnn.GRUCell):
    """BasicLSTMCell to transform the input before running the cell."""

    def __init__(self, num_units, *args, modifier_function=None, **kwargs):
        super(EmbeddedBasicGRUCell, self).__init__(
            num_units, *args, **kwargs)
        self.modifier_function = modifier_function

    def call(self, inputs, state):
        if self.modifier_function is not None:
            inputs = self.modifier_function(inputs, state)
        else:
            inputs = tf.abs(tf.subtract(inputs, state))
        return super(EmbeddedBasicGRUCell, self).call(inputs, state)


class CoEmbeddedSeqGRUModel(EmbeddedSeqGRUModel, CoEmbeddedSeqLSTMModel):
    """A Recurrent Neural Network model with GRU cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.

    The embedded layer is combined with the hidden state of the recurrent
    network before entering the hidden layer.
    """
    def _build_rnn_cell(self):
        return EmbeddedBasicGRUCell(self.hidden_layer_size)


class EmbeddedBasicRNNCell(tf.contrib.rnn.BasicRNNCell):
    """BasicLSTMCell to transform the input before running the cell."""

    def __init__(self, num_units, *args, modifier_function=None, **kwargs):
        super(EmbeddedBasicRNNCell, self).__init__(num_units, *args, **kwargs)
        self.modifier_function = modifier_function

    def call(self, inputs, state):
        if self.modifier_function is not None:
            inputs = self.modifier_function(inputs, state)
        else:
            inputs = tf.abs(tf.subtract(inputs, state))
        return super(EmbeddedBasicRNNCell, self).call(inputs, state)


class CoEmbeddedSeqRNNModel(EmbeddedSeqGRUModel, CoEmbeddedSeqLSTMModel):
    """A Recurrent Neural Network model with LSTM cells.

    Predicts the probability of the next element on the sequence. The
    input is first passed by an embedding layer to reduce dimensionality.

    The embedded layer is combined with the hidden state of the recurrent
    network before entering the hidden layer. The embedding_size will be the
    same as the hidden layer size.
    """

    def _build_rnn_cell(self):
        return EmbeddedBasicRNNCell(self.hidden_layer_size)


class EmbeddedBiLSTMModel(EmbeddedSeqLSTMModel, seq_lstm.SeqBiLSTMModel):
    pass


class CoEmbeddedBiLSTMModel(CoEmbeddedSeqLSTMModel, seq_lstm.SeqBiLSTMModel):
    def _build_rnn_cell(self):
        dist = tf.distributions.Normal(loc=0.0, scale=1.0)

        def modifier_function(input, state):
            return dist.prob(tf.subtract(input, state))

        return (
            EmbeddedBasicLSTMCell(self.hidden_layer_size,
                                  modifier_function=modifier_function),
            tf.contrib.rnn.BasicLSTMCell(self.hidden_layer_size)
        )
