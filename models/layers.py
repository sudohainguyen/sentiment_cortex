import math
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Layer


__all__ = [
    'gelu', 'ScaledDotProductAttention', 'Sinusoidal_Position_Embedding',
    'ModMultiHeadAttention', 'LayerNormalization', 'FeedForward', 'binary_focal_loss'
]


def gelu(x):
    inner_tanh = tf.square(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))
    cdf = 0.5 * (1.0 + tf.tanh(inner_tanh))
    return x * cdf


class Sinusoidal_Position_Embedding(Layer):
    """
    Represents trainable positional embeddings for the Transformer model:
    word position embeddings - one for each position in the sequence.
    """

    def __init__(self, size=None, mode='concat', **kwargs):
        self.mode = mode
        self.size = size
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['mode'] = self.mode
        config['size'] = self.size
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(input_shape[-1])
        super().build(input_shape)
        
    def compute_output_shape(self, input_shape):
        print('Position_Embedding : ', input_shape)
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)

    def call(self, inputs, **kwargs):
        
        # batch_size, seq_len = K.shape(inputs)[0], K.shape(inputs)[1]

        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(inputs[:, :, 0]), 1) - 1 
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)

        if self.mode == 'sum':
            result = position_ij + inputs
        elif self.mode == 'concat':
            result = K.concatenate([inputs, position_ij], 2)
        
        return result


class ModMultiHeadAttention(keras.layers.Layer):

    def __init__(self,
                 head_num,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='glorot_normal',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 normalized=False,
                 **kwargs):
        """Initialize the layer.
        :param head_num: Number of heads.
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """
        self.supports_masking = True
        self.head_num = head_num
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.history_only = history_only
        self.normalized = normalized

        # self.Wq, self.Wk, self.Wv, self.Wo = None, None, None, None
        # self.bq, self.bk, self.bv, self.bo = None, None, None, None
        self.Wk, self.Wv, self.Wo = None, None, None
        self.bk, self.bv, self.bo = None, None, None
        super(ModMultiHeadAttention, self).__init__(**kwargs)

      
    
    def get_config(self):
        config = {'head_num': self.head_num,
                  'activation': keras.activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
                  'bias_initializer': keras.initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
                  'bias_constraint': keras.constraints.serialize(self.bias_constraint),
                  'history_only': self.history_only,
                  'normalized':self.normalized,
                  }
        base_config = super(ModMultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

    
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
            return q[:-1] + (v[-1],)  #[input[0](0,1) + input[0](2)] = input(allshape)
        return input_shape

      
      
    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

      
      
    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = v[-1]
        
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        
        # self.Wq = self.add_weight(shape=(q[-1], feature_dim),
        #                           initializer=self.kernel_initializer,
        #                           regularizer=self.kernel_regularizer,
        #                           constraint=self.kernel_constraint,
        #                           name='%s_Wq' % self.name,
        #                           )
        # if self.use_bias:
        #     self.bq = self.add_weight(shape=(feature_dim,),
        #                               initializer=self.bias_initializer,
        #                               regularizer=self.bias_regularizer,
        #                               constraint=self.bias_constraint,
        #                               name='%s_bq' % self.name,
        #                               )
        
        self.Wk = self.add_weight(shape=(k[-1], feature_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  name='%s_Wk' % self.name,
                                  )  
        if self.use_bias:
            self.bk = self.add_weight(shape=(feature_dim,),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint,
                                      name='%s_bk' % self.name,
                                      )
        
        self.Wv = self.add_weight(shape=(v[-1], feature_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  name='%s_Wv' % self.name,
                                  )
        if self.use_bias:
            self.bv = self.add_weight(shape=(feature_dim,),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint,
                                      name='%s_bv' % self.name,
                                      )
        
        self.Wo = self.add_weight(shape=(feature_dim, feature_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  name='%s_Wo' % self.name,
                                  )
        if self.use_bias:
            self.bo = self.add_weight(shape=(feature_dim,),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint,
                                      name='%s_bo' % self.name,
                                      )
        super(ModMultiHeadAttention, self).build(input_shape)

        
    
    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        
        head_dim = feature_dim // head_num
        
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))

      
      
    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))

      
      
    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, K.stack([1, head_num, 1]))
        return K.reshape(mask, (-1, seq_len))

      
      
    def call(self, inputs, mask=None):
        
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
            
            
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
            
            
        # q = K.dot(q, self.Wq)
        q = q
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            # q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            # q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
            
        y = ScaledDotProductAttention(history_only=self.history_only, normalized = self.normalized,name='%s-Attention' % self.name)(
            inputs=[
                    self._reshape_to_batches(q, self.head_num),
                    self._reshape_to_batches(k, self.head_num),
                    self._reshape_to_batches(v, self.head_num),
                    ],
            mask=[
                  self._reshape_mask(q_mask, self.head_num),
                  self._reshape_mask(k_mask, self.head_num),
                  self._reshape_mask(v_mask, self.head_num),
                 ],
        )
        
        y = self._reshape_from_batches(y, self.head_num)
        
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)
        return y


class ScaledDotProductAttention(keras.layers.Layer):

  
    def __init__(self, return_attention=False, history_only=False, normalized = False,**kwargs):
        """Initialize the layer.
        :param return_attention: Whether to return attention weights.
        :param history_only: Whether to only use history data.
        :param kwargs: Arguments for parent class.
        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only
        self.normalized = normalized
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
                  'return_attention': self.return_attention,
                  'history_only': self.history_only,
                  'normalized': self.normalized,
                 }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
      
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask
      
    def call(self, inputs, mask=None, **kwargs):
        
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        
        if isinstance(mask, list):
            mask = mask[1]
        
        e = K.batch_dot(query, key, axes=2)
        
        if self.normalized:
            feature_dim = K.shape(query)[-1]
            e = e/ K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        
        if self.history_only: #forward attention
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            ones = tf.ones((query_len, key_len))
            e -= (ones - tf.matrix_band_part(ones, -1, 0)) * 1e9
        
        if mask is not None:
            e -= (1.0 - K.cast(K.expand_dims(mask, axis=-2), K.floatx())) * 1e9
        
        a = keras.activations.softmax(e)
        v = K.batch_dot(a, value)
        
        if self.return_attention:
            return [v, a]
        
        return v


class LayerNormalization(keras.layers.Layer):
    
    def __init__(self,
                 center=False,
                 scale=False,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

        
        
    def get_config(self):
        config = {'center': self.center,
                  'scale': self.scale,
                  'epsilon': self.epsilon,
                  'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
                  'beta_initializer': keras.initializers.serialize(self.beta_initializer),
                  'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
                  'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
                  'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
                  'beta_constraint': keras.constraints.serialize(self.beta_constraint),
                  }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

      
      
    def compute_output_shape(self, input_shape):
        return input_shape

      
      
    def compute_mask(self, inputs, input_mask=None):
        return input_mask

      
      
    def build(self, input_shape):
        self.input_spec = keras.engine.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint,
                                         name='gamma',
                                         )
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        name='beta',
                                        )
        super(LayerNormalization, self).build(input_shape)

        
        
    def call(self, inputs, training=None):
      
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        
        std = K.sqrt(variance + self.epsilon)
        
        outputs = (inputs - mean) / std
        
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        
        return outputs


class FeedForward(keras.layers.Layer):
    """Position-wise feed-forward layer. 
    # Arguments
        units: int >= 0. Dimension of hidden units.
        activation: Activation function to use
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout_rate: 0.0 <= float <= 1.0. Dropout rate for hidden units.
    # Input shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # Output shape
        3D tensor with shape: `(batch_size, ..., input_dim)`.
    # References
        - [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(self,
                 ratio=4,
                 activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 outputdim=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 dropout_rate=0.0,
                 **kwargs):
        self.ratio = ratio
        self.supports_masking = True
        self.units = None
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.dropout_rate = dropout_rate
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        self.outputdim = outputdim
        super(FeedForward, self).__init__()

    def get_config(self):
        config = {
            'ratio': self.ratio,
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if (self.outputdim == None):
            return input_shape
        else:
            return input_shape[:-1] + tuple([self.outputdim])

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        if (self.outputdim == None):
            outputdim = feature_dim
        else:
            outputdim = self.outputdim
        
        if(self.units == None):
          self.units = int(feature_dim // self.ratio)
        
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             constraint=self.kernel_constraint,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b1'.format(self.name),
            )


        self.W2 = self.add_weight(
            shape=(self.units, outputdim),
            initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             constraint=self.kernel_constraint,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(outputdim,),
                initializer=self.bias_initializer,
#                 regularizer=self.bias_regularizer,
#                 constraint=self.bias_constraint,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        h = K.dot(x, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        if 0.0 < self.dropout_rate < 1.0:
            def dropped_inputs():
                return K.dropout(h, self.dropout_rate, K.shape(h))
            h = K.in_train_phase(dropped_inputs, h, training=training)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y


def binary_focal_loss(gamma=2., alpha=.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return binary_focal_loss_fixed
