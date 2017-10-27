import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer, batch_norm
from tensorflow.contrib.framework import arg_scope
initializer = tf.truncated_normal_initializer(stddev=0.02)

def conv_layer(x, filter_size, kernel, stride=2, padding="VALID", layer_name="conv"):
    with tf.variable_scope(layer_name):
        if padding == 1 :
            x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]])
            x = tf.layers.conv2d(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=initializer, strides=stride, use_bias=False)
        else :
            x = tf.layers.conv2d(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=initializer, strides=stride, padding=padding, use_bias=False)

        return x


def deconv_layer(x, filter_size, kernel, stride=2, padding="VALID", layer_name="deconv") :
    with tf.variable_scope(layer_name):
        if padding == 1 :
            x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]])
            x = tf.layers.conv2d_transpose(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=initializer, strides=stride, use_bias=False)
        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=filter_size, kernel_size=kernel, kernel_initializer=initializer, strides=stride, padding=padding, use_bias=False)


        return x

def Batch_Normalization(x, training, scope='batch_norm'):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def tanh(x):
    return tf.tanh(x)

def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.sigmoid(x)

def swish(x): # may be it will be test
    return x * tf.sigmoid(x)


