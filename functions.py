

import tensorflow.compat.v1 as tf

#__________________________________________

#

# Convolutional layer with bias and a particular initialization
def tfconv(layer_number, givesize, take, conv_window, stride):
    with tf.variable_scope('conv' + str(layer_number)):
	    #givesize: number of nodes in the next layer
	    #take: thing outputted by previous layer
	    
	    init = tf.truncated_normal_initializer(stddev=0.02)
	    takesize = take.shape[-1]	

	    W_conv = tf.get_variable("W_conv" + str(layer_number), shape=[conv_window, conv_window, takesize, givesize], initializer=init)
	    b_conv = tf.get_variable("b_conv" + str(layer_number), shape=[givesize], initializer=tf.constant_initializer(0.0))

	    _c = tf.nn.conv2d(take, W_conv, strides=[1, stride, stride, 1], padding='SAME')
	    c = tf.nn.bias_add(_c, b_conv)

	    return c

# leaky relu
def lrelu(x):
    return tf.maximum(x, 0.2*x)

# batch normalization
def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)                                      
#
   
# Dense a.k.a. fully connected layer
def tfdense(layer_number, imsize, givesize, take, resize, scope):
    with tf.variable_scope(scope or "Linear"):
        take_size = take.shape[-1]	
        shape_in = take_size
	    
        if resize:
            shape_in = shape_in*imsize*imsize
            take = tf.reshape(take, [-1, shape_in])
	    
        W_fc = tf.get_variable("W_fc" + str(layer_number), shape=[shape_in, givesize],
		                 initializer = tf.random_normal_initializer(stddev=0.02))
        b_fc = tf.get_variable("b_fc"+str(layer_number), shape=[givesize])
	    
		
        h_fc = tf.matmul(take, W_fc) + b_fc
	    
        return h_fc

#
       
# Upsampling a.k.a. deconvolution a.k.a. transposed convolution layer
def deconv(layer_number, givesize, take, conv_window, stride):
    with tf.variable_scope('dc' + str(layer_number)):
	    #givesize: number of nodes in the next layer
	    #take: thing outputted by previous layer
	      
	    init = tf.truncated_normal_initializer(stddev=0.02)

	    new_height = take.shape[1]*stride
	    new_width = take.shape[2]*stride	

	    W_conv = tf.get_variable("W_conv" + str(layer_number), shape=[conv_window, conv_window, givesize, take.shape[-1]], initializer=init)
	    b_conv = tf.get_variable("b_conv" + str(layer_number), shape=[givesize], initializer=tf.constant_initializer(0.0))

	    _c = tf.nn.conv2d_transpose(take, W_conv, output_shape=[int(take.shape[0]),
			int(new_height), int(new_width), givesize], strides=[1, stride, stride, 1], padding='SAME')
	    c = tf.nn.bias_add(_c, b_conv)

	    return c



