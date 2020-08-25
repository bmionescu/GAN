
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.misc

import glob
import re
import random
import os, gzip

# Non-layer functions
#____________________________________________

# Sorts the list of paths that glob returns
def globsorter(array):
	sorted_array,interim=[],[]
	for i in range(0,len(array)):
		temp=int(re.findall('(\d{1,10})',array[i])[0])
		interim+=[[temp,array[i]]]
	interim.sort(key=lambda x: x[0])
	sorted_array=(np.asarray(interim))[:,1].tolist()

	return sorted_array

 
# Loads the images from two folders
def load(path1,path2,image_size,channels):
	X_train,y_train = [],[] # Training data, training labels
	store1,store2=[],[] # keeps the original images to check the data
	_glob1=glob.glob(path1)
	_glob2=glob.glob(path2)

	glob1,glob2=globsorter(_glob1),globsorter(_glob2)
	for i in range(0,len(glob1)):
		if channels==3:
			img=cv2.resize(cv2.imread(glob1[i]),(image_size,image_size)).astype(np.float32)/255
		else:
			img=cv2.resize(cv2.imread(glob1[i],0),(image_size,image_size)).astype(np.float32)/255
		X_train+=[img.reshape(image_size*image_size*channels)]
		store1+=[img]

	for i in range(0,len(glob2)):
		if channels==3:
			img=cv2.resize(cv2.imread(glob2[i]),(image_size,image_size)).astype(np.float32)/255
		else:
			img=cv2.resize(cv2.imread(glob2[i],0),(image_size,image_size)).astype(np.float32)/255
		y_train+=[img.reshape(image_size*image_size*channels)]
		store2+=[img]

	X_train = np.array(X_train, dtype=np.float32)
	y_train = np.array(y_train, dtype=np.float32)

	return X_train, y_train

# Grabs sample images from a folder
def test(path,batch_size,image_size,channels):

	_globb=glob.glob(path)
	globb=globsorter(_globb)

	r=random.sample(range(len(globb)), 1)[0]

	if channels==3:
		im=cv2.resize(cv2.imread(globb[r]), (image_size,image_size), interpolation = cv2.INTER_AREA)
	else:
		im=cv2.resize(cv2.imread(globb[r],0), (image_size,image_size), interpolation = cv2.INTER_AREA)

	bucket=[]
	for i in range(0,batch_size):
		bucket+=[im.reshape(image_size*image_size*channels)]

	bucket=np.asarray(bucket)
	return bucket, r

# Makes the output of the network possible to view
def result_filter(images,image_size):
    bucket=[]
    for image in images:
    	output=np.zeros((image_size,image_size))
    	for i in range(0,image_size):
    		for j in range(0,image_size):
    			output[i][j]=image[i][j][0]
    	bucket+=[output]
    return np.asarray(bucket)


# # # ______ Layer functions _______ # # #

def tfconv(layer_number,givesize,take,conv_window,stride):

    with tf.variable_scope('conv'+str(layer_number)):
	    #givesize: number of nodes in the next layer
	    #take: thing outputted by previous layer
	    
	    init=tf.truncated_normal_initializer(stddev=0.02)
	    takesize=take.shape[-1]	

	    W_conv=tf.get_variable("W_conv"+str(layer_number),shape=[conv_window,conv_window,takesize,givesize],initializer=init)
	    b_conv=tf.get_variable("b_conv"+str(layer_number),shape=[givesize],initializer=tf.constant_initializer(0.0))

	    _c=tf.nn.conv2d(take,W_conv,strides=[1,stride,stride,1],padding='SAME')
	    c=tf.nn.bias_add(_c,b_conv)

	    return c

def lrelu(x):
    return tf.maximum(x, 0.2*x)

def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)                                      

def tfdense(layer_number,image_size,givesize,take,resize,scope):
    with tf.variable_scope(scope or "Linear"):
	    take_size=take.shape[-1]	
	    shape_in=take_size
	    
	    if resize:
		shape_in=shape_in*image_size*image_size
		take=tf.reshape(take,[-1,shape_in])
	    
	    W_fc=tf.get_variable("W_fc"+str(layer_number),shape=[shape_in,givesize],
		                 initializer=tf.random_normal_initializer(stddev=0.02))
	    b_fc=tf.get_variable("b_fc"+str(layer_number),shape=[givesize])
	    
		
	    h_fc=tf.matmul(take,W_fc)+b_fc
	    
	    return h_fc

def deconv(layer_number,givesize,take,conv_window,stride):

    with tf.variable_scope('dc'+str(layer_number)):
	    #givesize: number of nodes in the next layer
	    #take: thing outputted by previous layer
	      
	    init=tf.truncated_normal_initializer(stddev=0.02)

	    new_height=take.shape[1]*stride
	    new_width=take.shape[2]*stride	

	    W_conv=tf.get_variable("W_conv"+str(layer_number),shape=[conv_window,conv_window,givesize,take.shape[-1]],initializer=init)
	    b_conv=tf.get_variable("b_conv"+str(layer_number),shape=[givesize],initializer=tf.constant_initializer(0.0))

	    _c=tf.nn.conv2d_transpose(take,W_conv,output_shape=[int(take.shape[0]),
			int(new_height),int(new_width),givesize],strides=[1,stride,stride,1],padding='SAME')
	    c=tf.nn.bias_add(_c,b_conv)

	    return c



