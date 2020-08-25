
import tensorflow as tf
import numpy as np
import idx2numpy
from funcs import *
import sys

#______________________________

tf.reset_default_graph()

result_dir = './images2/'

epochs=50
batch_size=5
imsize=28 # mnist

learning_rate=0.0002
beta1=0.5

#_____________________________

inputs = tf.placeholder(tf.float32, [batch_size,imsize,imsize,1], name='real_images')
z = tf.placeholder(tf.float32, [batch_size, imsize*imsize], name='z')

#_____________________________



def discriminator(x, is_training=True, reuse=False):
	conv_window,stride=4,2
	with tf.variable_scope("discriminator", reuse=reuse):
	    net = lrelu(tfconv('_D1',64,x,conv_window,stride))
	    net = lrelu(bn(tfconv('_D2',128,net,conv_window,stride), is_training=is_training, scope='D_bn2'))
	    net = tf.reshape(net, [batch_size, -1])
	    net = lrelu(bn(tfdense('_D3',28,1024,net,False,scope='d_fc3'), is_training=is_training, scope='D_bn3'))
	    out_logit = tfdense('_D4',28,1,net,False,scope='d_fc4')
	    out = tf.nn.sigmoid(out_logit)

	    return out, out_logit

def generator(z, is_training=True, reuse=False):
	conv_window,stride=4,2
	with tf.variable_scope("generator", reuse=reuse):
	    net = tf.nn.relu(bn(tfdense('_G1',28,1024,z,False,scope='G_fc1'), is_training=is_training, scope='G_bn1'))
	    net = tf.nn.relu(bn(tfdense('_G2',28,7*7*128,net,False,scope='G_fc2'), is_training=is_training, scope='G_bn2'))
	    net = tf.reshape(net, [batch_size, 7, 7, 128])
	    net = tf.nn.relu(
		bn(deconv('G_dc3',64,net,conv_window,stride), is_training=is_training,
		   scope='G_bn3'))
	    out = tf.nn.sigmoid(deconv('G_dc4',1,net,conv_window,stride))

	    return out

gen_out=generator(z,True,False)

disc_out_y,doy_logits=discriminator(inputs,True,False)
disc_out_x,dox_logits=discriminator(gen_out,True,True)



#_____________________________

disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=doy_logits, labels=tf.ones_like(disc_out_y)))
disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=dox_logits, labels=tf.zeros_like(disc_out_x)))

disc_loss = disc_loss_real + disc_loss_fake

gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=dox_logits, labels=tf.ones_like(disc_out_x)))


#_____________________________

t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'D' in var.name]
g_vars = [var for var in t_vars if 'G' in var.name]

d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(disc_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate*5, beta1=beta1).minimize(gen_loss, var_list=g_vars)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

savestep=300

sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()

mnist_path='./train-images.idx3-ubyte'
_labels=idx2numpy.convert_from_file(mnist_path)

labels=[]
random_inputs=[]
for i in range(0, 60000):
	random_inputs+=[np.random.rand(imsize,imsize).flatten()]
	labels+=[_labels[i].flatten()/255.]

data=np.asarray(random_inputs)
labels=np.asarray(labels).reshape((60000, 28, 28, 1))
channels=1

for ep in range(0,epochs):
    for i in range(0,int(len(data)/batch_size)):

        sess.run(d_optim,feed_dict={z:data[batch_size*i:batch_size*(i+1)],inputs:labels[batch_size*i:batch_size*(i+1)]})
        sess.run(g_optim,feed_dict={z:data[batch_size*i:batch_size*(i+1)]})
        
        if (i + ep*int(len(data)/batch_size))%savestep==0:            
            loss1=sess.run(disc_loss,feed_dict={z:data[batch_size*i:batch_size*(i+1)],inputs:labels[batch_size*i:batch_size*(i+1)]})
            loss2=sess.run(gen_loss,feed_dict={z:data[batch_size*i:batch_size*(i+1)]})
            
            print("epoch: "+str(ep)+", iteration: " + str(i) + ", losses: " + str(loss1) + ', '+ str(loss2))
            saver.save(sess,"./models/savemodel.ckpt",global_step=1000)

	    bucket=data[batch_size*i:batch_size*(i+1)]
            result=sess.run(gen_out,feed_dict={z:bucket}).reshape((batch_size,imsize,imsize,channels))
	

            res=result_filter(result,imsize)
	    cv2.imwrite("./images/epoch_"+str(ep)+"_iter_"+str(i)+".jpg",res[0]*255)


tf.reset_default_graph()




















