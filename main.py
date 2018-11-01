# coding: utf-8
# 2018-10-11， mason peng
# deep learning not all for recognition/detection
# this project show hwo DL trained for normal application case.

import tensorflow as tf
import numpy as np
import scipy.io
import cv2
import os
import time,timeit
import logging




TM=time.strftime("%Y-%m-%d %H-%M-%S",time.localtime())
LOG_FORMAT = "%(asctime)s - %(levelname)s - [:%(lineno)d]- %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logging.info('************************** Mason (%s):Transfer *******************'%(TM))

def func_track(func):
    def track(*args, **kw):
        name = func.__name__
        t0 = timeit.default_timer()
        logging.info('****%s ....'%(name))

        result = func(*args, **kw)

        elapsed = timeit.default_timer() - t0
        logging.info('****%s done\n' % (name))
        #logging.info('****%s done[%0.8fs]\n' %(name, elapsed))
        return result
    return track



###### configuration
device='/cpu:0'#or '/gpu:0'
video = 0

max_train_step=90000
show_iter=100
learning_rate=0.01
noise_ratio=0.5

image_size=(800,600)

path_vgg="imagenet-vgg-verydeep-19.mat"
vgg_raw_net = scipy.io.loadmat(path_vgg)
vgg_layers = vgg_raw_net['layers'][0]

path="/Users/mason/PycharmProjects/transfer/"

content_image_path=os.path.join(path,"content_images")
content_image_name="lion.jpg"
content_layer_pick='conv4_2'

style_image_path =os.path.join(path,"style_images")
style_image_name= ["starry-night.jpg"]
style_layer_pick=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
style_layer_factor=[0.2, 0.2, 0.2, 0.2, 0.2]
style_imgs_weights=[1.0] # single style

output_path=os.path.join(path,"output/result.jpg")



class my_assist_func:
    ##### get image
    # image shape for net shape, and image value for loss cal
    @func_track
    def get_content_image(self):
        #logging.info('****get_content_image')
        path=os.path.join(content_image_path, content_image_name)
        img=cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise OSError("No such fule", path)
        img=img.astype(np.float32)
        img=cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
        img=self.pre_process(img)
        return img

    @func_track
    def get_style_image(self):
        #logging.info('****get_style_image')
        style_images_=[]
        for img_i in style_image_name:
            path=os.path.join(style_image_path, img_i)
            img=cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise OSError("No such fule", path)
            img=img.astype(np.float32)
            img=cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            img=self.pre_process(img)
            style_images_.append(img)
        return style_images_

    @func_track
    def get_init_image(self,init_type, content_img, style_imgs): # using content_image is fastest
        #logging.info('****get_init_image')
        if init_type == 'content':
            return content_img
        elif init_type == 'style':
            return style_imgs[0]
        elif init_type == 'random':
            np.random.seed(args.seed)
            noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
            init_img = noise_ratio * noise_img + (1. - noise_ratio) * content_img
            return init_img


    #@func_track
    def pre_process(self, img):
        logging.info('****pre_process')
        # bgr to rgb
        img = img[..., ::-1]
        # shape (h, w, d) to (1, h, w, d)
        img = img[np.newaxis, :, :, :]

        img -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))# normalization
        return img

    #@func_track
    def post_process(self, img):
        logging.info('****post_process')
        img += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
        # shape (1, h, w, d) to (h, w, d)
        img = img[0]
        img = np.clip(img, 0, 255).astype('uint8')
        # rgb to bgr
        img = img[..., ::-1]
        return img

    #@func_track
    def get_weights(self, i):
        logging.info('****get_weights')
        weights =vgg_layers[i][0][0][2][0][0]
        W = tf.constant(weights)
        return W

    #@func_track
    def get_bias(self, i):
        logging.info('****get_bias')
        bias = vgg_layers[i][0][0][2][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))
        return b

    ###### loss func definition
    # the losses  originate from net['input']
    @func_track
    def content_loss(self, fmp_content, fmp_noise_image):
        #logging.info('****content_loss func')
        P = tf.convert_to_tensor(fmp_content)  # matrix value  to tensor
        F =fmp_noise_image
        _, h, w, d =P.get_shape()
        area =h.value * w.value
        depth =d.value
        K= 1. / (2. * depth ** 0.5 * area ** 0.5)
        loss= K * tf.reduce_sum(tf.pow((F - P),2))
        return loss

    #@func_track
    def gamma_matrix (self, matrix, area, depth):
        logging.info('****gamma_matrix')
        M=tf.reshape(matrix, (area, depth))
        # M·M' is covariance matrix
        gm=tf.matmul(tf.transpose(M),M) # different to tf.multiply()
        return gm

    @func_track
    def style_loss(self, fmp_style, fmp_noise_image):
        #logging.info('****style_loss func')
        Tensor = tf.convert_to_tensor(fmp_style)  # matrix value  to tensor
        _, h, w, d =Tensor.get_shape()
        area =h.value * w.value
        depth =d.value
        A= self.gamma_matrix(fmp_style,area,depth)
        G= self.gamma_matrix(fmp_noise_image,area,depth)
        K= (1. / (4 * depth ** 2 * area ** 2))
        loss =  K * tf.reduce_sum(tf.pow((G - A), 2))
        return loss



###########################################

####### network #### this network does not need to be trained,
#  network here is because we used feature map
class my_network:
    # layer
    def conv_layer(self,name, input, layer_i):
        logging.info('****conv_layer: %s'%name)
        W=assist_func.get_weights(layer_i)
        conv=tf.nn.conv2d(input, W, strides=[1,1,1,1], padding='SAME')
        return conv

    def relu_layer(self, name, input,layer_i):
        logging.info('****relu_layer: %s' % name)
        b=assist_func.get_bias(layer_i)
        relu=tf.nn.relu(input+b)
        return relu

    def pool_layer(self, name, input, type_='max'):
        logging.info('****pooling_layer: %s' % name)
        if type_=='max':
            pool=tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
        elif type_=='avg':
            pool = tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool


    # network
    @func_track
    def network(self,image):
        input_shape=image.shape
        logging.info('****build_network, and input_shape: {} \n'.format(input_shape))
        #network is constructed by net['']
        # the only rf.variable is net['input'],this also the result
        _,h,w,d=input_shape
        net={}
        net['input']=tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))

        net['conv1_1']=self.conv_layer('conv1_1', net['input'],0)
        net['relu1_1']=self.relu_layer('relu1_1',net['conv1_1'],0)
        net['conv1_2'] = self.conv_layer('conv1_2', net['relu1_1'], 2)
        net['relu1_2'] = self.relu_layer('relu1_2', net['conv1_2'], 2)
        net['pool1'] = self.pool_layer('pool1', net['relu1_2'])

        net['conv2_1'] = self.conv_layer('conv2_1', net['pool1'], 5)
        net['relu2_1'] = self.relu_layer('relu2_1', net['conv2_1'], 5)
        net['conv2_2'] = self.conv_layer('conv2_2', net['relu2_1'], 7)
        net['relu2_2'] = self.relu_layer('relu1_2', net['conv2_2'], 7)
        net['pool2'] = self.pool_layer('pool2', net['relu2_2'])

        net['conv3_1'] = self.conv_layer('conv3_1', net['pool2'], 10)
        net['relu3_1'] = self.relu_layer('relu3_1', net['conv3_1'], 10)
        net['conv3_2'] = self.conv_layer('conv3_2', net['relu3_1'], 12)
        net['relu3_2'] = self.relu_layer('relu3_2', net['conv3_2'], 12)
        net['conv3_3'] = self.conv_layer('conv3_3', net['relu3_2'], 14)
        net['relu3_3'] = self.relu_layer('relu3_3', net['conv3_3'], 14)
        net['conv3_4'] = self.conv_layer('conv3_4', net['relu3_3'], 16)
        net['relu3_4'] = self.relu_layer('relu3_4', net['conv3_4'], 16)
        net['pool3'] = self.pool_layer('pool3', net['relu3_4'])

        net['conv4_1'] = self.conv_layer('conv4_1', net['pool3'], 19)
        net['relu4_1'] = self.relu_layer('relu4_1', net['conv4_1'], 19)
        net['conv4_2'] = self.conv_layer('conv4_2', net['relu4_1'], 21)
        net['relu4_2'] = self.relu_layer('relu4_2', net['conv4_2'], 21)
        net['conv4_3'] = self.conv_layer('conv4_3', net['relu4_2'], 23)
        net['relu4_3'] = self.relu_layer('relu4_3', net['conv4_3'], 23)
        net['conv4_4'] = self.conv_layer('conv4_4', net['relu4_3'], 25)
        net['relu4_4'] = self.relu_layer('relu4_4', net['conv4_4'], 25)
        net['pool4'] = self.pool_layer('pool4', net['relu4_4'])

        net['conv5_1'] = self.conv_layer('conv5_1', net['pool4'], 28)
        net['relu5_1'] = self.relu_layer('relu5_1', net['conv5_1'], 28)
        net['conv5_2'] = self.conv_layer('conv5_2', net['relu5_1'], 30)
        net['relu5_2'] = self.relu_layer('relu5_2', net['conv5_2'], 30)
        net['conv5_3'] = self.conv_layer('conv5_3', net['relu5_2'], 32)
        net['relu5_3'] = self.relu_layer('relu5_3', net['conv5_3'], 32)
        net['conv5_4'] = self.conv_layer('conv5_4', net['relu5_3'], 34)
        net['relu5_4'] = self.relu_layer('relu5_4', net['conv5_4'], 34)
        net['pool5'] = self.pool_layer('pool5', net['relu5_4'])
        #logging.info('**** network build done \n')
        return net









###################### core business func ##########
@func_track
def single_image_trans():
    #logging.info('****single_image_transfer in...')
    content_image= assist_func.get_content_image() # image for net, and loss cal
    style_images= assist_func.get_style_image()

    with tf.Graph().as_default():
        init_img= assist_func.get_init_image('content', content_image,style_images)
        stylize(content_image, style_images,init_img)

# session
@func_track
def stylize(content_image,style_images,init_img):
    #logging.info('****stylize begin....')
    with tf.device(device), tf.Session() as sess : #or '/gpu:0'

        ##### set up network
        net = network.network(image=content_image)

        ##### core loss func
        logging.info('****core loss func build ...')
        #content
        logging.info('****sess.run: feed content image to net_input')
        sess.run(net['input'].assign(content_image))
        t1 = timeit.default_timer()
        content_feature_map = sess.run(net[content_layer_pick])  # cos sess.run, so it is a matrix value
        tm = timeit.default_timer() - t1
        logging.info('****sess.run: content image pass through net and pick feature map:%s[%0.8fs]'%(content_layer_pick,tm))
        noise_image_feature_map= net[content_layer_pick] # it is a graph
        total_content_loss= assist_func.content_loss(content_feature_map, noise_image_feature_map)

        #style
        total_style_loss=0.
        for img, img_weight in zip (style_images, style_imgs_weights):# for multiple-style trans
            logging.info('****sess.run: feed style image to net_input')
            sess.run(net['input'].assign(img))
            style_loss_=0.
            for layer, factor in zip(style_layer_pick, style_layer_factor):
                t1 = timeit.default_timer()
                style_feature_map = sess.run(net[layer])
                tm = timeit.default_timer() - t1
                logging.info('****sess.run: style image pass through net and pick feature map:%s[%0.8fs]'%(layer,tm))
                noise_image_feature_map=net[layer]
                style_loss_ += assist_func.style_loss(style_feature_map, noise_image_feature_map)*factor
            style_loss_avg=style_loss_ / float(len(style_layer_pick))
            total_style_loss +=style_loss_avg * img_weight
        total_style_loss= total_style_loss / float(len(style_images))

        logging.info('****total loss func')
        alpha = tf.constant(5, tf.float32)  # cos this should be contained into flow graph
        beta = tf.constant(1e4, tf.float32)
        theta = tf.constant(1e-3, tf.float32)
        tv_loss = tf.image.total_variation(net['input'])# de-noising loss  ?
        total_loss=alpha*total_content_loss + beta*total_style_loss + theta*tv_loss
        logging.info('****loss func build done \n')

        ##### optimization,using adam
        # optimize the loss to get result:net_input
        logging.info('****set optimizer...with adam_optimizer; optimizing device:{}'.format(device))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op=optimizer.minimize(total_loss)
        #Since AdamOptimizer has it's own variables, you should define the initilizer init after opt, not before.
        logging.info('****sess.run: global variable init')
        sess.run(tf.global_variables_initializer())
        # tf.global_variables_initialize().run() # sess.run(tf.initialize_all_variables()) # different version for variable init

        logging.info('****sess.run: init_img feed to net_input/forward')
        #first forward
        sess.run(net['input'].assign(init_img))
        logging.info('****sess.run: backward and optimizing,minimizing')
        for step in xrange(max_train_step):
            # backward + forward
            sess.run(train_op)
            if step % show_iter ==0:
                curr_loss=total_loss.eval()
                logging.info (" optimization_iterate {}\tcurr_loss= {}".format(step,curr_loss))

        #output_image is input of the net,/ net['input'], and net['input'] is a graph,
        logging.info('****optimization done')
        logging.info('****sess.run:  get the result, net_input')
        output_image=sess.run(net['input'])

        output_image=assist_func.post_process(output_image)
        logging.info('****save result to:{}'.format(output_path))
        cv2.imwrite(output_path,output_image)



        #writer=tf.summary.FileWriter(os.path.join(path,"tensorboard"), sess.graph)
        #writer.close()


if __name__=='__main__':

    assist_func=my_assist_func()
    network=my_network()

    if video==0:
        single_image_trans()
    else:
        video()

