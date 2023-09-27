# import tensorflow as tf

#from VGG import VGG16
#from VGG_cir import VGG16_cir
from pyexpat import model
#from MobileNet import MobileNetClassifier

# from utils import *
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, ReLU, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, BatchNormalization
import tensorflow as tf

class ProcessFeatures():
    def __init__(self):
        super(ProcessFeatures, self).__init__()

        #self.groundNet = MobileNetClassifier()
        #self.satelliteNet = MobileNetClassifier()
    

    def tf_shape(self, x, rank):
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(input=x), rank)
        #print("-----------------")
        #print(static_shape)
        #for a in dynamic_shape:
            #print(a)
        #print("-----------------")

        return [s if s is not None else d for s,d in zip(static_shape, dynamic_shape)]

    # TENSORFLOW
    def corr(self, sat_matrix, grd_matrix):
        #print("SAT SHAPE: {}".format(sat_matrix.shape))
        s_h, s_w, s_c = sat_matrix.get_shape().as_list()[1:]
        g_h, g_w, g_c = grd_matrix.get_shape().as_list()[1:]
        assert s_h == g_h, s_c == g_c #devono avere la stessa altezza e lo stesso numero di canali
        def warp_pad_columns(x, n):
            out = tf.concat([x, x[:, :, :n, :]], axis=2)
            return out
        n = g_w - 1
        #print("X PERMUTATO: {}".format(sat_matrix.shape))
        x = warp_pad_columns(sat_matrix, n)
        #print("X PERMUTATO2: {}".format(x.shape))
        f = tf.transpose(a=grd_matrix, perm=[1, 2, 3, 0])
        out = tf.nn.conv2d(input=x, filters=f,  strides=[1, 1, 1, 1], padding='VALID')
        h, w = out.get_shape().as_list()[1:-1]
        assert h==1, w==s_w
        
        #print("OUT SHAPE: {}".format(out.shape))
        out = tf.squeeze(out,axis=1)  # shape = [batch_sat, w, batch_grd]
        #print("OUT SHAPE:  {}".format(out))
        #print(out.shape)
        #print("OUT SHAPE: {}".format(out.shape))
        #input("WAIT")
        orien = tf.argmax(input=out, axis=1)  # shape = [batch_sat, batch_grd]
        #print(orien.shape)
        #print(orien)
        #print(tf.cast(orien, tf.int32))
        #print("ORIEN SHAPE: {}".format(orien.shape))
        #print("OUT SHAPE: {}".format(out.shape))
        #print("OUT: {}".format(out))
        #print("COORD --------")
        #print(orien)
        #print(out[0][0])
        #print(out[1][0])
        #print(out[0])
        #print(out[orien[0][0][orien[0][0]]])
        #print(out[orien[0][0][orien[1][0]]])
        #print("COORD --------")

        return out, tf.cast(orien, tf.int32)


    def crop_sat(self, sat_matrix, orien, grd_width):
        #print("SAT SHAPE: {}".format(sat_matrix.shape))
        #print("GRD SHAPE: {}".format(grd_width))
        #print("ORIEN SHAPE: {}".format(orien.shape))
        #print("ORIEN VALUES: {}".format(orien))
        batch_sat, batch_grd = self.tf_shape(orien, 2) #ritorna le dimensioni dei due batch
        #print(batch_sat)
        #print(batch_grd)
        h, w, channel = sat_matrix.get_shape().as_list()[1:]
        #print(sat_matrix.get_shape().as_list()[1:])
        sat_matrix = tf.expand_dims(sat_matrix, 1) # shape=[batch_sat, 1, h, w, channel]
        sat_matrix = tf.tile(sat_matrix, [1, batch_grd, 1, 1, 1])
        #print("SAT SHAPE AFTER TILE: {}".format(sat_matrix.shape))
        sat_matrix = tf.transpose(a=sat_matrix, perm=[0, 1, 3, 2, 4])  # shape = [batch_sat, batch_grd, w, h, channel]
        #print("SAT SHAPE AFTER TRANSP: {}".format(sat_matrix.shape))
        orien = tf.expand_dims(orien, -1) # shape = [batch_sat, batch_grd, 1]

        i = tf.range(batch_sat)
        j = tf.range(batch_grd)
        k = tf.range(w)

        x, y, z = tf.meshgrid(i, j, k, indexing='ij')
        #print("Z: {}".format(z))
        z_index = tf.math.floormod(z + orien, w)
        #print("Z INDEX: {}".format(z_index))
        #print("ORIEN: {}".format(orien))
        x1 = tf.reshape(x, [-1])
        y1 = tf.reshape(y, [-1])
        z1 = tf.reshape(z_index, [-1])
        index = tf.stack([x1, y1, z1], axis=1)
        #print("INDEX SHAPE: {}".format(index.shape))
        #print("INDEX: {}".format(index))
        #[batch,batch,]
        #print("SAT MATRIX SHAPE: {}".format(sat_matrix.shape))
        sat = tf.reshape(tf.gather_nd(sat_matrix, index), [batch_sat, batch_grd, w, h, channel])
        #print("SAT MATRIX SHAPE2: {}".format(sat.shape))
        index1 = tf.range(grd_width)
        sat_crop_matrix = tf.transpose(a=tf.gather(tf.transpose(a=sat, perm=[2, 0, 1, 3, 4]), index1), perm=[1, 2, 3, 0, 4])
        #print("SAT CROP MATRIX SHAPE: {}".format(sat_crop_matrix.shape))
        # shape = [batch_sat, batch_grd, h, grd_width, channel]
        assert sat_crop_matrix.get_shape().as_list()[3] == grd_width

        return sat_crop_matrix


    def corr_crop_distance(self, sat_vgg, grd_vgg):
        corr_out, corr_orien = self.corr(sat_vgg, grd_vgg)
        sat_cropped = self.crop_sat(sat_vgg, corr_orien, grd_vgg.get_shape().as_list()[2])
        # shape = [batch_sat, batch_grd, h, grd_width, channel]

        sat_matrix = tf.nn.l2_normalize(sat_cropped, axis=[2, 3, 4])
        #print("SAT CROPPED NORMALIZED: {}".format(sat_matrix.shape))
        distance = 2 - 2 * tf.transpose(a=tf.reduce_sum(input_tensor=sat_matrix * tf.expand_dims(grd_vgg, axis=0), axis=[2, 3, 4]))
        # shape = [batch_grd, batch_sat]
        #print("DISTANCE SHAPE: {}".format(distance.shape))
        #print(distance)
        return sat_matrix, distance, corr_orien


    def conv2(self, x, scope_name, dimension=16):
        with tf.compat.v1.variable_scope(scope_name):
            layer15_output = self.conv_layer2(x, 3, [1, 2, 1, 1], 512, 256, self.trainable, True, 'conv_dim_reduct_1')

            layer16_output = self.conv_layer2(layer15_output, 3, [1, 2, 1, 1], 256, 64, self.trainable, True, 'conv5_dim_reduct_2')

            layer17_output = self.conv_layer(layer16_output, 3, 64, dimension, self.trainable, False, 'conv5_dim_reduct_3')  # shape = [7, 39]

        return layer17_output

    def VGG_13_conv_v2_cir(self,x_sat, x_grd):

        #mNet1 = MobileNetClassifier()
        #mNet2 = MobileNetClassifier()
        
        #vgg_grd = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
        #vgg_grd = mNet1.model(x_grd)
        #grd_vgg = tf.nn.l2_normalize(vgg_grd, axis=[1, 2, 3])

        #vgg_sat = mNet2.model(x_sat)
        #vgg_sat = VGG16_cir(x_sat, keep_prob, trainable, 'VGG_grd')
        #sat_vgg = vgg_sat

        #mobilenet_tf.summary()
        #vgg_grd = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
        #grd_layer13 = vgg_grd.layer13_output
        #print("PROVAAAA {}".format(grd_layer13))
        #print("PROVAAAA {}".format(grd_layer13.shape))
        #grd_vgg = conv2(grd_layer13, 'grd', dimension=16)
        #grd_vgg = tf.nn.l2_normalize(grd_vgg, axis=[1, 2, 3])
        
        #vgg_sat = VGG16_cir(x_sat, keep_prob, trainable, 'VGG_sat')
        #vgg_sat = mobilenet_grd(x_sat)

        #sat_layer13 = vgg_sat.layer13_output
        #sat_layer13 = vgg_sat

        #sat_vgg = conv2(sat_layer13, 'sat', dimension=16)
        sat_matrix, distance, pred_orien = self.corr_crop_distance(x_sat, x_grd)
        return x_sat, x_grd, distance, pred_orien
    
"""out = model
out2 = model

out1,out2 = model
sat_matrix, distance, pred_orien = corr_crop_distance(sat_vgg, grd_vgg)
loss(distance)
model.back"""



"""img1 = cv2.imread('./Data/CVUSA/bingmap/19/0000001.jpg')
img1 = cv2.resize(img1, (512, 128), interpolation=cv2.INTER_AREA)
random_shift = int(np.random.rand() * 512 * 360 / 360)
grd_width = int(360/360*512)
j = np.arange(0, 512)
img_dup = img1[:, ((j - random_shift) % 512)[:grd_width], :]
batch_grd = np.zeros([2, 128, grd_width, 3], dtype = np.float32)
batch_grd[0,:,:,:] = img_dup
batch_grd[1,:,:,:] = img_dup


img2 = cv2.imread('./Data/CVUSA/polarmap/19/0000001.png')
img2 = cv2.resize(img2, (512, 128), interpolation=cv2.INTER_AREA)
batch_sat = np.zeros([3, 128, 512, 3], dtype = np.float32)
batch_sat[0,:,:,:] = img2
batch_sat[1,:,:,:] = img2
batch_sat[2,:,:,:] = img2


vgg_sat, vgg_grd, distance, pred_orien = VGG_13_conv_v2_cir(batch_sat, batch_grd, 0.8, True)
print("VGG SAT: {}".format(vgg_sat.shape))
print("VGG GRD: {}".format(vgg_grd.shape))
print("DISTANCE: {}".format(distance.shape))
print("PRED ORIEN: {}".format(pred_orien.shape))
"""

