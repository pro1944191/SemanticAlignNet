import os
from sre_compile import dis
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from cir_net_FOV import *
from polar_input_data_orien_FOV_3 import InputData

import tensorflow as tf
import numpy as np

import argparse
tf.compat.v1.disable_eager_execution()
parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type', type=str, help='network type', default='VGG_13_conv_v2_cir')

parser.add_argument('--start_epoch', type=int, help='from epoch', default=0)
parser.add_argument('--number_of_epoch', type=int, help='number_of_epoch', default=100)
parser.add_argument('--polar', type=int, help='0 or 1', default=1)

parser.add_argument('--train_grd_noise', type=int, help='0~360', default=360)
parser.add_argument('--test_grd_noise', type=int, help='0~360', default=0)

parser.add_argument('--train_grd_FOV', type=int, help='70, 90, 180, 360', default=180)
parser.add_argument('--test_grd_FOV', type=int, help='70, 90, 180, 360', default=180)

args = parser.parse_args()

# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
network_type = args.network_type

start_epoch = 0#args.start_epoch
polar = args.polar

train_grd_noise = args.train_grd_noise
test_grd_noise = args.test_grd_noise

train_grd_FOV = args.train_grd_FOV
test_grd_FOV = args.test_grd_FOV

number_of_epoch = 5
"""args.number_of_epoch"""

data_type = 'CVUSA'

loss_type = 'l1'

batch_size = 16
is_training = True
loss_weight = 10.0

learning_rate_val = 1e-4
keep_prob_val = 0.8

dimension = 4

# -------------------------------------------------------- #

def validate(dist_array, topK):
    accuracy = 0.0
    data_amount = 0.0

    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[i, :] < gt_dist)
        if prediction < topK:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy


def compute_loss(dist_array):

    with tf.compat.v1.name_scope('weighted_soft_margin_triplet_loss'):

        pos_dist = tf.linalg.tensor_diag_part(dist_array)

        pair_n = batch_size * (batch_size - 1.0)
        tf.print("--------------LOSS---------------")
        tf.print(dist_array)
        tf.print("---------------")

        # satellite to ground
        triplet_dist_g2s = pos_dist - dist_array
        loss_g2s = tf.reduce_sum(input_tensor=tf.math.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n
        tf.print(triplet_dist_g2s * loss_weight)
        tf.print("--------------LOSS---------------")
        # ground to satellite
        triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
        loss_s2g = tf.reduce_sum(input_tensor=tf.math.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

        loss = (loss_g2s + loss_s2g) / 2.0

    return loss


def train(start_epoch=0):
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 0.
    '''

    # import data
    input_data = InputData()

    width = int(test_grd_FOV / 360 * 512)
    #print(width)
    grd_x = tf.compat.v1.placeholder(tf.float32, [None, 128, width, 3], name='grd_x')
    sat_x = tf.compat.v1.placeholder(tf.float32, [None, 256, 512, 3], name='sat_x')
    polar_sat_x = tf.compat.v1.placeholder(tf.float32, [None, 128, 512, 3], name='polar_sat_x')

    grd_orien = tf.compat.v1.placeholder(tf.int32, [None], name='grd_orien')

    keep_prob = tf.compat.v1.placeholder(tf.float32)
    learning_rate = tf.compat.v1.placeholder(tf.float32)

    # build model
    sat_matrix, grd_matrix, distance, pred_orien = VGG_13_conv_v2_cir(polar_sat_x, grd_x, keep_prob, is_training)
    #tf.print(distance)
    #print("GRD SHAPE: {}".format(grd_matrix))

    loss = compute_loss(distance)
    #print(loss)
    #input("WAITTTTT")
    s_height, s_width, s_channel = sat_matrix.get_shape().as_list()[1:]
    g_height, g_width, g_channel = grd_matrix.get_shape().as_list()[1:]
    sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
    grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])
    orientation_gth = np.zeros([input_data.get_test_dataset_size()])

    # set training
    global_step = tf.Variable(0, trainable=False)

    with tf.compat.v1.name_scope('train'):
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate, 0.9, 0.999).minimize(loss, global_step=global_step)

    print('setting saver...')
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=None)
    print('setting saver done...')

    # run model
    print('run model...')
    config = tf.compat.v1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    print('open session ...')
    with tf.compat.v1.Session(config=config) as sess:
        print('initialize...')
        sess.run(tf.compat.v1.global_variables_initializer())

        print('load model...')

        if start_epoch == 0:
            load_model_path = './Model/Initialize/initial_model.ckpt'
            #load_model_path = './Model/polar_1/CVUSA/VGG_13_conv_v2_cir/train_grd_noise_360/train_grd_FOV_360/model.ckpt'
            variables = tf.train.list_variables('./Model/Initialize/')
            saver.restore(sess, load_model_path)
            print("Variables in the checkpoint:")
            for var in variables:
                print(var)
            input("WAITTTT")
        else:

            """load_model_path = './Model/polar_' + str(polar) + '/' + data_type + '/' + network_type \
                              + '/train_grd_noise_' + str(train_grd_noise) + '/train_grd_FOV_' + str(train_grd_FOV) \
                              + '/FOV/' + str(start_epoch - 1) + '/model.ckpt'"""
            load_model_path = './Model/TRY_polar_1/CVUSA/VGG_13_conv_v2_cir/train_grd_noise_360/train_grd_FOV_ADAM_1e3_PADDING_IMAGES_NO_NORMALIZATION_NO_FLOAT_str(train_grd_FOV)/FOV/20/model.ckpt'
            saver.restore(sess, load_model_path)
        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')

        # Train
        loss_val = 0
        for epoch in range(start_epoch, start_epoch + number_of_epoch):
            iter = 0
            while True:
                # train
                batch_sat_polar, batch_sat, batch_grd, batch_orien = input_data.next_pair_batch(8, grd_noise=train_grd_noise,
                                                                               FOV=train_grd_FOV)
                #print(batch_grd.shape)
                #input("")
                if batch_sat is None:
                    break

                global_step_val = tf.compat.v1.train.global_step(sess, global_step)

                feed_dict = {grd_x: batch_grd, polar_sat_x: batch_sat_polar, grd_orien: batch_orien,
                             learning_rate: learning_rate_val, keep_prob: keep_prob_val}
                if iter % 20 == 0:

                    #_,loss_val = sess.run([train_step, loss],
                    #                                             feed_dict=feed_dict)
                    _, loss_val, sat_matrix_val, grd_matrix_val, distance_val, pred_orien_val = sess.run([train_step, loss, sat_matrix, grd_matrix, distance, pred_orien], feed_dict=feed_dict)
                    print("GRD MATRIX VAL: {}".format(grd_matrix_val))
                    print("SAT MATRIX VAL: {}".format(sat_matrix_val))

                    tensor_np = sat_matrix_val
                    # Iterate over all the elements of the array
                    for i in range(tensor_np.shape[0]):
                        for j in range(tensor_np.shape[1]):
                            for k in range(tensor_np.shape[2]):
                                for l in range(tensor_np.shape[3]):
                                    element = tensor_np[i, j, k, l]
                                    if element > 1 or element < -1:
                                        print("HEREEEE")
                                # Do something with the element
                                #print(f"Element at position ({i}, {j}, {k}): {element}")

                    input("WAIT")
                    print('global %d, epoch %d, iter %d: loss : %.4f ' %
                          (global_step_val, epoch, iter, loss_val))
                else:
                    sess.run(train_step, feed_dict=feed_dict)

                iter += 1

            #model_dir = './Model/TRY_polar_' + str(polar) + '/' + data_type + '/' + network_type \
            #            + '/train_grd_noise_' + str(train_grd_noise) + '/train_grd_FOV_' + 'ADAM_1e3_PADDING_IMAGES_NO_NORMALIZATION_NO_FLOAT_'"""str(train_grd_FOV)""" \
            #            + '/FOV/' + str(epoch) + '/'

            #if not os.path.exists(model_dir):
            #    os.makedirs(model_dir)
            #save_path = saver.save(sess, model_dir + 'model.ckpt')
            #print("Model saved in file: %s" % save_path)

            # ---------------------- validation ----------------------

            print('validate...')
            print('   compute global descriptors')
            input_data.reset_scan()

            val_i = 0
            while True:
                #print('      progress %d' % val_i)
                batch_sat_polar, batch_sat, batch_grd, batch_orien = input_data.next_batch_scan(batch_size, grd_noise=test_grd_noise,
                                                                               FOV=test_grd_FOV)
                if batch_sat is None:
                    break

                feed_dict = {grd_x: batch_grd, polar_sat_x: batch_sat_polar, keep_prob: 1.0}
                sat_matrix_val, grd_matrix_val = \
                    sess.run([sat_matrix, grd_matrix], feed_dict=feed_dict)

                sat_global_matrix[val_i: val_i + sat_matrix_val.shape[0], :] = sat_matrix_val
                grd_global_matrix[val_i: val_i + grd_matrix_val.shape[0], :] = grd_matrix_val
                orientation_gth[val_i: val_i + grd_matrix_val.shape[0]] = batch_orien
                val_i += sat_matrix_val.shape[0]
            
            print(sat_global_matrix.shape)
            print(grd_global_matrix.shape)

            print('   compute accuracy')
            sat_descriptor = np.reshape(sat_global_matrix[:, :, :g_width, :], [-1, g_height*g_width*g_channel])
            print(sat_descriptor.shape)
            sat_descriptor = sat_descriptor / np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
            grd_descriptor = np.reshape(grd_global_matrix, [-1, g_height*g_width*g_channel])
            print(sat_descriptor.shape)
            
            print(grd_descriptor.shape)
            # test_grd_noise == 0:
            dist_array = 2 - 2 * np.matmul(grd_descriptor, sat_descriptor.transpose())
            print(dist_array.shape)
            val_accuracy = validate(dist_array, 1)
            input("ASPETTAAAA")
            print('   %d: accuracy = %.1f%%' % (epoch, val_accuracy * 100.0))
            with open('./Result/' + data_type + '/FOV_polar_' + str(polar) + '_' + str(network_type)
                      + '_train_grd_noise_' + str(train_grd_noise) + '_train_grd_FOV_' + str(train_grd_FOV)
                      + '_test_grd_noise_' + str(0) + '_test_grd_FOV_' + str(test_grd_FOV)
                      + '_accuracy_ADAM_1e3_PADDING_IMAGES_NO_NORMALIZATION_NO_FLOAT.txt', 'a') as file:
                file.write(str(epoch) + ' ' + str(iter) + ' : ' + str(val_accuracy) + ' , ' + str(loss_val) + '\n')


if __name__ == '__main__':
    train(start_epoch)