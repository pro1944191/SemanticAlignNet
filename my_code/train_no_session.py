import os
from pickletools import optimize
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from cir_net_FOV_mb import *
from polar_input_data_orien_FOV_3_Segmap_Concatenation import InputData
#from ConvNeXt import *
#from VGG_patches import *
#from MobileNet import *
#from ResNet50 import *
from VGG16_SegMap import *
from VGG_no_session import *

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
import sys
import argparse
from PIL import Image
print(keras.__version__)
#tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()
parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type', type=str, help='network type', default='VGG_13_conv_v2_cir')

parser.add_argument('--start_epoch', type=int, help='from epoch', default=0)
parser.add_argument('--number_of_epoch', type=int, help='number_of_epoch', default=100)
parser.add_argument('--polar', type=int, help='0 or 1', default=1)

parser.add_argument('--train_grd_noise', type=int, help='0~360', default=360)
parser.add_argument('--test_grd_noise', type=int, help='0~360', default=0)

parser.add_argument('--train_grd_FOV', type=int, help='70, 90, 180, 360', default=70)
parser.add_argument('--test_grd_FOV', type=int, help='70, 90, 180, 360', default=70)

args = parser.parse_args()

# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
network_type = args.network_type

start_epoch = 15#args.start_epoch
polar = args.polar

train_grd_noise = args.train_grd_noise
test_grd_noise = args.test_grd_noise

train_grd_FOV = args.train_grd_FOV
test_grd_FOV = args.test_grd_FOV

number_of_epoch = 10
"""args.number_of_epoch"""

data_type = 'CVUSA'

loss_type = 'l1'

batch_size = 8
is_training = True
loss_weight = 10.0

learning_rate_val = 1e-5
keep_prob_val = 0.8
keep_prob = 0.8
dimension = 4

print("SETTED PARAMETERS: ")
print("Train ground FOV: {}".format(train_grd_FOV))
print("Train ground noise: {}".format(train_grd_noise))
print("Test ground FOV: {}".format(test_grd_FOV))
print("Test ground noise: {}".format(test_grd_noise))
print("Number of epochs: {}".format(number_of_epoch))
print("Learning rate: {}".format(learning_rate_val))
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


    pos_dist = tf.linalg.tensor_diag_part(dist_array)

    pair_n = batch_size * (batch_size - 1.0)

    # satellite to ground
    triplet_dist_g2s = pos_dist - dist_array
    loss_g2s = tf.reduce_sum(input_tensor=tf.math.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

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
    #test_grd_FOV = 360
    width = int(train_grd_FOV / 360 * 512)
    #learning_rate_val = 1e-4
    # Assuming you have defined the CustomModel and have the training data ready

    # Define the loss function
    #loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Define the optimizer
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_val)
    #optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate_val,weight_decay=0.004,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,clipnorm=None,clipvalue=None,global_clipnorm=None,name='AdamW')
    # Define the metrics
    #metrics = [tf.keras.metrics.BinaryAccuracy()]
    
    

    #groundNet = MobileNetClassifier(tf.keras.Input(shape=(None, None, 3)),'ground')
    #satNet = MobileNetClassifier(tf.keras.Input(shape=(None, None, 3)),'sat')
    #groundNet = ResNet(tf.keras.Input(shape=(None, None, 3)),'ground')
    #satNet = ResNet(tf.keras.Input(shape=(None, None, 3)),'sat')
    #groundNet = ConvNext(tf.keras.Input(shape=(None, None, 3)),'grd')
    #satNet = ConvNext(tf.keras.Input(shape=(None, None, 3)),'sat')
    
    #satNet = VGG_SegMap("sat") 
    groundNet = VGGModel(tf.keras.Input(shape=(None, None, 3)))
    satNet = VGGModelCir(tf.keras.Input(shape=(None, None, 3)),'_sat')
    segMapNet = VGGModelCir(tf.keras.Input(shape=(None, None, 3)),'_segmap')
    
    processor = ProcessFeatures()



    #combined_output = tf.keras.layers.concatenate([groundNet.model.output, satNet.model.output])
    #model = Model(inputs=[groundNet.model.input, satNet.features.input], outputs=[groundNet.model.output, satNet.features.output])    
    model = Model(inputs=[groundNet.model.input, satNet.model.input, segMapNet.model.input], outputs=[groundNet.model.output, satNet.model.output, segMapNet.model.output])    
    
    print("Model created")

    grd_x = np.float32(np.zeros([2, 128, width, 3]))
    sat_x = np.float32(np.zeros([2, 256, 512, 3]))
    #polar_sat_x = np.float32(np.zeros([2, 128, 512, 6]))
    polar_sat_x = np.float32(np.zeros([2, 128, 512, 3]))
    segmap_x = np.float32(np.zeros([2, 128, 512, 3]))

    grd_features, sat_features, segmap_features = model([grd_x, polar_sat_x, segmap_x])
    sat_features = tf.concat([sat_features, segmap_features], axis=-1)
    # build model
    sat_matrix, grd_matrix, distance, pred_orien = processor.VGG_13_conv_v2_cir(sat_features,grd_features)#model.call([grd_x,polar_sat_x])

    s_height, s_width, s_channel = sat_matrix.get_shape().as_list()[1:]
    g_height, g_width, g_channel = grd_matrix.get_shape().as_list()[1:]
    sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
    grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])
    orientation_gth = np.zeros([input_data.get_test_dataset_size()])

    # load Model
    if start_epoch != 0:
        model_path = "./saved_models/FOV70_segmap_concatenation/14/"
        model = keras.models.load_model(model_path)
        print("Model checkpoint uploaded")


    # Iterate over the desired number of epochs
    for epoch in range(start_epoch, start_epoch + number_of_epoch):
        print(f"Epoch {epoch+1}/{start_epoch + number_of_epoch}")
        
        #if epoch == 8:
        #    learning_rate_val = 1e-5
        #    print("Learning rate changed")
        # Iterate over the training data in batches
        iter = 0
        end = False
        finalEpochLoss = 0
        while True:
        #for batch_inputs, batch_labels in train_dataset:

            total_loss = 0
            #Gradient accumulation
            for i in range(4):
                
                """if i==0:
                    train_grd_FOV = 360
                elif i==1:
                    train_grd_FOV = 180
                elif i==2:
                    train_grd_FOV = 90
                else:
                    train_grd_FOV = 70"""
                
                #try:
                batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien = input_data.next_pair_batch(8, grd_noise=train_grd_noise, FOV=train_grd_FOV)
                #except Exception as e:
                #     print(iter)
                #     print(e)
                #     print(batch_sat_polar.shape)
                #     print(batch_sat.shape)
                #     print(batch_grd.shape)
                #     print(batch_orien.shape)
                #     print(batch_segmap)

                """cv2.imshow('image',batch_grd[-1])
                print(batch_grd[0])
                pil_image = Image.fromarray(np.uint8(batch_grd[0]))

                # Display the image
                pil_image.show()
                cv2.waitKey(0)
                cv2.imshow('image',batch_sat_polar[0])
                cv2.waitKey(0)
                """
                #print(batch_sat_polar[0])
                #input("WAIT")
                #batch_sat_polar, batch_sat, batch_grd, batch_orien = input_data.next_pair_batch(8, grd_noise=train_grd_noise, FOV=train_grd_FOV)
                if batch_sat is None:
                    end = True
                    break

                with tf.GradientTape() as tape:
                    # Forward pass through the model
                    grd_features, sat_features, segmap_features = model([batch_grd, batch_sat_polar, batch_segmap])
                    grd_features = tf.nn.l2_normalize(grd_features, axis=[1, 2, 3])
                    
                    sat_features = tf.concat([sat_features, segmap_features], axis=-1)

                    #L2 Regularization da provare qui
                    #Gradient clipping

                    sat_matrix, grd_matrix, distance, orien = processor.VGG_13_conv_v2_cir(sat_features,grd_features)

                    # Compute the loss
                    loss_value = compute_loss(distance)
                    total_loss += loss_value 
                    
                # Compute the gradients
                gradients = tape.gradient(loss_value, model.trainable_variables)
                if i == 0:
                        accumulated_gradients = gradients
                else:
                        accumulated_gradients = [(acum_grad + grad) for acum_grad, grad in zip(accumulated_gradients, gradients)]

            gradients = [acum_grad / tf.cast(4, tf.float32) for acum_grad in accumulated_gradients]
            #optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # Update the model's weights
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            if iter % 100 == 0:
                print("ITERATION: {}, LOSS VALUE: {}, TOTAL LOSS: {}".format(iter, loss_value.numpy(),total_loss/4))

            # Update the metrics
            #for metric in metrics:
            #    metric.update_state(distance)
            iter+=1

            if end:
                 break
        # Print the training metrics for this epoch
        #print("Training Metrics:")
        #for metric in metrics:
        #    print(f"{metric.name}: {metric.result()}")
        
        # Reset the metrics for the next epoch
        #for metric in metrics:
        #    metric.reset_states()
    
        #save the model
        model_path = "./saved_models/FOV70_segmap_concatenation/"+str(epoch)+"/"
        model.save(model_path)
        #print("FLIPPED IMAGES: {}".format(input_data.flipped))

        print("Validation...")
        val_i = 0
        count = 0
        while True:
            # print('      progress %d' % val_i)
            batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien  = input_data.next_batch_scan(8, grd_noise=test_grd_noise,
                                                                           FOV=test_grd_FOV)
            if batch_sat is None:
                break
            grd_features, sat_features, segmap_features = model([batch_grd, batch_sat_polar, batch_segmap])
            grd_features = tf.nn.l2_normalize(grd_features, axis=[1, 2, 3])

            sat_features = tf.concat([sat_features, segmap_features], axis=-1)
            sat_matrix, grd_matrix, distance, orien = processor.VGG_13_conv_v2_cir(sat_features,grd_features)

            sat_global_matrix[val_i: val_i + sat_matrix.shape[0], :] = sat_matrix
            grd_global_matrix[val_i: val_i + grd_matrix.shape[0], :] = grd_matrix
            orientation_gth[val_i: val_i + grd_matrix.shape[0]] = batch_orien

            val_i += sat_matrix.shape[0]
            count += 1

        sat_descriptor = np.reshape(sat_global_matrix[:, :, :g_width, :], [-1, g_height * g_width * g_channel])
        sat_descriptor = sat_descriptor / np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
        grd_descriptor = np.reshape(grd_global_matrix, [-1, g_height * g_width * g_channel])

        data_amount = grd_descriptor.shape[0]
        print('      data_amount %d' % data_amount)
        top1_percent = int(data_amount * 0.01) + 1
        print('      top1_percent %d' % top1_percent)


        dist_array = 2 - 2 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))
        #dist_array = 2 - 2 * np.matmul(grd_descriptor, sat_descriptor.transpose())

        #ME
        val_accuracy = validate(dist_array, 1)
        print('accuracy = %.1f%%' % (val_accuracy * 100.0))
        with open('./saved_models/FOV70_segmap_concatenation/FOV70_segmap_concatenation.txt', 'a') as file:
                file.write(str(epoch) + ': ' + str(val_accuracy) + ', Loss: ' + str(loss_value.numpy()) + '\n')
        #ME

if __name__ == '__main__':
    train(start_epoch)
