from email.mime import base
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
import numpy as np
import cv2 
from tensorflow.keras.models import Model

class VGG_SegMap(tf.keras.Model):
    def __init__(self,name):
        super(VGG_SegMap, self).__init__()
        # Carica il modello VGG16 preaddestrato su ImageNet senza i fully connected layers
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(None, None, 3))
        weights = base_model.layers[1].get_weights()[0]
        kernel_size = base_model.layers[1].kernel_size
        stride = base_model.layers[1].strides
        out_shape = base_model.layers[1].output.shape

        new_layer = tf.keras.layers.Conv2D(out_shape[-1],kernel_size,stride,padding='same',activation='relu',input_shape=(None,None,6),name="3x3x6_conv"+name)
        new_layer.build((None, None, None, 6))
        new_layer.trainable = False
        # Create a new model by replacing the first convolutional layer in the base model
        self.features = tf.keras.Sequential()

        self.features.add(new_layer)
        for layer in base_model.layers[2:-5]: 
             layer._name = layer.name+"_"+self.name
             layer.trainable = False
             self.features.add(layer)
             if "block4" in layer.name:
                 layer.trainable = True
                 self.features.add(tf.keras.layers.Dropout(0.2))
        self.features.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',strides=(2,1)))
        
        self.features.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',strides=(2,1)))
        
        self.features.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',strides=(1,1)))
        self.features.layers[0].set_weights([tf.concat([weights]*2,axis=2),self.features.layers[0].get_weights()[1]])
        """for i,layer in enumerate(self.features.layers):   
            if "pool" in layer.name:
               print(f"Layer {i}: {layer.name} - Padding: {layer.padding}, Strides: {layer.strides}, Ksize: {layer.pool_size}")
            elif "dropout" in layer.name:
                continue
            else:
                print(f"Layer {i}: {layer.name} - Padding: {layer.padding}, Strides: {layer.strides}, trainable: {layer.trainable}, in_shape: {layer.input_shape}, , w_shape: {layer.weights[0].shape}")"""
        #self.features.summary()

        
    def call(self, inputs):
        # Concatena l'input con se stesso 3 volte per avere 6 canali
        #concatenated_input = tf.concat([inputs, inputs, inputs], axis=-1)
        # Passa l'input attraverso il modello VGG16 preaddestrato
        features = self.features(inputs)
        return features

    def summary(self):
        self.features.summary()

# Crea un'istanza del modello personalizzato
#custom_vgg = VGG_SegMap('SAT')

# Stampa un riepilogo del modello
#custom_vgg.summary()


"""inputs = Input(shape=(None,None,6))#0000001
img = cv2.imread('../Data/CVUSA/polarmap/19/prova.png')
print("---------------------------------")
print("IMAGE SHAPE: {}".format(img.shape))
print("---------------------------------")
img = cv2.resize(img, (512, 128), interpolation=cv2.INTER_AREA)
#img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
img = img.astype(np.float32)
print("---------------------------------")
print("IMAGE SHAPE: {}".format(img.shape))
print("---------------------------------")

combined_img = np.concatenate((img,img),axis=2)

#custom_vgg_model = VGGModel(inputs)
batch_grd = np.zeros([1, 128, 512, 6], dtype = np.float32)
batch_grd[0,:,:,:] = combined_img
# Build the model
#custom_vgg_model.build_model(inputs)
# Print the model summary
#custom_vgg_model.summary()
#print("-------------------------------------------------")
print("SHAPE FEATURES BATCH: {}".format(custom_vgg(batch_grd).shape))"""
