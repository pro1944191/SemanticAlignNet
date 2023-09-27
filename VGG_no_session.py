from email.mime import base
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from PIL import Image
class VGGModelCir:
    def __init__(self,input_shape,name):
        self.input_shape = input_shape
        self.model = None
        self.name = name
        self.build_model(input_shape)

    def warp_pad_columns(self, x, n=1):

        out = tf.concat([x[:, :, -n:, :], x, x[:, :, :n, :]], axis=2)

        return tf.pad(tensor=out, paddings=[[0, 0], [n, n], [0, 0], [0, 0]])

    def build_model(self, input_shape):
        # Load the VGG16 model without the top (dense) layers
        base_model = VGG16(weights='imagenet', include_top=False)

        # Define the input layer
          # Input shape for the VGG16 model

        # Get the output of each layer and process it before passing it to the next layer
        x = input_shape
        for i,layer in enumerate(base_model.layers):
            
            layer._name = layer.name + self.name
            
            if i <= 9:
                layer.trainable = False
            if i == 0:
                continue
            """if hasattr(layer, 'padding'):
                print("NAME: {}".format(layer.name))
                if "pool" in layer.name:
                    layer.padding = 'same'
                elif "conv" in layer.name:
                    layer.padding = 'valid'
                else:
                    print("ok")
                
                padding = layer.padding #if hasattr(layer, 'padding') else 'valid'
            if "pool" in layer.name:
                print(f"Layer {i}: {layer.name} - Padding: {layer.padding}, Strides: {layer.strides}, Ksize: {layer.pool_size}")
            else:
                print(f"Layer {i}: {layer.name} - Padding: {layer.padding}, Strides: {layer.strides}, trainable: {layer.trainable}")"""

            
            #print("PRE WARP: {}".format(x.shape))
            #x = self.warp_pad_columns(x,1)
            #print("AFRTER WARP: {}".format(x.shape))
            x = layer(x)
            #print("AFTER CONV: {}".format(x.shape))
            #input("WAIT")
            if x.name == "block4_conv1_sat/Relu:0":
                x = Dropout(0.2)(x)
            elif x.name == "block4_conv2_sat/Relu:0":
                x = Dropout(0.2)(x)
            elif x.name == "block4_conv3_sat/Relu:0":
                x = Dropout(0.2)(x)
            if i >= len(base_model.layers) - 6:  # Skip the last three convolutional layers
                break
            # Process the output of each layer before passing it to the next layer
            # For example, you can apply additional convolutional layers, pooling, or any desired operations
        
        
        # Add three additional convolutional layers
        x = self.warp_pad_columns(x,1)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid',strides=(2,1))(x)
        x = self.warp_pad_columns(x,1)
        x = Conv2D(64, (3, 3), activation='relu', padding='valid',strides=(2,1))(x)
        x = self.warp_pad_columns(x,1)
        x = Conv2D(8, (3, 3), activation='relu', padding='valid',strides=(1,1))(x)
        # Create the modified VGG16 model
        self.model = Model(inputs=input_shape, outputs=x)

    def call(self, input):
        return self.model(input)
    
    def summary(self):
        # Print the summary of the model
        if self.model is None:
            raise ValueError("The model has not been built. Call build_model() first.")

        self.model.summary()
        

class VGGModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.build_model(input_shape)

    def warp_pad_columns(self, x, n=1):

        out = tf.concat([x[:, :, -n:, :], x, x[:, :, :n, :]], axis=2)
        return tf.pad(tensor=out, paddings=[[0, 0], [n, n], [0, 0], [0, 0]])

    def build_model(self, input_shape):
        # Load the VGG16 model without the top (dense) layers
        base_model = VGG16(weights='imagenet', include_top=False)

        # Define the input layer
          # Input shape for the VGG16 model

        # Get the output of each layer and process it before passing it to the next layer
        x = input_shape
        for i,layer in enumerate(base_model.layers):
            if i <= 9:
                layer.trainable = False
            if i == 0:
                continue
            """if hasattr(layer, 'padding'):
                print("NAME: {}".format(layer.name))
                if "pool" in layer.name:
                    layer.padding = 'same'
                elif "conv" in layer.name:

                    layer.padding = 'valid'
                else:
                    print("ok")
                
                padding = layer.padding #if hasattr(layer, 'padding') else 'valid'"""
            #if "pool" in layer.name:
            #    print(f"Layer {i}: {layer.name} - Padding: {layer.padding}, Strides: {layer.strides}, Ksize: {layer.pool_size}")
            #else:
            #    print(f"Layer {i}: {layer.name} - Padding: {layer.padding}, Strides: {layer.strides}, trainable: {layer.trainable}")

            
            #print("PRE WARP: {}".format(x.shape))
            #x = self.warp_pad_columns(x,1)
            #print("AFRTER WARP: {}".format(x.shape))
            x = layer(x)
            #print("AFTER CONV: {}".format(x.shape))
            #input("WAIT")
            if x.name == "block4_conv1/Relu:0":
                x = Dropout(0.2)(x)
            elif x.name == "block4_conv2/Relu:0":
                x = Dropout(0.2)(x)
            elif x.name == "block4_conv3/Relu:0":
                x = Dropout(0.2)(x)
            if i >= len(base_model.layers) - 6:  # Skip the last three convolutional layers
                #x = Dropout(0.5)(x)
                break
            # Process the output of each layer before passing it to the next layer
            # For example, you can apply additional convolutional layers, pooling, or any desired operations
        
        
        # Add three additional convolutional layers
        
        x = Conv2D(256, (3, 3), activation='relu', padding='same',strides=(2,1))(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same',strides=(2,1))(x)
        
        x = Conv2D(16, (3, 3), activation='relu', padding='same',strides=(1,1))(x)
        # Create the modified VGG16 model
        self.model = Model(inputs=input_shape, outputs=x)
        """for i,layer in enumerate(self.model.layers):
            if i==0:
                continue   
            if "pool" in layer.name:
                print(f"Layer {i}: {layer.name} - Padding: {layer.padding}, Strides: {layer.strides}, Ksize: {layer.pool_size}")
            elif "dropout" in layer.name:
                continue
            else:
                print(f"Layer {i}: {layer.name} - Padding: {layer.padding}, Strides: {layer.strides}, trainable: {layer.trainable}, w_shape: {layer.weights[0].shape}")"""

    def call(self, input):
        return self.model(input)

    def summary(self):
        # Print the summary of the model
        if self.model is None:
            raise ValueError("The model has not been built. Call build_model() first.")

        self.model.summary()


def extract_patches(image, patch_size):
    patches = []
    height, width, channels = image.shape
    patch_height, patch_width, _ = patch_size

    for y in range(0, height, patch_height):
        for x in range(0, width, patch_width):
            patch = image[y:y+patch_height, x:x+patch_width, :]
            patches.append(patch)

    return patches


#inputs = Input(shape=(128, 512, 3))
#custom_vgg_model = VGGModelCir(inputs)

# Print the model summary
#custom_vgg_model.summary()

"""inputs = Input(shape=(128,512,3))#0000001
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

batch_grd = np.zeros([16, 128, 512, 3], dtype = np.float32)
batch_grd[0,:,:,:] = img
batch_grd[1,:,:,:] = img
batch_grd[2,:,:,:] = img
batch_grd[3,:,:,:] = img
batch_grd[4,:,:,:] = img
batch_grd[5,:,:,:] = img
batch_grd[6,:,:,:] = img
batch_grd[7,:,:,:] = img
batch_grd[8,:,:,:] = img
batch_grd[9,:,:,:] = img
batch_grd[10,:,:,:] = img
batch_grd[11,:,:,:] = img
batch_grd[12,:,:,:] = img
batch_grd[13,:,:,:] = img
batch_grd[14,:,:,:] = img
batch_grd[15,:,:,:] = img


batch_size = 16
image_height = 128
image_width = 128
num_channels = 3

images_batch = batch_grd#np.random.rand(batch_size, image_height, image_width, num_channels)
# Create an instance of the CustomVGGModel class
custom_vgg_model = VGGModel(inputs)

# Build the model
#custom_vgg_model.build_model(inputs)
# Print the model summary
#custom_vgg_model.summary()
#print("-------------------------------------------------")
print("SHAPE FEATURES BATCH: {}".format(custom_vgg_model.model(batch_grd).shape))"""
"""
# Dividi ogni immagine in 16 parti di dimensione (32, 32)
num_parts = 16
part_height = 32
part_width = 32

parts_list = []
for image in images_batch:
    parts = []
    for j in range(0, image_height, part_height): 
        for i in range(0, image_width, part_width):
            part = image[i:i+part_height, j:j+part_width, :]
            parts.append(part)
    parts_list.append(parts)
#print("PARTS LIST SHAPE: {}".format(parts_list.size()))

# Converti la lista di parti in un array di NumPy
parts_array = np.array(parts_list)

print("PARTS ARRAY SHAPE {}".format(parts_array.shape))

reshaped_input = tf.reshape(parts_array, shape=(-1, 32, 32, 3))
print("RESHAPE ARRAY SHAPE {}".format(reshaped_input.shape))
#custom_vgg_model = VGGModel(inputs)
#vgg_processed = custom_vgg_model.model(reshaped_input)
print("VGG SHAPE {}".format(reshaped_input.shape))
reshaped_output = tf.reshape(reshaped_input, shape=(16, 16, 32, 32, 3))
print("RESHAPED VGG SHAPE {}".format(reshaped_output.shape))

reconstructed_images = []

for parts in reshaped_output:
    # Concatenazione lungo l'asse orizzontale per ottenere parti orizzontali
    concatenated_horizontal_parts = [np.concatenate(parts_row, axis=1) for parts_row in parts]
    # Concatenazione lungo l'asse verticale per ottenere l'immagine completa
    reconstructed_image = np.concatenate(concatenated_horizontal_parts, axis=0)
    reconstructed_images.append(reconstructed_image)
print("RECOSTRUCTED IMAGE SHAPE {}".format(reconstructed_images))

pil_image = Image.fromarray(np.uint8(reconstructed_images[0]))

# Display the image
pil_image.show()
input("WAIT")
# Create an instance of the CustomVGGModel class
custom_vgg_model = VGGModel(inputs)

# Build the model
#custom_vgg_model.build_model(inputs)
# Print the model summary
#custom_vgg_model.summary()
#print("-------------------------------------------------")
print("SHAPE FEATURES BATCH: {}".format(custom_vgg_model.model(batch_grd).shape))

#input("WAIT")

#######################################
# Define the patch size
patch_size = (32, 32, 3)

# Extract patches
patches = extract_patches(img, patch_size)

#print(patches[0].shape)

# Convert the list of patches to a numpy array
patches_array = np.array(patches)

# patches_array will have shape (num_patches, patch_height, patch_width, 3)
#print("Number of patches:", patches_array.shape[0])
#######################################

batch_size = 16
image_height = 128
image_width = 128
num_channels = 3

images_batch = batch_grd#np.random.rand(batch_size, image_height, image_width, num_channels)
#print("BATCH SHAPE: {}".format(images_batch.shape))

# Dividi ogni immagine in 16 parti di dimensione (32, 32)
num_parts = 16
part_height = 32
part_width = 32

parts_list = []
for image in images_batch:
    parts = []
    for j in range(0, image_height, part_height): 
        for i in range(0, image_width, part_width):
            part = image[i:i+part_height, j:j+part_width, :]
            parts.append(part)
    parts_list.append(parts)
#print("PARTS LIST SHAPE: {}".format(parts_list.size()))

# Converti la lista di parti in un array di NumPy
parts_array = np.array(parts_list)
#print("PARTS ARRAY SHAPE: {}".format(parts_array.shape))

# Ora hai un array di dimensione (16, 256, 256, 3) contenente le parti di tutte le immagini



# Passa ogni parte attraverso la rete neurale separatamente
output_parts = []
final_tensor = []
for part in parts_array:
    #print("PARTS SHAPE: {}".format(part.shape)) #(16,32,32,3)
    
    # Divide il tensore in quattro parti lungo l'asse 0
    #tensor_parts = part#np.split(part, 4, axis=0)
    #print("TENSOR PART LEN: {}".format(tensor_parts[1]).shape) #(16,32,32,3)
    print(part.shape)
    res = custom_vgg_model.model(part)
    #print("RES SHAPE: {}".format(res.shape)) #(16,32,32,3)
    
    #input("WAIT")
    
    tensor_parts = res
    # Inizializza una lista vuota per i risultati intermedi
    concatenated_horizontal_parts = []
    
    i = 0
    # Concatenazione orizzontale delle parti
    #for i in range(0,4):
    #    concatenated_horizontal_part = np.concatenate(tensor_parts[i], axis=2)
    #    concatenated_horizontal_parts.append(concatenated_horizontal_part)



    # Concatena le prime quattro parti lungo l'asse orizzontale (asse 2)
    concatenated_horizontal_1 = np.concatenate(tensor_parts[0:4], axis=0)

    # Concatena le seconde quattro parti lungo l'asse orizzontale (asse 2)
    concatenated_horizontal_2 = np.concatenate(tensor_parts[4:8], axis=0)

    # Concatena le terze quattro parti lungo l'asse orizzontale (asse 2)
    concatenated_horizontal_3 = np.concatenate(tensor_parts[8:12], axis=0)

    # Concatena le quarte quattro parti lungo l'asse orizzontale (asse 2)
    concatenated_horizontal_4 = np.concatenate(tensor_parts[12:16], axis=0)

    # Concatena i quattro tensori ottenuti lungo l'asse verticale (asse 1)
    final_result = np.concatenate([concatenated_horizontal_1, concatenated_horizontal_2,
                                concatenated_horizontal_3, concatenated_horizontal_4], axis=1)

    #print(final_result.shape)  # Stampa la nuova dimensione

    #pil_image = Image.fromarray(np.uint8(final_result))

    # Display the image
    #pil_image.show()
    # Concatenazione verticale delle parti orizzontali
    #final_result = np.concatenate(concatenated_horizontal_parts, axis=1)
    #print("FINAL RESULT: {}".format(final_result.shape)) #(16,32,32,3)
    #input("WAIT")
    #part_output = custom_vgg_model.model(part)
    #print("PART OUT SHAPE: {}".format(part_output.shape))
    
    output_parts.append(final_result)
    # Concatenazione lungo l'asse 1 (asse delle profondità)
final_tensor = np.stack(output_parts, axis=0)
#print("_---------------------")
print(final_tensor.shape)  # Stampa la nuova dimensione
#print("OUT SHAPE: {}".format(output_parts.shape))"""





"""img = cv2.imread('../Data/CVUSA/bingmap/19/0000001.jpg')
img = cv2.resize(img, (256, 128), interpolation=cv2.INTER_AREA)
img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
img = img.astype(np.float32)

batch_grd = np.zeros([1, 128, 256, 3], dtype = np.float32)
batch_grd[0,:,:,:] = img
print(custom_vgg_model.model(batch_grd).shape)
#print(custom_vgg_model.call(batch_grd))"""

"""inputs = Input(shape=(128, 512, 3))
custom_vgg_model = VGGModel(inputs)

# Print the model summary
custom_vgg_model.summary()"""