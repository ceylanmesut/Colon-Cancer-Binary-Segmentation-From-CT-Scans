
"""U-Net model."""

from tensorflow import keras
from keras import layers


def unet(img_size, padding):
    """The U-Net model.
    
    This code is adapted from:
    https://keras.io/examples/vision/oxford_pets_image_segmentation/
    """
    
    inputs = keras.Input(shape=img_size)
    x = layers.Conv2D(64, 3, padding=padding)(inputs) 
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(rate=0.2, seed=43)(x) # New addition
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(rate=0.2, seed=43)(x) # New addition
    x = layers.ReLU()(x)
    
    # Contraction
    skips = []  # the skips we will use in the expansion
    # for filters in [128, 256, 512, 1024]: # commented out for new experiment
    for filters in [8, 16, 32, 64]:
    
        # save output for skip
        skips.append(x)
        
        x = layers.MaxPooling2D(2, strides=2)(x)
        x = layers.Conv2D(filters, 3, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
    # Expansion
    # for filters in [512, 256, 128, 64]: # commented out for new experiment
    for filters in [64, 32, 16, 8]:        
        # up-conv
        x = layers.Conv2DTranspose(filters, 2, 2)(x)
        x = layers.BatchNormalization()(x)
        
        # concatenate with cropped skip
        skip = skips.pop()
        x = layers.Concatenate()([skip, x])
        
        # conv 3x3, ReLU
        x = layers.Conv2D(filters, 3, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)
    return model