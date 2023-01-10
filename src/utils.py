
"""Utility functions to read and load data for training the U-Net model."""

import skimage.transform as skTrans
import nibabel as nib
from tqdm import tqdm
import numpy as np
import math
import os
import tensorflow as tf


class ColonCTSequenceSegmentation(tf.keras.utils.Sequence):
    """Colon CT Scan loader.
    
    Attributes:
    
        imgs: images
        lbls: labels
        class_w_0: class weights for 0 label
        class_w_1: class weights for 1 label
        indexes: indexes of the samples included in the generator
        batch_size: image batch size
        img_size: image size
        depth: depth of the CT scans
        padding: padding method to apply on images, valid or same.
        shuffle: shuffling 
        drop_reminder: dropping reminder of the CT slices acc. to batch size
        
    Returns:
    
        Data generator for the model training.
    """
    
    def __init__(self, imgs, lbls, class_w_0, class_w_1, indexes, batch_size, img_size=(64,64),
                 depth=3, padding="same", shuffle=True, drop_remainder=True):
        assert depth % 2 == 1
        
        self.imgs = imgs
        self.lbls = lbls
        self.class_w_0 = class_w_0
        self.class_w_1 = class_w_1
        self.indexes = indexes
        self.batch_size = batch_size
        self.img_size = img_size
        self.depth = depth
        self.padding = padding
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
                
        self.center_offset = (self.depth - 1) // 2
        self.slices = []
        for i in range(0, len(self.indexes)):
            img = self.imgs[i]
            img_depth = img.shape[-1]
            
            if padding == "valid":
                valid_slices = img_depth - self.depth + 1
                for first_channel in range(valid_slices):
                    self.slices.append((i, first_channel))
                    
            elif padding == "same":
                same_slices = img_depth
                for middle_channel in range(same_slices):
                    first_channel = middle_channel - self.center_offset
                    self.slices.append((i, first_channel))                
        
        self.on_epoch_end()
        
    def __len__(self):
        length = len(self.slices)/self.batch_size
        if self.drop_remainder:
            return math.floor(length)
        else:
            return math.ceil(length)
        
    def __getitem__(self, index):
        slice_indexes = self.slices[index * self.batch_size:(index+1) * self.batch_size]
        batch_size = len(slice_indexes)
        
        X = np.empty((batch_size, *self.img_size, self.depth))
        y = np.empty((batch_size, *self.img_size, 1))
        
        for k, (i, j) in enumerate(slice_indexes):
            img = self.imgs[i]
            img_height = img.shape[-1]
            
            channels = np.arange(j, j+self.depth).clip(0, img_height-1)
            X[k] = img[:,:,channels]
            
            lbl = self.lbls[i]
            y[k,:,:,0] = lbl[:,:,j+self.center_offset]
            
        w = np.where(y, self.class_w_1, self.class_w_0)
            
        return X, y, w
                
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.slices)

def read_training_data(data_path):
    """Reads training data."""
    
    imgs = []
    lbls = []
    
    for f in tqdm(os.listdir(os.path.join(data_path, 'imagesTr'))):
        if '.DS_Store' in f:
            continue
        img = nib.load(os.path.join(data_path, "imagesTr",f)).get_fdata().astype("float32")
        lbl = nib.load(os.path.join(data_path, 'labelsTr', f)).get_fdata().astype("float32")

        result = skTrans.resize(img, (64,64,img.shape[2]), order=1, preserve_range=True, anti_aliasing=True)
        label = skTrans.resize(lbl, (64,64,lbl.shape[2]), order=0, preserve_range=True, anti_aliasing=False)
        
        imgs.append(result)
        lbls.append(label)
        
    return imgs, lbls

def read_testing_data(data_path):
    """Reads test data."""

    imgs = []

    for f in os.listdir(os.path.join(data_path, 'imagesTs')):
        if '.DS_Store' in f:
            continue
        imgs.append( nib.load(os.path.join(data_path, 'imagesTs', f)).get_fdata().astype("float32"))

    return imgs

    
def read_val_indices(data_path, file_list, val_indexes):
    """Reads indexes of validation set samples."""
    
    img_array=[]
    label_array=[]
    for ind in tqdm(val_indexes):

        file_name = file_list[ind]

        if file_name in tqdm(os.listdir(os.path.join(data_path, 'imagesTr'))):
            
            if '.DS_Store' in file_name:
                continue
            img_array.append( nib.load(os.path.join(data_path, 'imagesTr', file_name)).get_fdata().astype("float32") )
            label_array.append( nib.load(os.path.join(data_path, 'labelsTr', file_name)).get_fdata().astype("float32") )
            
    return img_array, label_array

def get_file_list(data_path):
    """Gets file list from data"""
    
    file_list= os.listdir(os.path.join(data_path, 'imagesTr'))
    
    return(file_list)

def get_foreground(lbls):
    """Gathers boolean list of foreground information from images"""
    
    has_foreground = [np.max(lbl, axis=(0,1)).astype(np.bool) for lbl in lbls]
    
    return(has_foreground)

def calculate_class_weights(data_path, train_indexes):
    """Calculates class weights from samples."""    

    _, lbls = read_training_data(data_path)

    pos = 0
    total = 0
    for i in range(0, len(train_indexes[0:11])):
        lbl = lbls[i]
        pos += lbl.sum()
        total += lbl.size
    neg = total-pos

    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    return class_weight

def read_train_ind():
    """Reads training set indices."""
    train_indexes = [89, 2, 46, 4, 77, 8, 32, 22, 13, 60, 47, 
                     79, 74, 73, 81, 56, 51, 30, 6, 35, 92, 
                     28, 37, 83, 3, 23, 59, 97, 61, 34, 68, 
                     93, 45, 58, 31, 75, 71, 55, 80, 20, 43, 
                     72, 76, 39, 69, 65, 9, 96, 27, 84, 67, 
                     17, 95, 99, 64, 11, 53, 88, 42, 40, 15, 
                     82, 18, 98, 19, 36, 10, 25, 90, 41, 14, 
                     38, 78, 5, 52, 54, 50, 16, 49, 63]
    
    return(train_indexes)

def read_val_ind():
    """Reads validation set indices."""
    val_indexes = [48, 66, 26, 33, 87, 70, 12, 24, 
                21, 29, 91, 62, 44, 86, 94, 57, 85]    
    
    return(val_indexes)
