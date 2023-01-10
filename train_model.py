"""Main model training snippet."""

import argparse
import numpy as np
import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from keras.utils.generic_utils import get_custom_objects
from tensorflow import keras

from src import utils
from model import unet
from loss import *
from eval import *

def main(args):
    
    np.random.seed(1603)
    tf.random.set_seed(2021)     
    
    # Checking GPU device availability.
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    data_path = r"data\\ml4h_proj1_colon_cancer_ct\\"

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Gathering arguments for model training
    DEPTH = args.depth
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    EPOCHS = args.epochs
    PADDING = args.padding    
    
    # Building U-Net model
    model = unet((IMG_SIZE[0], IMG_SIZE[1], DEPTH), PADDING)
    # model.summary()

    train_indexes = utils.read_train_ind()
    val_indexes = utils.read_val_ind()
    # Reading training and validation sets.
    file_list= utils.get_file_list(data_path)
    imgs, lbls = utils.read_training_data(data_path)
    img_test, lbls_test = utils.read_val_indices(data_path, file_list, val_indexes)
    has_foreground = utils.get_foreground(lbls)
    class_weight = utils.calculate_class_weights(data_path, train_indexes)

    # Defining training and validation generator
    train_gen = utils.ColonCTSequenceSegmentation(imgs, lbls, class_weight[0], class_weight[1], train_indexes, BATCH_SIZE, IMG_SIZE,
                                                depth=DEPTH, shuffle=True, padding=PADDING,
                                                drop_remainder=True)


    val_gen = utils.ColonCTSequenceSegmentation(imgs, lbls, class_weight[0], class_weight[1], val_indexes, BATCH_SIZE, IMG_SIZE,
                                                depth=DEPTH, shuffle=False, padding=PADDING,
                                                drop_remainder=True)


    optimizer = keras.optimizers.Adam(learning_rate=LR) 
    model.compile(optimizer=optimizer,
                loss=jaccard_distance_loss,
                metrics=[tf.keras.metrics.MeanIoU(num_classes=2),JaccardCoefficient(0.3, "JC_at_0.3"),
                        JaccardCoefficient(0.5, "JC_at_0.5"),
                        JaccardCoefficient(0.7, "JC_at_0.7"),
                        pos_IoU,
                        keras.metrics.Precision(), keras.metrics.Recall()])

    filepath = 'models/my_best_model.hdf5'

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]
    save_callback = [keras.callbacks.ModelCheckpoint(filepath=filepath, monitor="JC_at_0.5", save_best_only=True)]

    hist = model.fit(train_gen,
                    epochs=EPOCHS,
                    validation_data=val_gen,
                    callbacks=[tensorboard_callback,save_callback])



    get_custom_objects().update({'jaccard_distance_loss': jaccard_distance_loss, 
                                "JaccardCoefficient":JaccardCoefficient, "pos_IoU":pos_IoU})

    model_to_test = keras.models.load_model("models/my_best_model.hdf5")


    evaluate_model_iou(imgs, lbls, val_indexes, model_to_test, DEPTH, [0.25, 0.5, 0.75])

if __name__ == "__main__":
    
    # Parsing the arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--img_size", default=(64,64), type=tuple)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=2e-3, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--padding", default="same", type=str)
    
    args = parser.parse_args()
    
    main(args)

