from utils.custom_nuts import  ReadOCT, ImagePatchesByMaskRetouch_resampled
from nutsflow import *
from nutsml import *

import numpy as np
import os
from config import *
import skimage.transform as skt
from keras.optimizers import  Adam
from keras.layers import Input
from keras.models import Model
from keras.layers import Cropping2D, Lambda
import keras.backend as K
from keras import backend as KB
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, SpatialDropout2D, GlobalAveragePooling2D, Input, Dense, UpSampling2D, \
    AveragePooling2D, GlobalMaxPooling2D, Lambda, MaxPooling2D, Flatten, Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers import Activation
from keras.models import Model
from keras.utils import plot_model
from model_utils.model_layers import Softmax4D
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.layers import Cropping2D
from keras.regularizers import l2
from utils.data_prepare import get_iterator_nuts, get_data_iterator, read_csv_data
import keras.backend as K
from model_utils.loss_functions import get_combined_cross_entropy_and_dice_loss_function,process_loss_fn_inputs
from model_utils.model import create_unet_model
from time import time
from model_utils.model import create_unet_model
import csv

from utils.metrics import IOUScore, FScore

BATCH_SIZE = 16


if __name__=="__main__":

    #*****************************************

    val_csv_data_path= r'C:\Users\ASUS\Downloads\code1.1\test_data.csv'
    DATA_ROOT = r'C:\Users\ASUS\Downloads\code1.1\pre_processed'
    processed_img_fol= os.path.join(DATA_ROOT, "oct_imgs")
    processed_oct_mask_fol= os.path.join(DATA_ROOT, "oct_masks")
    processed_roi_mask_fol= os.path.join(DATA_ROOT, "roi_masks")
    generator_pre_weight_file = r'C:\Users\ASUS\Downloads\retouch\outputs\best_weight.h5'

    #*****************************************

    def read_csv_data(csv_data_path):
        csv_data = ReadPandas(csv_data_path, dropnan=True)
        data = csv_data >> Collect()   
        return data

    val_data = read_csv_data(val_csv_data_path)

    patch_size = (PATCH_SIZE_H, PATCH_SIZE_W)
    loss_fn = get_combined_cross_entropy_and_dice_loss_function(BATCH_SIZE, N_CLASSES)

    # Data loader
    data_iterator_nuts = get_iterator_nuts(processed_img_fol, processed_oct_mask_fol, \
                            processed_roi_mask_fol, patch_size, BATCH_SIZE,)

    # define the metrics & the model
    BORDER_WIDTH = 46

    metric = [IOUScore(threshold=0.5), FScore(threshold=0.5)]

    model_G = create_unet_model(input_shape=(PATCH_SIZE_H, PATCH_SIZE_W, 3))
    model_G.compile(optimizer=Adam(learning_rate=ADAM_LR, beta_1=ADAM_BETA_1), loss=loss_fn, metrics=metric)

    # validation dataset loader

    (img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train) = data_iterator_nuts

    val_data_iterator = get_data_iterator(val_data, "", img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train, 
                    data_type="val")

    model_G.load_weights(generator_pre_weight_file)

    results = model_G.evaluate(val_data_iterator, verbose=1, batch_size = BATCH_SIZE)