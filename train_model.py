import os, cv2
import numpy as np
from time import time
from utils.metrics import IOUScore, FScore

from nutsflow import *
from nutsml import *

import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from utils.data_prepare import get_iterator_nuts, get_data_iterator, read_csv_data
from model_utils.model import create_unet_model
from model_utils.loss_functions import get_combined_cross_entropy_and_dice_loss_function,process_loss_fn_inputs

from utils.logging import log_scalar, log_sequence_of_scalars

from config import BATCH_SIZE, PATCH_SIZE_W, PATCH_SIZE_H
from config import TRAIN_CLASSES, N_CLASSES, PRECISION
from config import EPOCH, ADAM_LR, ADAM_BETA_1
from keras import backend as KB
from keras.models import Sequential
from keras.layers import Conv2D, SpatialDropout2D, GlobalAveragePooling2D, Input, Dense, UpSampling2D, \
    AveragePooling2D, GlobalMaxPooling2D, Lambda, MaxPooling2D, Flatten, Conv2DTranspose
# from tensorflow.keras.layers.advanced_activations import LeakyReLU
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
from model_utils.model import create_unet_model

BATCH_SIZE = 8

def create_folder_if_absent(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def train_batch(model, sample):
    outp = model.train_on_batch(sample[0], sample[1])
    # outp = (outp,)
    # print(outp)

    return model, outp

def validate_batch(model, sample):
    outp = model.train_on_batch(sample[0], sample[1])
    # outp = (outp,)

    return outp

def train_epoch(epoch, model, train_data, labeldist, data_iterator_nuts, summary_writer, training_step, step_log_interval=10):
    (img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train) = data_iterator_nuts

    train_data_iterator = get_data_iterator(train_data, labeldist, img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train, 
                data_type="train")

    train_error = []
    step=0
    for enum, sample_item in enumerate(train_data_iterator):
        im = sample_item[0][0]
        mask = sample_item[1][0]
        v1 = sample_item[1][1]
        v2 = sample_item[1][2]
        v3 = sample_item[1][3]

        if im.shape[0]!=BATCH_SIZE or mask.shape[0]!=BATCH_SIZE:
            continue

        sample = [im, mask, v1, v2, v3]

        model, error = train_batch(model, sample)

        # IOU_score = error[1]
        # Fscore = error[2]

        # print("error", error)
        # print("metrics",metricss)

        train_error.append(error)

        # mean error of 5 iterations of train_batch
        if step%step_log_interval==0:
            mean_error = np.mean([ x[0] for x in train_error ])
            mean_iou = np.mean([ x[1] for x in train_error ])
            mean_fscore = np.mean([ x[2] for x in train_error ])

            if mean_error>10**(-PRECISION):
                mean_error = mean_error.round(PRECISION)
                mean_iou = mean_iou.round(PRECISION)
                mean_fscore = mean_fscore.round(PRECISION)

            print("Epoch: {}, Step: {}\t-> train error: {}, IOUScore: {}, FScore: {}".format(epoch, step, mean_error, mean_iou, mean_fscore))

        # log for tensorboard view    
        log_scalar(summary_writer, "STEP Train Error", error[0], training_step)

        step+=1
        training_step+=1

    # print("train_error",train_error)
    # print("shape",train_error.shape)

    trainn_error = [ x[0] for x in train_error ]
    train_iou = [ x[1] for x in train_error ]
    train_fscore = [ x[2] for x in train_error ]

    return model, trainn_error, training_step, train_iou, train_fscore
    

def validate_epoch(epoch, model, val_data, data_iterator_nuts, step_log_interval=10):
    (img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train) = data_iterator_nuts

    val_data_iterator = get_data_iterator(val_data, "", img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train, 
                data_type="val")

    val_error = []
    step=0
    for enum, sample_item in enumerate(val_data_iterator):
        im = sample_item[0][0]
        mask = sample_item[1][0]
        v1 = sample_item[1][1]
        v2 = sample_item[1][2]
        v3 = sample_item[1][3]

        if im.shape[0]!=BATCH_SIZE or mask.shape[0]!=BATCH_SIZE:
            continue

        sample = [im, mask, v1, v2, v3]
        error = validate_batch(model, sample)

        val_error.append(error)

        if step%step_log_interval==0:
            mean_error = np.mean([ x[0] for x in val_error ])
            mean_iou = np.mean([ x[1] for x in val_error ])
            mean_fscore = np.mean([ x[2] for x in val_error ])

            if mean_error>10**(-PRECISION):
                mean_error = mean_error.round(PRECISION)
                mean_iou = mean_iou.round(PRECISION)
                mean_fscore = mean_fscore.round(PRECISION)
            print("Epoch: {}, Step: {}\t-> val error: {}, IOUScore: {}, FScore: {}".format(epoch, step, mean_error, mean_iou, mean_fscore))   
            
        step+=1

    val_errorr = [ x[0] for x in val_error ]
    val_iou = [ x[1] for x in val_error ]
    val_fscore = [ x[2] for x in val_error ]

    return val_errorr, val_iou, val_fscore


if __name__=="__main__":

    # **************************************************************************
    train_csv_data_path = r'C:\Users\ASUS\Downloads\code1.1\train_data.csv'
    val_csv_data_path = r'C:\Users\ASUS\Downloads\code1.1\test_data.csv'

    # folders and their path to proccessed inputs
    data_root = r"C:\Users\ASUS\Downloads\code1.1\pre_processed\\"
    processed_img_fol= os.path.join(data_root, "oct_imgs")
    processed_oct_mask_fol= os.path.join(data_root, "oct_masks")
    processed_roi_mask_fol= os.path.join(data_root, "roi_masks")

    # folder to save training weights
    weight_fol = r"C:\Users\ASUS\Downloads\code1.1\weights"

    # folder to save logs viewable by tensorboard
    log_fol = r"C:\Users\ASUS\Downloads\code1.1\logs"
    
    # if not present, keep it None
    pre_trained_model_path = None #r"C:\Users\ASUS\Downloads\code1.1\weights\best_weight.h5"
    # **************************************************************************

    # prepare data for training and validation
    train_data = read_csv_data(train_csv_data_path)
    val_data = read_csv_data(val_csv_data_path)

    labeldist = train_data >> CountValues(1)
    patch_size = (PATCH_SIZE_H, PATCH_SIZE_W)

    # Data loader
    data_iterator_nuts = get_iterator_nuts(processed_img_fol, processed_oct_mask_fol, \
                            processed_roi_mask_fol, patch_size, BATCH_SIZE,)
    
    # create model
    model = create_unet_model(input_shape=(PATCH_SIZE_H, PATCH_SIZE_W, 3))
    print(model.summary())

    # load pre-trained model, if any
    if pre_trained_model_path is not None:
        print("Loading weights : ", pre_trained_model_path)
        model.load_weights(pre_trained_model_path)

    # get loss function
    loss_fn = get_combined_cross_entropy_and_dice_loss_function(BATCH_SIZE, N_CLASSES)

    # the dice index
    # source: https://github.com/qubvel/segmentation_models

    metric = [IOUScore(threshold=0.5), FScore(threshold=0.5)]

    # compile model with loss and optimizer
    model.compile(optimizer=Adam(learning_rate=ADAM_LR, beta_1=ADAM_BETA_1), loss=loss_fn, metrics=metric)

    # create folders to save resources
    create_folder_if_absent(weight_fol)
    log_dir=os.path.join(log_fol, str(int(time())))
    create_folder_if_absent(log_dir)

    summary_writer = tf.summary.create_file_writer(log_dir)

    best_error = float("inf")
    training_step=0

    print("\n")
    print("Starting Training:")
    for epoch in range(EPOCH):
        # train model for single epoch
        model, train_epoch_error, training_step, train_iou, train_fscore = train_epoch(epoch+1, model, train_data, labeldist, data_iterator_nuts, 
                                                                        summary_writer,training_step, step_log_interval=10)

        print("\nEvaluating model on validation data : ")
    # validate model for single epoch
    val_epoch_error, val_iou, val_fscore = validate_epoch(epoch+1, model, val_data, data_iterator_nuts, step_log_interval=10)

    print("*"*55)
    # get the final log values
    mean_train_error=np.mean(train_epoch_error).round(PRECISION)
    mean_train_iou=np.mean(train_iou).round(PRECISION)
    mean_train_fscore=np.mean(train_fscore).round(PRECISION)

    mean_val_error=np.mean(val_epoch_error).round(PRECISION)
    mean_val_iou=np.mean(val_iou).round(PRECISION)
    mean_val_fscore=np.mean(val_fscore).round(PRECISION)

    if mean_train_error>10**(-PRECISION):
        mean_train_error=mean_train_error.round(PRECISION)
        mean_train_iou=mean_train_iou.round(PRECISION)
        mean_train_fscore=mean_train_fscore.round(PRECISION)

    if mean_val_error>10**(-PRECISION):
        mean_val_error=mean_val_error.round(PRECISION)
        mean_val_iou=mean_val_iou.round(PRECISION)
        mean_val_fscore=mean_val_fscore.round(PRECISION)

    print("Summary: Epoch : {},  Train errror : {}, IOUScore_train: {}, FScore_train: {}, Val error : {}, IOUScore_val: {}, FScore_val: {} \n".format\
          (epoch+1, mean_train_error, mean_train_iou, mean_train_fscore, mean_val_error, mean_val_iou, mean_val_fscore))

    log_scalar(summary_writer, "EPOCH Train Error", mean_train_error, epoch+1)
    log_scalar(summary_writer, "EPOCH Val Error", mean_val_error, epoch+1)

    # if log value better, save the weight
    if mean_val_error<best_error:
        weight_path=weight_fol+"/best_weight.h5"
        model.save_weights(weight_path)
        # model.save(weight_path)
        best_error=mean_val_error

    # saving weights each epoch for longer epochs if user wants to
    weight_path=weight_fol+"/"+"model_G_ep_{}.h5".format(epoch+1)
    model.save_weights(weight_path)
    # model.save(weight_path)
