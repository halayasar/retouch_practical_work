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
from model_utils.loss_functions import get_combined_cross_entropy_and_dice_loss_function,process_loss_fn_inputs, generator_loss, discriminator_loss

from utils.logging import log_scalar, log_sequence_of_scalars

from config import BATCH_SIZE, PATCH_SIZE_W, PATCH_SIZE_H
from config import TRAIN_CLASSES, N_CLASSES, PRECISION
from config import EPOCH, ADAM_LR, ADAM_BETA_1, D_ADAM_BETA_1, D_ADAM_LR
from keras import backend as KB
from keras.models import Sequential
from keras.layers import Conv2D, SpatialDropout2D, GlobalAveragePooling2D, Input, Dense, UpSampling2D, \
    AveragePooling2D, GlobalMaxPooling2D, Lambda, MaxPooling2D, Flatten, Conv2DTranspose
# from tensorflow.keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers import Activation, Concatenate, Resizing
from keras.models import Model
from keras.utils import plot_model
from model_utils.model_layers import Softmax4D
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.layers import Cropping2D
from keras.regularizers import l2
from model_utils.model import create_unet_model
import csv

BATCH_SIZE = 8

def create_folder_if_absent(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def create_discriminator(input_shape1, input_shape2, target_shape):
    input_1 = Input(shape=input_shape1)
    input_2 = Input(shape=input_shape2)

    resized_1 = Resizing(*target_shape)(input_1)
    resized_2 = Resizing(*target_shape)(input_2)

    x = Concatenate()([resized_1, resized_2])

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    x = GlobalMaxPooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=[input_1, input_2], outputs=x)

F1_metric = FScore()

@tf.function
def train_batch(generator,  discriminator, gen_optimizer, dis_optimizer, sample):

    # computing gradients of the loss wrt the variables of the generator & discriminator networks
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        # generate batch of samples using the generator network
        pred = generator(sample[0], training=True)

        # computes the outputs of the discriminator network on both 
        # the generated samples (pred_output) & GT samples (gt_output)
        pred_output = discriminator([sample[0], pred], training=True)    
        gt_output = discriminator([sample[0], sample[1]], training=True)

        # loss computation
        gen_loss = generator_loss(pred_output)
        disc_loss = discriminator_loss(gt_output, pred_output)

        # L2 regularization on the discriminator weights
        weight_decay = 1e-3
        l2_loss = weight_decay * tf.reduce_sum([tf.nn.l2_loss(v) for v in discriminator.trainable_variables])
        # total loss
        total_loss = 1/3*get_combined_cross_entropy_and_dice_loss_function(generator, BATCH_SIZE, N_CLASSES)(sample[1], pred) + 1/3*disc_loss + l2_loss
        
        # evaluate GAN model
        # F1 score between the GT & the generated samples  
        f1_Score = F1_metric( tf.cast(sample[1], tf.float32), pred)

    gen_gradients = gen_tape.gradient(total_loss, generator.trainable_variables)
    dis_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    dis_optimizer.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))

    return total_loss, disc_loss, f1_Score


def validate_batch(model, discriminator, sample):
    # outp = model.test_on_batch(sample[0], sample[1])

    # return outp
    pred = model(sample[0], training=True)

    pred_output = discriminator([sample[0], pred])       
    # gt_output = discriminator(sample[1])    

    gen_loss = generator_loss(pred_output)
    # disc_loss = discriminator_loss(gt_output, pred_output)

    model_loss =mask_loss(sample[1], pred) #+ 1/3*gen_loss  
    # model_loss - mask_loss(sample[1], pred) + 0.5*disc_loss

    f1_Score = F1_metric( tf.cast(sample[1], tf.float32), pred)    

    return model_loss, f1_Score


def train_epoch(epoch, model, discriminator, train_data, labeldist, data_iterator_nuts, summary_writer, training_step, step_log_interval=10):
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

        model_loss, disc_loss, f1_Score  = train_batch(model, discriminator, optimizer, discriminator_optimizer, sample)
        # model_loss, disc_loss  = train_batch(model, discriminator, sample)

        # print("error", error)
        # print("metrics",metricss)

        # train_error.append(error)
        train_error.append((model_loss, disc_loss, f1_Score))

        # mean error of 5 iterations of train_batch
        if step%step_log_interval==0:
            mean_error = np.mean([ x[0] for x in train_error ])
            mean_disc_loss = np.mean([ x[1] for x in train_error ])
            mean_fscore = np.mean([ x[2] for x in train_error ])

            if mean_error>10**(-PRECISION):
                mean_error = mean_error.round(PRECISION)
                mean_disc_loss = mean_disc_loss.round(PRECISION)
                mean_fscore = mean_fscore.round(PRECISION)

            print("Epoch: {}, Step: {}\t-> train error: {}, discriminator error: {}, FScore: {}".format(epoch, step, mean_error, mean_disc_loss, mean_fscore))

        # log for tensorboard view    
        # log_scalar(summary_writer, "STEP Train Error", mean_error, training_step)

        step+=1
        training_step+=1

    # print("train_error",train_error)
    # print("shape",train_error.shape)

    trainn_error = [ x[0] for x in train_error ]
    train_disc_loss = [ x[1] for x in train_error ]
    train_fscore = [ x[2] for x in train_error ]

    return model, trainn_error, training_step, train_disc_loss, train_fscore
    

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
        model_loss, f1_Score = validate_batch(model, discriminator, sample)

        val_error.append((model_loss, f1_Score))

        if step%step_log_interval==0:
            mean_error = np.mean([ x[0] for x in val_error ])
            mean_fscore = np.mean([ x[1] for x in val_error ])

            if mean_error>10**(-PRECISION):
                mean_error = mean_error.round(PRECISION)
                mean_fscore = mean_fscore.round(PRECISION)
            print("Epoch: {}, Step: {}\t-> val error: {}, FScore: {}".format(epoch, step, mean_error, mean_fscore))   
            
        step+=1

    val_errorr = [ x[0] for x in val_error ]
    val_fscore = [ x[1] for x in val_error ]

    return val_errorr, val_fscore


if __name__=="__main__":

    # **************************************************************************
    train_csv_data_path= r'C:\Users\halay\Desktop\project\train_data.csv'
    val_csv_data_path= r'C:\Users\halay\Desktop\project\test_data.csv'

    # folders and their path to proccessed inputs
    data_root = r"C:\Users\halay\Desktop\project\pre_processed\\"
    processed_img_fol= os.path.join(data_root, "oct_imgs")
    processed_oct_mask_fol= os.path.join(data_root, "oct_masks")
    processed_roi_mask_fol= os.path.join(data_root, "roi_masks")

    # folder to save training weights
    weight_fol=r"C:\Users\halay\Desktop\project\weights_adv"

    # folder to save logs viewable by tensorboard
    log_fol=r"C:\Users\halay\Desktop\project\logs"

    # if not present, keep it None
    pre_weight_file= None #r'C:\Users\halay\Desktop\project\weights\best_weight.h5' 
    # **************************************************************************

    # prepare data for training and validation
    train_data = read_csv_data(train_csv_data_path)
    val_data = read_csv_data(val_csv_data_path)

    

    input_shape1 = (None, None, 3)
    input_shape2 =  (None, None, N_CLASSES)
    target_shape = (256, 128)  

    discriminator = create_discriminator(input_shape1, input_shape2, target_shape)
    discriminator_optimizer = Adam(learning_rate=D_ADAM_LR, beta_1=D_ADAM_BETA_1)

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
    # print(model.summary())

    optimizer = Adam(learning_rate=ADAM_LR, beta_1=ADAM_BETA_1)

    # load pre-trained model, if any
    if pre_weight_file is not None:
        print("Loading weights : ", pre_weight_file)
        model.load_weights(pre_weight_file)


    # create folders to save resources
    create_folder_if_absent(weight_fol)
    log_dir=os.path.join(log_fol, str(int(time())))
    create_folder_if_absent(log_dir)

    summary_writer = tf.summary.create_file_writer(log_dir)

    best_error = float("inf")
    training_step=0

    EPOCH = 10
    error_hold = []
    print("Starting Training:")

    train_err = []
    val_err = []

    for epoch in range(EPOCH):

        # train model for single epoch
        model, train_epoch_error, training_step, train_disc_loss, train_fscore = train_epoch(epoch+1, model, discriminator, train_data, labeldist, data_iterator_nuts, 
                                                                        summary_writer,training_step, step_log_interval=10)

        print("\nEvaluating model on validation data : ")
        # validate model for single epoch
        val_epoch_error, val_fscore = validate_epoch(epoch+1, model, val_data, data_iterator_nuts, step_log_interval=10)

        print("*"*65)

        error_hold.append(
                [EPOCH, np.mean([train_epoch_error]), np.mean([val_epoch_error]),
                np.std([train_epoch_error]), np.std([val_epoch_error])])
        
        # get the final log values
        mean_train_error=np.mean(train_epoch_error).round(PRECISION)
        mean_train_disc_loss=np.mean(train_disc_loss).round(PRECISION)
        mean_train_fscore=np.mean(train_fscore).round(PRECISION)

        mean_val_error=np.mean(val_epoch_error).round(PRECISION)
        mean_val_fscore=np.mean(val_fscore).round(PRECISION)

        if mean_train_error>10**(-PRECISION):
            mean_train_error=mean_train_error.round(PRECISION)
            mean_train_disc_loss=mean_train_disc_loss.round(PRECISION)
            mean_train_fscore=mean_train_fscore.round(PRECISION)

        if mean_val_error>10**(-PRECISION):
            mean_val_error=mean_val_error.round(PRECISION)
            mean_val_fscore=mean_val_fscore.round(PRECISION)

        print("Summary: Epoch : {},  Train error : {}, Discriminator error: {}, FScore_train: {} \n".format\
            (epoch+1, mean_train_error, mean_train_disc_loss, mean_train_fscore))
        print("Summary: Epoch : {},  Val error : {}, FScore_val: {} \n".format\
            (epoch+1,mean_val_error, mean_val_fscore))
        # print("Summary: Epoch : {},  Train errror : {}, Val error : {} \n".format\
        #       (epoch+1, mean_train_error, mean_val_error))


        train_err.append(mean_train_error)
        val_err.append(mean_val_error)

        log_scalar(summary_writer, "EPOCH Train Error", mean_train_error, mean_val_error, epoch+1)
        # log_scalar(summary_writer, "EPOCH Val Error", mean_val_error, epoch+1)

        # if log value better, save the weight
        if mean_val_error<best_error:
            weight_path=weight_fol+"/best_weight.h5"
            model.save_weights(weight_path)
            best_error=mean_val_error

        # saving weights each epoch for longer epochs if user wants to
        weight_path=weight_fol+"/"+"model_ep_{}.h5".format(epoch+1)
        model.save_weights(weight_path)

        with open(r'C:\Users\halay\OneDrive\Desktop\project\training_logs.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, mean_train_error, mean_train_disc_loss, mean_train_fscore, mean_val_error, mean_val_fscore])
