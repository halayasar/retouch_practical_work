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
from time import time
import csv

from utils.metrics import IOUScore, FScore
from model_utils.loss_functions import get_combined_cross_entropy_and_dice_loss_function,process_loss_fn_inputs
from model_utils.model import create_unet_model

# ***********************************************

train_csv_data_path= r'C:\Users\ASUS\Downloads\code1.1\train_data.csv'
val_csv_data_path= r'C:\Users\ASUS\Downloads\code1.1\test_data.csv'

# ***********************************************
    

BATCH_SIZE = 16
BORDER_WIDTH = 46

def set_trainability(model, trainable=True):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def print_trainability(model):
    print('trainble: ',)
    for l in model.layers:
        if l.trainable:
            print (l.name,)
    print( " ")

    print ('non trainble: ',)
    for l in model.layers:
        if not l.trainable:
            print( l.name,)

    print(" ")


def binarize_softmax(x):
    import tensorflow as tf
    x = K.argmax(x, axis=-1)
    # x = tf.cast(x, dtype=tf.int8)
    x = K.one_hot(x, num_classes=N_CLASSES)

    return x


def binarize_softmax_output_shape(input_shape):
    return input_shape


def read_csv_data(csv_data_path):
    csv_data = ReadPandas(csv_data_path, dropnan=True)
    data = csv_data >> Collect()   
    return data


train_data = read_csv_data(train_csv_data_path)
val_data = read_csv_data(val_csv_data_path)

# get loss function
loss_fn = get_combined_cross_entropy_and_dice_loss_function(BATCH_SIZE, N_CLASSES)

def retouch_discriminator(input_shape=(224, 224, 3), regularize_weight=0.0001):
    from keras.utils import plot_model
    
    in_image = Input(shape=input_shape)
    in_mask = Input(shape=(input_shape[0], input_shape[1], N_CLASSES))

    conv1_0_im = Conv2D(64, (7, 7), activation='relu', name='conv1_0_', padding='same', data_format='channels_last',
                        kernel_regularizer=l2(regularize_weight))(in_image)
    conv1_1_im = Conv2D(64, (3, 3), activation='relu', name='conv1_1_', data_format='channels_last',
                        kernel_regularizer=l2(regularize_weight))(conv1_0_im)
    conv1_2_im = Conv2D(64, (3, 3), name='conv1_2_', data_format='channels_last',
                        kernel_regularizer=l2(regularize_weight))(conv1_1_im)
    conv1_2_im = BatchNormalization(axis=-1, name='bn1_')(conv1_2_im)
    conv1_2_im = Activation('relu')(conv1_2_im)

    conv1_0_mask = Conv2D(64, (7, 7), activation='relu', name='conv1_0_m', padding='same', data_format='channels_last',
                          kernel_regularizer=l2(regularize_weight))(in_mask)
    conv1_1_mask = Conv2D(64, (3, 3), activation='relu', name='conv1_1_m', data_format='channels_last',
                          kernel_regularizer=l2(regularize_weight))(conv1_0_mask)
    conv1_2_mask = Conv2D(64, (3, 3), name='conv1_2_m', data_format='channels_last',
                          kernel_regularizer=l2(regularize_weight))(conv1_1_mask)
    conv1_2_mask = BatchNormalization(axis=-1, name='bn1_m')(conv1_2_mask)
    conv1_2_mask = Activation('relu')(conv1_2_mask)

    conv1_2 = concatenate([conv1_2_im, conv1_2_mask], axis=-1, name='merge_in')

    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1', data_format='channels_last')(conv1_2)
    conv2_1 = Conv2D(128, (3, 3), activation='relu', name='conv2_1', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(pool1)
    conv2_2 = Conv2D(128, (3, 3), name='conv2_2', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(conv2_1)
    conv2_2 = BatchNormalization(axis=-1, name='bn2')(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)

    pool2 = MaxPooling2D(pool_size=(2, 1), name='pool2', data_format='channels_last')(conv2_2)
    conv3_1 = Conv2D(256, (3, 3), activation='relu', name='conv3_1', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(pool2)
    conv3_2 = Conv2D(256, (3, 3), name='conv3_2', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(conv3_1)
    conv3_2 = BatchNormalization(axis=-1, name='bn3')(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)

    pool3 = MaxPooling2D(pool_size=(2, 1), name='pool3', data_format='channels_last')(conv3_2)
    conv4_1 = Conv2D(512, (3, 3), activation='relu', name='conv4_1', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(pool3)
    conv4_2 = Conv2D(512, (3, 3), name='conv4_2', data_format='channels_last',
                     kernel_regularizer=l2(regularize_weight))(conv4_1)
    conv4_2 = BatchNormalization(axis=-1, name='bn4')(conv4_2)
    conv4_2 = Activation('relu')(conv4_2)

    apool = GlobalAveragePooling2D(data_format='channels_last')(conv4_2)
    disc_out = Dense(2, activation='softmax', name='disc_out', kernel_regularizer=l2(regularize_weight))(apool)

    model = Model(inputs=[in_image, in_mask], outputs=disc_out)

    #********************************************************************

    plot_model(model, to_file = r'C:\Users\ASUS\Downloads\code1.1\discriminator_model.png', show_shapes=True)

    #********************************************************************

    model.summary()

    return model

def rearange_cols(sample):
    """
    Re-arrange the incoming data stream to desired outputs
    :param sample: (image_name, vendor, root, slice, is_IRF, is_SRF, is_PED)
    :return: (img, mask, IRF_label, SRF_label, PED_label, roi_m)
    """
    img = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
    mask = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
    IRF_label = sample[4]
    SRF_label = sample[5]
    PED_label = sample[6]
    roi_m = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'

    return (img, mask, IRF_label, SRF_label, PED_label, roi_m)

# training image augementation (flip-lr and rotate)
def myrotate(image, angle):
    return skt.rotate(image, angle, preserve_range=True, order=0).astype('uint8')

# Filter to drop some non-pathology patches
def drop_patch(sample, drop_prob=0.9):
    """
    Randomly drop a patch from iterator if there is no pathology
    """
    if (int(sample[2]) == 0) and (int(sample[3]) == 0) and (int(sample[4]) == 0):
        return float(np.random.random_sample()) < drop_prob
    else:
        return False
    
def train_batch(sample):
    # train the discriminator
    # generate fake images
    set_trainability(model_D, trainable=True)
    generated_images = model_G.predict(sample[0], batch_size=BATCH_SIZE)
    generated_images = np.argmax(generated_images, axis=-1).astype(np.uint8)
    generated_images = np.eye(N_CLASSES, dtype=np.uint8)[generated_images]

    images = np.copy(sample[0][:, BORDER_WIDTH:-BORDER_WIDTH, BORDER_WIDTH:-BORDER_WIDTH, :])
    X_img = np.concatenate((sample[0][:, BORDER_WIDTH:-BORDER_WIDTH, BORDER_WIDTH:-BORDER_WIDTH, :], images), axis=0)
    X_mask = np.concatenate((sample[1][:, BORDER_WIDTH:-BORDER_WIDTH, BORDER_WIDTH:-BORDER_WIDTH, :], generated_images), axis=0)
    y = np.asarray([[0, 1]]*BATCH_SIZE + [[1, 0]]*BATCH_SIZE, dtype=np.int8)

    D_error = model_D.train_on_batch([X_img, X_mask], y)

    # train the Generator
    set_trainability(model_D, trainable=False)
    y = np.asarray([[0, 1]] * BATCH_SIZE, dtype=np.int8)

    G_error = model_D_G.train_on_batch(sample[0], [y, sample[1]])

    print("D_error",D_error)
    print("G_error",G_error)

    return (D_error, G_error[2])

def test_batch(sample):
    outp = model_G.test_on_batch(sample[0], sample[1])
    return (outp,)

# from utils.logging import log_scalar, log_sequence_of_scalars

# def train_epoch(epoch, train_data, labeldist, data_iterator_nuts):
#     (img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train) = data_iterator_nuts

#     train_data_iterator = get_data_iterator(train_data, labeldist, img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train, 
#                 data_type="train")

#     train_error = []
#     for enum, sample_item in enumerate(train_data_iterator):
#         im = sample_item[0][0]
#         mask = sample_item[1][0]
#         v1 = sample_item[1][1]
#         v2 = sample_item[1][2]
#         v3 = sample_item[1][3]

#         if im.shape[0]!=BATCH_SIZE or mask.shape[0]!=BATCH_SIZE:
#             continue

#         sample = [im, mask, v1, v2, v3]

#         error = train_batch(sample)

#         # print("error", error)

#         train_error.append(error)

#         # mean error of train_batch
#         mean_error = np.mean([ x[0] for x in train_error ])
#         # mean_iou = np.mean([ x[1] for x in train_error ])
#         # mean_fscore = np.mean([ x[2] for x in train_error ])

#         if mean_error>10**(-PRECISION):
#             mean_error = mean_error.round(PRECISION)
#             # mean_iou = mean_iou.round(PRECISION)
#             # mean_fscore = mean_fscore.round(PRECISION)

#         # print("Epoch: {}, Step: {}\t-> train error: {}, IOUScore: {}, FScore: {}".format(epoch, step, mean_error, mean_iou, mean_fscore))
#         print("Epoch: {} \t-> train error: {}".format(epoch,  mean_error))


#     # print("train_error",train_error)
#     # print("shape",train_error.shape)

#     trainn_error = [ x[0] for x in train_error ]
#     # train_iou = [ x[1] for x in train_error ]
#     # train_fscore = [ x[2] for x in train_error ]

#     return trainn_error #, train_iou, train_fscore
    

# def validate_epoch(epoch, val_data, data_iterator_nuts):
#     (img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train) = data_iterator_nuts

#     val_data_iterator = get_data_iterator(val_data, "", img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train, 
#                 data_type="val")

#     val_error = []
#     for enum, sample_item in enumerate(val_data_iterator):
#         im = sample_item[0][0]
#         mask = sample_item[1][0]
#         v1 = sample_item[1][1]
#         v2 = sample_item[1][2]
#         v3 = sample_item[1][3]

#         if im.shape[0]!=BATCH_SIZE or mask.shape[0]!=BATCH_SIZE:
#             continue

#         sample = [im, mask, v1, v2, v3]
#         error = test_batch(sample)

#         val_error.append(error)

#     mean_error = np.mean([ x[0] for x in val_error ])
#     # mean_iou = np.mean([ x[1] for x in val_error ])
#     # mean_fscore = np.mean([ x[2] for x in val_error ])

#     if mean_error>10**(-PRECISION):
#         mean_error = mean_error.round(PRECISION)
#         # mean_iou = mean_iou.round(PRECISION)
#         # mean_fscore = mean_fscore.round(PRECISION)
#     # print("Epoch: {}\t-> val error: {}, IOUScore: {}, FScore: {}".format(epoch,  mean_error, mean_iou, mean_fscore))   
#     print("Epoch: {}\t-> val error: {}".format(epoch,  mean_error))  
            

#     val_errorr = [ x[0] for x in val_error ]
#     # val_iou = [ x[1] for x in val_error ]
#     # val_fscore = [ x[2] for x in val_error ]

#     return val_errorr #, val_iou, val_fscore


if __name__=="__main__":



    #**************************************

    log_cols_train = LogCols(r'C:\Users\ASUS\Downloads\code1.1\logs\train_log.csv', cols=None, colnames=('D error', 'G error'))
    log_cols_test = LogCols(r'C:\Users\ASUS\Downloads\code1.1\logs\test_log.csv', cols=None, colnames=('D error', 'G error'))
    log_fol=r"C:\Users\ASUS\Downloads\code1.1\logs_adversarial"
    weight_fol = r"C:\Users\ASUS\Downloads\code1.1\weights_adversarial"
    DATA_ROOT = r'C:\Users\ASUS\Downloads\code1.1\pre_processed'
    processed_img_fol= os.path.join(DATA_ROOT, "oct_imgs")
    processed_oct_mask_fol= os.path.join(DATA_ROOT, "oct_masks")
    processed_roi_mask_fol= os.path.join(DATA_ROOT, "roi_masks")
    generator_pre_weight_file = None # r'C:\Users\ASUS\Downloads\code1.1\weights\best_weight.h5'
    
    #**************************************

    log_dir=os.path.join(log_fol, str(int(time())))
    summary_writer = tf.summary.create_file_writer(log_dir)
    labeldist = train_data >> CountValues(1)

    TransformImage.register('myrotate', myrotate)

    augment_1 = (AugmentImage((0, 1, 5))
                    .by('identical', 1.0)
                    .by('fliplr', 0.5))

    augment_2 = (AugmentImage((0, 1, 5))
                    .by('identical', 1.0)
                    .by('myrotate', 0.5, [0, 10]))


    # setting up image ad mask readers
    imagepath = DATA_ROOT + '\oct_imgs\*'
    maskpath = DATA_ROOT + '\oct_masks\*'
    roipath = DATA_ROOT + r'\roi_masks\*'
    img_reader = ReadOCT(0, imagepath)
    mask_reader = ReadImage(1, maskpath)
    roi_reader = ReadImage(5, roipath)

    # randomly sample image patches from the interesting region (based on entropy)
    image_patcher = ImagePatchesByMaskRetouch_resampled(imagecol=0, maskcol=1, IRFcol=2, SRFcol=3, PEDcol=4, roicol=5,
                                                        pshape=(PATCH_SIZE_H, PATCH_SIZE_W),
                                                        npos=7, nneg=2, pos=1, use_entropy=True, patch_border=42)

    # img_viewer = ViewImage(imgcols=(0, 1), layout=(1, 2), pause=1)

    # building image batches
    build_batch_train = (BuildBatch(BATCH_SIZE, prefetch=0)
                            .input(0, 'image', 'float32', channelfirst=False)
                            .output(1, 'one_hot', 'float32', 4)
                            .output(2, 'one_hot', 'float32', 2)
                            .output(3, 'one_hot', 'float32', 2)
                            .output(4, 'one_hot', 'float32', 2))

    is_cirrus = lambda v: v[1] == 'Cirrus'
    is_topcon = lambda v: v[1] == 'Topcon'
    is_spectralis = lambda v: v[1] == 'Spectralis'

    patch_size = (PATCH_SIZE_H, PATCH_SIZE_W)
    filter_batch_shape = lambda s: s[0][0].shape[0] == BATCH_SIZE

    patch_mean = 128.
    patch_sd = 128.
    remove_mean = lambda s: (s.astype(np.float32) - patch_mean) / patch_sd

    # Data loader
    data_iterator_nuts = get_iterator_nuts(processed_img_fol, processed_oct_mask_fol, \
                        processed_roi_mask_fol, patch_size, BATCH_SIZE,)

    # define the metrics
    metric = [IOUScore(threshold=0.5), FScore(threshold=0.5)]

    # define the discriminator model
    model_D = retouch_discriminator(input_shape=(PATCH_SIZE_H-BORDER_WIDTH*2, PATCH_SIZE_W-BORDER_WIDTH*2, 3))
    set_trainability(model_D, trainable=True)
    model_D.compile(optimizer=Adam(learning_rate=ADAM_LR, beta_1=ADAM_BETA_1), loss='categorical_crossentropy')#, metrics=metric)

    # define the generator model (segmentation model)
    model_G = create_unet_model(input_shape=(PATCH_SIZE_H, PATCH_SIZE_W, 3))
    model_G.compile(optimizer=Adam(learning_rate=ADAM_LR, beta_1=ADAM_BETA_1), loss=loss_fn, metrics=metric)
    set_trainability(model_D, trainable=False)

    im_input = Input(shape=(PATCH_SIZE_H, PATCH_SIZE_W, 3))
    im2_input = Cropping2D(cropping=((BORDER_WIDTH, BORDER_WIDTH), (BORDER_WIDTH, BORDER_WIDTH)), data_format='channels_last')(im_input)

    # print("im_input",im_input.shape)
    # print("im2_input",im2_input.shape)

    G_out = model_G(im_input)
    # print("G_out",G_out.shape)

    G_out_bz = Lambda(binarize_softmax, output_shape=binarize_softmax_output_shape)(G_out)
    # print("G_out_bz",G_out_bz.shape)

    D_out = model_D([im2_input, G_out_bz])
    # print("D_out",D_out.shape)

    # define the GAN model
    model_D_G = Model([im_input], [D_out, G_out])
    model_D_G.compile(optimizer=Adam(learning_rate=ADAM_LR, beta_1=ADAM_BETA_1), loss=['categorical_crossentropy', loss_fn])#, metrics=metric)

    if generator_pre_weight_file is not None:
        model_G.load_weights(generator_pre_weight_file)

    filter_batch_shape = lambda s: s[0][0].shape[0] == BATCH_SIZE

    patch_mean = 128.
    patch_sd = 128.
    remove_mean = lambda s: (s.astype(np.float32) - patch_mean) / patch_sd


    best_error = float("inf")
    print('Starting network training')
    for e in range(0, EPOCH):
        print( "Training Epoch", str(e+1))
        train_error = train_data >> Shuffle(1000) >> Map(
            rearange_cols) >> img_reader >> mask_reader >> roi_reader >> augment_1 >> augment_2 >> Shuffle(
            100) >> image_patcher >> MapCol(0, remove_mean) >> Shuffle(1000) >> FilterFalse(
            drop_patch) >> build_batch_train >> Filter(filter_batch_shape) >> Map(
            train_batch) >> log_cols_train >> Consume()
        
        # print("tr error",train_error)

        print("Testing Epoch", str(e+1))
        val_error = val_data >> Map(
            rearange_cols) >> img_reader >> mask_reader >> roi_reader >> image_patcher >> MapCol(0,
            remove_mean) >> FilterFalse(
            drop_patch) >> build_batch_train >> Filter(
            filter_batch_shape) >> Map(test_batch) >> log_cols_test >> Collect()
        
        val_error = np.mean([v[0] for v in val_error])
        # print("val_error error",val_error)

        print(f'Epoch {e+1}, Validation error : {val_error}')
        if val_error < best_error:
            print( f'saving weights at epoch: {e+1}, val_error: {val_error}')
            model_G.save_weights(weight_fol+"/best_gan_weights.h5")
            best_error=val_error
        

    # saving weights each epoch for longer epochs if user wants to
    weight_path=weight_fol+"/"+"gan_weights_ep{}".format(e+1)
    model_G.save_weights(weight_path)

    with open('/content/drive/MyDrive/project/adv_training_logs.csv', mode='a') as file:
      writer = csv.writer(file)
      writer.writerow([e+1, val_error])