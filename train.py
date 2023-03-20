import os, cv2
import numpy as np
from time import time

from nutsflow import *
from nutsml import *

import tensorflow as tf
from keras.optimizers import Adam

from utils.data_prepare import get_iterator_nuts, get_data_iterator, read_csv_data
from model_utils.model import create_unet_model, create_discriminator
from model_utils.loss_functions import get_combined_cross_entropy_and_dice_loss_function
from model_utils.loss_functions import generator_loss, discriminator_loss
from utils.framework_utils import KerasObject
from utils.metrics import FScore

from utils.logging import log_scalar, log_sequence_of_scalars

from config import BATCH_SIZE, PATCH_SIZE_W, PATCH_SIZE_H
from config import N_CLASSES, PRECISION
from config import EPOCH, ADAM_LR, ADAM_BETA_1, D_ADAM_LR, D_ADAM_BETA_1
import csv

BORDER_WIDTH = 46
EPOCH = 100 
BATCH_SIZE = 16
ADAM_LR = 0.0002
D_ADAM_LR = 0.0002
ADAM_BETA_1 = 0.7

KerasObject.set_submodules(
    backend=tf.keras.backend,
    layers=tf.keras.layers,
    models=tf.keras.models,
    utils=tf.keras.utils,
)


# ****************************************************************************************

mask_loss = get_combined_cross_entropy_and_dice_loss_function(BATCH_SIZE, N_CLASSES)
F1_metric = FScore()


def create_folder_if_absent(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


@tf.function
def train_batch(generator,  discriminator, gen_optimizer, dis_optimizer, sample):

    # computing gradients of the loss wrt the variables of the generator & discriminator networks
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # generate batch of samples using the generator network
        pred = generator(sample[0], training=True)

        # computes the outputs of the discriminator network on both 
        # the generated samples (pred_output) & GT samples (gt_output)
        pred_output = discriminator(pred, training=True)        # discriminate on mask predicted
        gt_output = discriminator(sample[1], training=True)     # discriminate on mask ground truth 

        # loss computation
        gen_loss = generator_loss(pred_output)
        disc_loss = discriminator_loss(gt_output, pred_output)

        # overall loss of model: computed as weighted sum of the generator loss & 
        # a mask loss that compares the generated samples to the GT samples
        model_loss = 1/3*gen_loss + mask_loss(sample[1], pred)

        # F1 score between the GT & the generated samples  
        f1_Score = F1_metric( tf.cast(sample[1], tf.float32), pred)


    gen_gradients = gen_tape.gradient(model_loss, generator.trainable_variables)
    dis_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    dis_optimizer.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))

    return model_loss, disc_loss, f1_Score


def validate_batch(generator, discriminator, sample):
    # outp = model.test_on_batch(sample[0], sample[1])
    # outp = (outp,)

    pred = generator(sample[0], training=True)

    pred_output = discriminator(pred)       
    # gt_output = discriminator(sample[1])    

    gen_loss = generator_loss(pred_output)
    # disc_loss = discriminator_loss(gt_output, pred_output)

    model_loss = 1/3*gen_loss + mask_loss(sample[1], pred)

    f1_Score = F1_metric( tf.cast(sample[1], tf.float32), pred)    

    return model_loss, f1_Score


def train_epoch(epoch, model, discriminator, optimizer, dis_optimizer, train_data, labeldist, data_iterator_nuts, summary_writer, training_step, step_log_interval=5):
    (img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train) = data_iterator_nuts

    train_data_iterator = get_data_iterator(train_data, labeldist, img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train, 
                data_type="train")


    train_error = []
    train_disc_error = []
    train_metric = []
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

        model_loss, disc_loss, f1_Score = train_batch(model, discriminator, optimizer, dis_optimizer, sample) 

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
        log_scalar(summary_writer, "STEP Train Error", mean_error, training_step)

        step+=1
        training_step+=1

    # print("train_error",train_error)
    # print("shape",train_error.shape)

    trainn_error = [ x[0] for x in train_error ]
    train_disc_loss = [ x[1] for x in train_error ]
    train_fscore = [ x[2] for x in train_error ]

    return model, trainn_error, training_step, train_disc_loss, train_fscore
    

def validate_epoch(epoch, model, discriminator, val_data, data_iterator_nuts, step_log_interval=10):
    (img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train) = data_iterator_nuts

    val_data_iterator = get_data_iterator(val_data, "", img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train, 
                data_type="val")


    val_error = []
    val_metric = []
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
    train_csv_data_path= r'C:\Users\ASUS\Downloads\code1.1\train_data.csv'
    val_csv_data_path= r'C:\Users\ASUS\Downloads\code1.1\test_data.csv'

    # folders and their path to proccessed inputs
    data_root = r"C:\Users\ASUS\Desktop\7ala - AI\Seminar in AI\Data\Train2\pre_processed\\"
    processed_img_fol= os.path.join(data_root, "oct_imgs")
    processed_oct_mask_fol= os.path.join(data_root, "oct_masks")
    processed_roi_mask_fol= os.path.join(data_root, "roi_masks")

    # folder to save training weights
    weight_fol=r"C:\Users\ASUS\Downloads\code1.1\weights"

    # folder to save logs viewable by tensorboard
    log_fol=r"C:\Users\ASUS\Downloads\code1.1\logs"

    # if not present, keep it None
    pre_trained_model_path= None # r"C:\Users\ASUS\Downloads\retouch\outputs\final_gan_weights.h5"
    # **************************************************************************

    # prepare data for training and validation
    train_data = read_csv_data(train_csv_data_path)
    val_data = read_csv_data(val_csv_data_path)

    labeldist = train_data >> CountValues(1)

    patch_size = (PATCH_SIZE_H, PATCH_SIZE_W)

    data_iterator_nuts = get_iterator_nuts(processed_img_fol, processed_oct_mask_fol, \
                            processed_roi_mask_fol, patch_size, BATCH_SIZE,)


    # ---------------------------------------------------------------

    # create model (generator)
    model = create_unet_model(input_shape=(PATCH_SIZE_H, PATCH_SIZE_W, 3))
    # print(model.summary())

    # load pre-trained model, if any
    if pre_trained_model_path is not None:
        print("Loading weights : ", pre_trained_model_path)
        model.load_weights(pre_trained_model_path)


    optimizer=Adam(learning_rate=ADAM_LR, beta_1=ADAM_BETA_1)

    # ----------        ----------      ----------      ----------      ----------      ----------

    # create discriminator

    D_backbone_name="MobileNetV2"
    # D_backbone_name="VGG16"
    discriminator = create_discriminator(D_backbone_name)

    dis_optimizer = Adam(learning_rate=D_ADAM_LR, beta_1=D_ADAM_BETA_1)

    # discriminator.compile( optimizer=Adam(learning_rate=D_ADAM_LR, beta_1=D_ADAM_BETA_1), loss=d_loss, metrics=["accuracy"] )


    # ---------------------------------------------------------------

    # create folders to save resources
    create_folder_if_absent(weight_fol)
    log_dir=os.path.join(log_fol, str(int(time())))
    create_folder_if_absent(log_dir)

    summary_writer = tf.summary.create_file_writer(log_dir)

    best_error = float("inf")

training_step=0

epoch = EPOCH
error_hold = []
print("Starting Training:")

for epoch in range(EPOCH):

    # train model for single epoch
    model, train_epoch_error, training_step, train_disc_loss, train_fscore = train_epoch(epoch+1, model, discriminator, optimizer, dis_optimizer, train_data, labeldist, data_iterator_nuts,summary_writer,training_step, step_log_interval=10)

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

    print("Summary: Epoch : {},  Train error : {}, Discriminator error: {}, FScore_train: {}, Val error : {}, FScore_val: {} \n".format\
          (epoch+1, mean_train_error, mean_train_disc_loss, mean_train_fscore, mean_val_error, mean_val_fscore))
    # print("Summary: Epoch : {},  Train errror : {}, Val error : {} \n".format\
    #       (epoch+1, mean_train_error, mean_val_error))

    log_scalar(summary_writer, "EPOCH Train Error", mean_train_error, epoch+1)
    log_scalar(summary_writer, "EPOCH Val Error", mean_val_error, epoch+1)

    # if log value better, save the weight
    if mean_val_error<best_error:
        weight_path=weight_fol+"/best_weight.h5"
        model.save_weights(weight_path)
        # model.save(weight_path)
        best_error=mean_val_error

    # saving weights each epoch for longer epochs if user wants to
    weight_path=weight_fol+"/"+"model_ep_{}.h5".format(epoch+1)
    model.save_weights(weight_path)
    # model.save(weight_path)

    with open(r'C:\Users\halay\OneDrive\Desktop\project\training_logs.csv', mode='a') as file:
      writer = csv.writer(file)
      writer.writerow([epoch+1, mean_train_error, mean_train_disc_loss, mean_train_fscore, mean_val_error, mean_val_fscore])
