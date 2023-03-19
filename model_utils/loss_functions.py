# import tensorflow.keras.backend as KB
from keras import backend as KB
import tensorflow as tf



def get_dice_loss(y_true, y_pred):
    # dice loss function

    g_0 = y_true[:, :, :, 0]
    p_0 = y_pred[:, :, :, 0]

    true_pos = KB.sum((1. - p_0) * (1. - g_0), keepdims=False)
    false_pos = KB.sum((1. - p_0) * g_0, keepdims=False)
    false_neg = KB.sum(p_0 * (1. - g_0), keepdims=False)
    dice_loss = (2. * true_pos) / (2. * true_pos + false_pos + false_neg + KB.epsilon())

    return dice_loss


def get_balanced_cross_entropy_loss_function(BATCH_SIZE, num_classes):
    """""
    balanced cross entropy loss function
    this function returns a function which then computes loss
    """"" 

    def get_balanced_cross_entropy_loss(y_true, y_pred):
        """""
        balanced cross entropy loss which takes properly formatted inputs
        to compute the loss value
        """""
        cross_ent = (KB.log(y_pred) * y_true)
        cross_ent = KB.sum(cross_ent, axis=-2, keepdims=False)
        cross_ent = KB.sum(cross_ent, axis=-2, keepdims=False)
        cross_ent = KB.reshape(cross_ent, shape=(BATCH_SIZE, num_classes)) #num_classes

        y_true_ = KB.sum(y_true, axis=-2, keepdims=False)
        y_true_ = KB.sum(y_true_, axis=-2, keepdims=False)
        y_true_ = KB.reshape(y_true_, shape=(BATCH_SIZE, num_classes)) + KB.epsilon() #num_classes

        cross_ent = (cross_ent / y_true_)

        loss_value = -1*KB.mean(cross_ent, axis=-1, keepdims=False)
        loss_value = KB.mean(loss_value)

        return loss_value

    return get_balanced_cross_entropy_loss


def process_loss_fn_inputs(y_true, y_pred):
    """""
    both y_true and y_pred are of different shapes appropriate functions must be applied to make them
    compatible for the final loss function
    """""
    CROP_SHAPE = KB.int_shape(y_pred)

    y_true_ = tf.compat.v1.image.resize_image_with_crop_or_pad(y_true, target_height=CROP_SHAPE[1], target_width=CROP_SHAPE[2])
    y_true_ = tf.cast(y_true_, tf.float32)

    y_pred_ = KB.clip(y_pred, KB.epsilon(), 1. - KB.epsilon())

    return (y_true_, y_pred_)


def get_combined_cross_entropy_and_dice_loss_function(BATCH_SIZE, num_classes):
    """""
    function that returns a loss function    
    loss function is the combined loss of cross entropy and dice loss
    """""

    cross_entropy_loss = get_balanced_cross_entropy_loss_function(BATCH_SIZE, num_classes)

    def get_balanced_cross_entropy_and_dice_loss(y_true, y_pred):

        y_true_, y_pred_ = process_loss_fn_inputs(y_true, y_pred)

        CE_loss_value = cross_entropy_loss(y_true_, y_pred_)
        dice_loss_value = get_dice_loss(y_true_, y_pred_)

        final_loss = 0.5*(CE_loss_value + dice_loss_value)

        return final_loss


    return get_balanced_cross_entropy_and_dice_loss


def generator_loss(predicted_mask_output):
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    loss_val = bce_loss(tf.ones_like(predicted_mask_output), predicted_mask_output)

    return loss_val



def discriminator_loss(label_mask_output, predicted_mask_output):
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    l1 = bce_loss(tf.ones_like(label_mask_output), label_mask_output)
    l2 = bce_loss(tf.zeros_like(predicted_mask_output), predicted_mask_output)

    total_loss = l1 + l2

    return total_loss