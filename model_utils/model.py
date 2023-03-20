import tensorflow as tf

from keras.layers import Conv2D, Input, UpSampling2D
from keras.layers import GlobalMaxPooling2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, SpatialDropout2D, Conv2DTranspose

from keras.layers import BatchNormalization, LeakyReLU, Cropping2D, Dense
from keras.layers import concatenate, Activation

from keras.models import  Model, Sequential 

from keras.utils import plot_model
from keras.optimizers import Adam

from keras import backend as K
from keras.regularizers import l2
from keras.activations import softmax

from model_utils.model_layers import Softmax4D

N_CLASSES = 4

def create_encoder_layer(layer_input, filters, regularizer, block_name, activation="relu", use_batch_normalization=True, 
                            data_format='channels_last' ,first_layer=False):

    if first_layer:
        name=block_name+"_conv_0"
        x = Conv2D(filters, (7, 7), activation=activation, name=name, padding='same', data_format=data_format,
                        kernel_regularizer=regularizer)(layer_input)        
    else:
        name=block_name+"_pool_1"
        x = MaxPooling2D(pool_size=(2, 2), name=name, data_format=data_format)(layer_input)


    name=block_name+"_conv_1"
    x = Conv2D(filters, (3, 3), activation='relu', name=name, data_format=data_format,
                     kernel_regularizer=regularizer)(x)


    name=block_name+"_conv_2"
    x = Conv2D(filters, (3, 3), name=name, data_format=data_format, kernel_regularizer=regularizer)(x)

    if use_batch_normalization:
        name=block_name+"_batch_norm_1"
        x = BatchNormalization(axis=-1, name=name)(x)

    x = Activation(activation)(x)

    return x



def create_decoder_layer(layer_input, encoder_layer_input, filters, regularizer, block_name, spatial_dropout, cropping, 
                        activation="relu",  use_batch_normalization=True, data_format='channels_last', 
                        final_spatial_dropout=None):


    x1 = Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same', data_format=data_format, name=block_name+'_upool1')(layer_input)
    x1 = SpatialDropout2D(spatial_dropout, data_format=data_format)(x1)

    x2 =  Cropping2D(cropping=cropping, data_format=data_format)(encoder_layer_input)
    x2 = SpatialDropout2D(spatial_dropout, data_format=data_format)(x2)

    name = block_name + "_merge_1"
    x = concatenate([x1,x2], axis=-1, name=name)

    name = block_name + "_conv_1"
    x = Conv2D(filters, (3, 3), activation=activation, name=name, data_format=data_format,
                      kernel_regularizer=regularizer)(x)

    name = block_name + "_conv_2"
    x = Conv2D(filters, (3, 3), name=name, data_format=data_format,
                      kernel_regularizer=regularizer)(x)

    if use_batch_normalization:
        name = block_name + "_batch_norm_1"
        x = BatchNormalization(axis=-1, name=name)(x)

    x = Activation(activation)(x)

    if final_spatial_dropout:
        x = SpatialDropout2D(final_spatial_dropout, data_format=data_format)(x)

    return x



def create_unet_model(input_shape=(224, 224, 3), regularizer_type=l2 , regularizer_weight=0.0001, activation="relu", 
                data_format='channels_last', TRAIN_CLASSES=False ):

    if regularizer_type is not None:
        regularizer = regularizer_type(regularizer_weight)
    else:
        regularizer = None    
        
    layer_input = Input(shape=input_shape)

    block_name = "encoder_block_1"
    x1 = create_encoder_layer(layer_input, filters=64, regularizer=regularizer, block_name=block_name, activation=activation, first_layer=True, data_format=data_format)
    block_name = "encoder_block_2"
    x2 = create_encoder_layer(x1, filters=128, regularizer=regularizer, block_name=block_name, activation=activation, data_format=data_format )
    block_name = "encoder_block_3"
    x3 = create_encoder_layer(x2, filters=256, regularizer=regularizer, block_name=block_name, activation=activation, data_format=data_format )
    block_name = "encoder_block_4"
    x4 = create_encoder_layer(x3, filters=512, regularizer=regularizer, block_name=block_name, activation=activation, data_format=data_format )

    block_name = "decoder_block_1"
    cropping=((5, 4), (5, 4))
    y1 = create_decoder_layer(layer_input=x4, encoder_layer_input=x3, filters=256, regularizer=regularizer, block_name=block_name, 
            spatial_dropout=0.25, cropping=cropping, activation=activation, data_format=data_format )
            
    block_name = "decoder_block_2"
    cropping=((17, 17), (17, 17))
    y2 = create_decoder_layer(layer_input=y1, encoder_layer_input=x2, filters=128, regularizer=regularizer, block_name=block_name, 
            spatial_dropout=0.25, cropping=cropping, activation=activation, data_format=data_format )   

    block_name = "decoder_block_3"
    cropping=(42, 42), (42, 42)
    y3 = create_decoder_layer(layer_input=y2, encoder_layer_input=x1, filters=64, regularizer=regularizer, block_name=block_name, 
            spatial_dropout=0.5, cropping=cropping, activation=activation, use_batch_normalization=False, final_spatial_dropout=0.25, data_format=data_format)   


    # multiscale seg out
    b1_up = UpSampling2D(size=(8, 8), data_format='channels_last')(x4)
    b1_up = Cropping2D(cropping=((14, 14), (14, 14)), data_format=data_format)(b1_up)
    b2_up = UpSampling2D(size=(4, 4), data_format='channels_last')(y1)
    b2_up = Cropping2D(cropping=((6, 6), (6, 6)), data_format=data_format)(b2_up)
    b3_up = UpSampling2D(size=(2, 2), data_format='channels_last')(y2)
    b3_up = Cropping2D(cropping=((2, 2), (2, 2)), data_format=data_format)(b3_up)

    block_name = "decoder_block_4"
    c1 = concatenate([b1_up, b2_up, b3_up, y3], axis=-1, name=block_name+'_merge_1')

    c1 = Conv2D(4, (3, 3), activation=activation, name='seg_out_', data_format=data_format, kernel_regularizer=regularizer, padding='same')(c1)    
    c1 = Softmax4D(axis=-1, name=block_name+'_seg_out')(c1)



    if not TRAIN_CLASSES:
        model = Model(inputs=layer_input, outputs=c1)
    else:
        # get  IRF, SRF, PED
        c_out_IRF = Conv2D(1, (3, 3), activation='sigmoid', name=block_name+'_c_out_', data_format=data_format,
                           kernel_regularizer=regularizer)(x4)
        c_out_IRF = GlobalMaxPooling2D(data_format='channels_last', name=block_name+'_gpool')(c_out_IRF)

        c_out_SRF = Conv2D(2, (3, 3), activation='sigmoid', name=block_name+'_c_out_', data_format=data_format,
                           kernel_regularizer=regularizer)(x4)
        c_out_SRF = GlobalAveragePooling2D(data_format='channels_last', name=block_name+'_gpool')(c_out_SRF)

        c_out_PED = Conv2D(2, (3, 3), activation='sigmoid', name=block_name+'_c_out_', data_format=data_format,
                           kernel_regularizer=regularizer)(x4)
        c_out_PED = GlobalAveragePooling2D(data_format==data_format, name=block_name+'_gpool')(c_out_PED)

        model = Model(inputs=layer_input, outputs=[c1, c_out_IRF, c_out_SRF, c_out_PED])

    return model


def create_discriminator(backbone_name):
    from keras.applications import MobileNetV2, VGG16

    if backbone_name=="MobileNetV2":
        bb = MobileNetV2(weights=None, include_top=False, input_shape=(None,None, N_CLASSES))
    elif backbone_name=="VGG16":
        bb = VGG16(weights=None, include_top=False, input_shape=(None,None,N_CLASSES))

    x = GlobalAveragePooling2D()(bb.output)
    x = Dense(256)(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation="sigmoid")(x)

    discriminator = Model(bb.input, x)

    return discriminator