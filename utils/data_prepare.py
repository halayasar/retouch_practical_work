import os
import numpy as np
import skimage.transform as skt

from nutsflow import *
from nutsml import *

from utils.custom_nuts import ReadOCT, ImagePatchesByMaskRetouch_resampled

from config import SLICE_MEAN, SLICE_SD


def rearange_cols(sample):
    """
    Re-arrange the incoming data stream to desired outputs
    :param sample: (image_name, vendor, root, slice, is_IRF, is_SRF, is_PED)
    :return: (image_name, GT_mask_name, is_IRF, is_SRF, is_PED, ROI_mask_name)
    """
    img = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
    mask = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
    IRF_label = sample[4]
    SRF_label = sample[5]
    PED_label = sample[6]
    roi_m = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'

    return (img, mask, IRF_label, SRF_label, PED_label, roi_m)

# training image augementation (flip-lr and rotate)
# normal rotate would interpolate pixel values
def myrotate(image, angle):
    return skt.rotate(image, angle, preserve_range=True, order=0).astype('uint8')

# Filter to drop some non-pathology patches
def drop_patch(sample, drop_prob=0.75):
    """
    Randomly drop a patch from iterator if there is no pathology
    :param sample: 
    :param drop_prob: 
    :return: 
    """
    if (int(sample[2]) == 0) and (int(sample[3]) == 0) and (int(sample[4]) == 0):
        return float(np.random.random_sample()) < drop_prob
    else:
        return False    


image_preprocess = lambda s: (s.astype(np.float32) - SLICE_MEAN) / SLICE_SD


def read_csv_data(csv_data_path):

    csv_data = ReadPandas(csv_data_path, dropnan=True)
    data = csv_data >> Collect()   
    return data


def get_iterator_nuts(processed_img_fol, processed_oct_mask_fol, processed_roi_mask_fol, patch_size, BATCH_SIZE, ):

    (PATCH_SIZE_H, PATCH_SIZE_W) = patch_size
 
    TransformImage.register('myrotate', myrotate)

    # train data augmentation
    augment_1 = (AugmentImage((0, 1, 5))
                 .by('identical', 1.0)
                 .by('fliplr', 0.5))

    augment_2 = (AugmentImage((0, 1, 5))
                 .by('identical', 1.0)
                 .by('myrotate', 0.5, [0, 10]))    

    # data locations
    processed_img_fol = processed_img_fol.rstrip("/").rstrip("\\")
    processed_oct_mask_fol = processed_oct_mask_fol.rstrip("/").rstrip("\\")
    processed_roi_mask_fol = processed_roi_mask_fol.rstrip("/").rstrip("\\")

    # data files 
    image_fol = processed_img_fol + '/*'
    mask_fol = processed_oct_mask_fol + '/*'
    roi_fol = processed_roi_mask_fol + '/*'

    # data readers
    img_reader = ReadOCT(0, image_fol)
    mask_reader = ReadImage(1, mask_fol)
    roi_reader = ReadImage(5, roi_fol)

    # randomly sample image patches from the interesting region (based on entropy)
    image_patcher = ImagePatchesByMaskRetouch_resampled(imagecol=0, maskcol=1, IRFcol=2, SRFcol=3, PEDcol=4, roicol=5,
                                                        pshape=(PATCH_SIZE_H, PATCH_SIZE_W),
                                                        npos=12, nneg=2, pos=1, use_entropy=True, patch_border=42)


    build_batch_train = (BuildBatch(BATCH_SIZE, prefetch=0)
                         .input(0, 'image', 'float32', channelfirst=False)
                         .output(1, 'one_hot', 'uint8', 4)
                         .output(2, 'number', 'uint8')
                         .output(3, 'number', 'uint8')
                         .output(4, 'number', 'uint8'))                  
              

    return (img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train)


def get_data_iterator(data, labeldist, img_reader, mask_reader, roi_reader, augment_1, augment_2, image_patcher,  build_batch_train, 
                data_type="train"):
    # data loader 
    # If data_type=="train": data is stratified according to the label distribution, 
    # shuffled, and various augmentations are applied. The image_patcher function is 
    # used to randomly crop a patch from each image, which is then preprocessed using 
    # image_preprocess and filtered based on whether it contains any invalid pixels 
    # (drop_patch). 
    # build_batch_train function is used to construct batches of data for training

    if data_type=="train":
        data_iterator= data >> Stratify(1, labeldist) >> Shuffle(1000) >> Map(
                rearange_cols) >> img_reader >> mask_reader >> roi_reader >> augment_1 >> augment_2 >> Shuffle(
                100) >> image_patcher >> MapCol(0, image_preprocess) >> Shuffle(1000) >> FilterFalse(
                drop_patch) >> build_batch_train    
    else:
        data_iterator = data >> Map(rearange_cols) >> Shuffle(
            1000) >> img_reader >> mask_reader >> roi_reader >> image_patcher >> Shuffle(1000) >> MapCol(0,
            image_preprocess) >> FilterFalse(drop_patch) >> build_batch_train 

    return data_iterator
