import os, cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFilter

import tensorflow as tf

from model_utils.model import create_unet_model
from utils.utils import mhd
from utils.utils.slice_op import hist_match 

import config

# ********************************************************************************

OS_WINDOWS = os.name=="nt"     # operating system is windows

N_CLASSES = config.N_CLASSES

def get_images_from_OCT_scan(oct_scan_path, oct_r):
    image_name = oct_scan_path.split('/')[-2]
    vendor = oct_scan_path.split('/')[-3]
    img, _, _ = mhd.load_oct_image(oct_scan_path)

    if 'Cirrus' in vendor:
        img = hist_match(img, oct_r)
    elif 'Topcon' in vendor:
        img = hist_match(img, oct_r)
    num_slices = img.shape[0]

    slice_array=[]
    details_array=[]

    for slice_num in range(0, num_slices):
        if 'Cirrus' in vendor and (slice_num > 0) and (slice_num < num_slices - 1):
            im_slice = np.median(img[slice_num - 1:slice_num + 2, :, :].astype(np.int32), axis=0).astype(
                np.int32)
        if 'Topcon' in vendor and (slice_num > 0) and (slice_num < num_slices - 1):
            im_slice = np.median(img[slice_num - 1:slice_num + 2, :, :].astype(np.int32), axis=0).astype(
                np.int32)
        else:
            im_slice = img[slice_num, :, :].astype(np.int32)
        im_slice = Image.fromarray(im_slice, mode='I')
        
        if 'Cirrus' in vendor:
            im_slice = im_slice.filter(ImageFilter.MedianFilter(size=3))
        elif 'Topcon' in vendor:
            im_slice = im_slice.filter(ImageFilter.MedianFilter(size=3))

        im_slice=np.array(im_slice)
        slice_array.append(im_slice)
        details_array.append((vendor, image_name, slice_num))

    slice_array = np.array(slice_array)

    # yield (im_slice, (vendor, image_name, slice_num))
    return (slice_array, details_array)


def get_stacked_OCT_scan_slices(slice_array, details_array):
    N=slice_array.shape[0]

    for i in range(N):
        s2=slice_array[i]
        if i==0:
            s1=slice_array[0]
        else:
            s1=slice_array[i-1]

        if i==N-1:
            s3=slice_array[i]
        else:
            s3=slice_array[i+1]

        detail=details_array[i]

        scan_stack=np.array([s1,s2,s3])

        yield (scan_stack, detail)


def get_saving_name(details):
    vendor, image_name, slice_num = details
    name = vendor + '_' + image_name + '_' + str(slice_num).zfill(3)
    return name


def pre_process_image(image, parameters=(config.SLICE_MEAN , config.SLICE_SD)):   
    PRE_PROCESS_MEAN = parameters[0]
    PRE_PROCESS_STD = parameters[1]

    image = image - PRE_PROCESS_MEAN
    image = image / PRE_PROCESS_STD

    return image


def process_seg(seg, threshold=0.5, N_CLASSES=N_CLASSES):  

  cls=np.argmax(seg, axis=-1)
  conf=np.max(seg, axis=-1)

  mask=np.zeros(seg.shape[:2])  

  for i in range(1, N_CLASSES):
    mask[cls==1]=i

  mask[cls<threshold]=0
  return mask


def predict_crop_mask(crop_image, model, crop_W, crop_H, threshold=0.5):
    img = pre_process_image(crop_image)
    img = np.expand_dims(img, axis=0)

    seg = model.predict(img)[0]
    mask = process_seg(seg, threshold, N_CLASSES)

    mask2=np.expand_dims(mask, axis=-1)
    mask2=tf.image.resize_with_crop_or_pad(mask2, crop_H, crop_W).numpy()
    mask2=np.squeeze(mask2)

    return mask2


def predict_mask(image, model, CROP_H, CROP_W, threshold=0.5):
    h,w=image.shape[:2]

    mask=np.zeros((h,w))

    for i in range(0, h, CROP_H):
        for j in range(0, w, CROP_W):

            h2=np.min([i+CROP_H, h])
            w2=np.min([j+CROP_W, w])
            crop_img=image[i:h2, j:w2, :]

            if crop_img.max()==0:
                continue

            pad_h, pad_v=0,0
            if crop_img.shape[0]<CROP_H:
                pad_v=CROP_H - crop_img.shape[0]
            if  crop_img.shape[1]>CROP_W:
                pad_h=CROP_W - crop_img.shape[1]

            
            crop_img_pad=crop_img        
            if pad_v:
                crop_img_pad=np.concatenate( [crop_img_pad, np.zeros( (pad_v, crop_img_pad.shape[1], 3)) ], axis=0)
            if pad_h:
                crop_img_pad=np.concatenate( [ crop_img_pad, np.zeros( (crop_img_pad.shape[0], pad_h, 3)),  ], axis=1)

            crop_mask = predict_crop_mask(crop_img_pad, model, config.PATCH_SIZE_W, config.PATCH_SIZE_H, threshold)

            if pad_h!=0:
                crop_mask=crop_mask[:,:-pad_h]
            if pad_v!=0:
                crop_mask=crop_mask[:-pad_v, :]

            mask[i:h2, j:w2] = crop_mask

    return mask


def process_stacked_slices(stacked_slice_iterator, model, H, W, mask_fol, OVERLAY, overlay_fol):

    for tup in stacked_slice_iterator:
        stack_img, details = tup
        save_name = get_saving_name(details)
        img=np.moveaxis(stack_img, 0, -1)

        mask=predict_mask(img, model, H, W, 0.9)

        # save mask
        mpath = os.path.join(mask_fol, save_name+".png")
        cv2.imwrite(mpath, mask)

        # save overlay
        if OVERLAY:
            overlay= create_overlay(img, mask, N_CLASSES)
            opath = os.path.join(overlay_fol, save_name+".png")
            im_pil = Image.fromarray(overlay)
            im_pil.save(opath)


def create_overlay(img, seg, n_classes=config.N_CLASSES):
  im2=np.zeros(img.shape, dtype=np.uint8)

  COL_LL=[ (0,0,0), (0,0,255), (0,255,0), (255,0,0), (255,0,255), (0,255,255), (255,255,0) ]
  for i in range(1,n_classes):
    im2[ seg==i ]=COL_LL[i]

  overlay=im2*0.5 + img*0.5
  overlay=overlay.astype(np.uint8)
  return overlay  




reference_im_path = os.path.join(r"C:\Users\ASUS\Desktop\7ala - AI\Seminar in AI\Data\Train2\Spectralis", "TRAIN028", "oct.mhd")
OCT_reference, _, _ = mhd.load_oct_image(reference_im_path)

threshold = 0.5
OVERLAY=True

if __name__=="__main__":
    # ********************************************************************************

    # script handles test images without preprocessing, need to run it separately for every vendor folder

    # input_path = r"C:\Users\ASUS\Downloads\code1.1\Test\Topcon"
    # input_path = r"C:\Users\ASUS\Downloads\code1.1\Test\Cirrus"
    input_path = r"C:\Users\ASUS\Downloads\code1.1\Test\Spectralis"

    output_fol = r"C:\Users\ASUS\Downloads\code1.1\output"
    weight_path = r"C:\Users\ASUS\Downloads\retouch\outputs\final_gan_weights.h5"

    # ********************************************************************************

    H, W = config.PATCH_SIZE_H, config.PATCH_SIZE_W

    model=create_unet_model(input_shape=(H, W, 3))
    model.load_weights(weight_path)

    print("Model Loaded : {}".format(weight_path))

    print("Processing input data :->")

    if os.path.isdir(input_path):
        dir_list = os.listdir(input_path)

        for fol in tqdm(dir_list):
            if os.path.isfile( os.path.join(input_path, fol)):
                continue

            print(fol)
            mask_dest_fol=os.path.join(output_fol, fol, "mask")
            if not os.path.exists(mask_dest_fol):
                os.makedirs(mask_dest_fol)

            if OVERLAY:
                overlay_dest_fol=os.path.join(output_fol, fol, "overlay")
                if not os.path.exists(overlay_dest_fol):
                    os.makedirs(overlay_dest_fol)
            else:
                overlay_dest_fol=""

            oct_path=os.path.join(input_path, fol, "oct.mhd")
            if OS_WINDOWS:
                oct_path=oct_path.replace("\\", "/")

            # actual inference starts now
            slice_array, details_array = get_images_from_OCT_scan(oct_path, OCT_reference)
            slice_stack_itr=get_stacked_OCT_scan_slices(slice_array, details_array)
            process_stacked_slices( slice_stack_itr, model, H, W, mask_dest_fol, OVERLAY, overlay_dest_fol)


