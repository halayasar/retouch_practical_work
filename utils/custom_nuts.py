import numpy as np
import os.path
import os.path as path
import random as rnd

import nutsml.datautil as ut
import nutsml.imageutil as ni
from nutsflow import Nut, NutFunction, nut_processor, nut_function
from nutsflow.common import as_tuple, as_set

import skimage
import skimage.transform
import skimage.io as sio
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk, rectangle, closing, opening, binary_closing, convex_hull_image

import scipy
from PIL import Image
import matplotlib.patches
import matplotlib.pyplot as plt


from config import *



SHAPE_MULT = {1024: 2., 496: 1., 650: 0.004 / 0.0035, 885: 0.004 / 0.0026}



def sample_oct_patch_centers(roimask, pshape, npos, pos=1, neg=0):
    PYINX = 0
    PXINX = 1
    h, w = roimask.shape

    x = range(pshape[PXINX] // 2, w - pshape[PXINX] // 2, 10)
    x_samples = np.random.choice(x, npos * 2, replace=False)
    np.random.shuffle(x_samples)
    c_hold = list()
    for x in x_samples:
        nz = np.nonzero(roimask[:, x] == pos)[0]
        if len(nz) > 0:
            y = int(float(np.min(nz)) + (
                (float(np.max(nz)) - float(np.min(nz))) / 2.) + np.random.uniform()) + np.random.randint(-10, 10)
            if (y - pshape[PYINX] / 2) < 1:
                y = int(pshape[PYINX] / 2) + 1
            elif (y + pshape[PYINX] / 2) >= h:
                y = h - int(pshape[PYINX] / 2) - 1

            c_hold.append([y, x, 1])
        if len(c_hold) >= npos:
            break

    return c_hold


def sample_patches_entropy_mask(img, mask=None, roimask=None, pshape=(224, 224), npos=10, nneg=1, pos=255, neg=0,
                                patch_border=12):
    """
    Generate patches from the interesting region of the OCT slice
    :param img: oct image slice
    :param mask: oct segmentation GT
    :param roimask: oct ROI
    :param pshape: patch shape
    :param npos: Number of patches to sample from interesting region
    :param nneg: Number of patches to sample from non interesting region
    :param pos: Mask value indicating positives
    :param neg: Mask value indicating negative
    :param patch_border: boder to ignore when creating IRF,SRF,PED labels for patches (ignore border pixels for predicting labels)
    :return: 
    """

    roimask = roimask.astype(np.int8)

    it = sample_oct_patch_centers(roimask, pshape=pshape, npos=npos, pos=pos, neg=neg)

    it1 = ni.sample_patch_centers(roimask, pshape=pshape, npos=int(float(npos)*.2), nneg=0, pos=pos, neg=neg)
    for r, c, l in it1:
        it.append([r, c, l])

    for r, c, label in it:
        img_patch = ni.extract_patch(img, pshape, r, c)
        mask_patch = ni.extract_patch(mask, pshape, r, c)
        label_IRF = np.int8(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == IRF_CODE))
        label_SRF = np.int8(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == SRF_CODE))
        label_PED = np.int8(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == PED_CODE))
        yield img_patch, mask_patch, label_IRF, label_SRF, label_PED


def sample_patches_retouch_mask(img, mask=None, pshape=(224, 224), npos=10, nneg=1, pos=255, neg=0, patch_border=12):
    """
    Generate patches from the interesting region of the OCT slice
    :param img: oct image slice
    :param mask: oct segmentation GT
    :param pshape: patch shape
    :param npos: Number of patches to sample from interesting region
    :param nneg: Number of patches to sample from non interesting region
    :param pos: Mask value indicating positives
    :param neg: Mask value indicating negative
    :param patch_border: boder to ignore when creating IRF,SRF,PED labels for patches (ignore border pixels for predicting labels)
    :return: 
    """
    assert mask.dtype == np.uint8
    roi_mask = np.logical_not(mask == 0)

    it = ni.sample_patch_centers(roi_mask, pshape=pshape, npos=npos, nneg=nneg, pos=pos, neg=neg)
    for r, c, label in it:
        img_patch = ni.extract_patch(img, pshape, r, c)
        mask_patch = ni.extract_patch(mask, pshape, r, c)
        label_IRF = np.int8(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == IRF_CODE))
        label_SRF = np.int8(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == SRF_CODE))
        label_PED = np.int8(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == PED_CODE))
        yield img_patch, mask_patch, label_IRF, label_SRF, label_PED
        

@nut_processor
def ImagePatchesByMaskRetouch_resampled(iterable, imagecol, maskcol, IRFcol, SRFcol, PEDcol, roicol, pshape, npos,
                                        nneg, pos=255, neg=0, patch_border=12, use_entropy=False):
    """
    :param iterable: iterable: Samples with images
    :param imagecol: Index of sample column that contain image
    :param maskcol: Index of sample column that contain mask
    :param IRFcol: Index of sample column that contain IRF label
    :param SRFcol: Index of sample column that contain SRF label
    :param PEDcol: Index of sample column that contain PED label
    :param pshape: Shape of patch
    :param npos: Number of patches to sample from interesting region
    :param nneg: Number of patches to sample from outside interesting region
    :param pos: Mask value indicating positives
    :param neg: Mask value indicating negativr
    :param patch_border: boder to ignore when creating IRF,SRF,PED labels for patches (ignore border pixels for predicting labels)
    :return: Iterator over samples where images and masks are replaced by image and mask patches
        and labels are replaced by labels [0,1] for patches
    """

    for sample in iterable:
        image, mask, roim = sample[imagecol], sample[maskcol], sample[roicol]
        img_height = image.shape[0]

        if image.shape[:2] != mask.shape:
            raise ValueError('Image and mask size don''t match!')

        assert img_height in {1024, 496, 650, 885}
        npshape = (int(pshape[0] * SHAPE_MULT[img_height]), pshape[1])
        if use_entropy:
            it = sample_patches_entropy_mask(image, mask, roimask=roim, pshape=npshape, npos=npos, nneg=nneg, pos=pos,
                                             neg=neg, patch_border=patch_border)

        else:
            it = sample_patches_retouch_mask(image, mask, pshape=npshape, npos=npos, nneg=nneg,
                                             pos=pos, neg=neg, patch_border=patch_border)

        for img_patch, mask_patch, label_IRF, label_SRF, label_PED in it:
            outsample = list(sample)[:]

            if img_height == 496:
                outsample[imagecol] = img_patch
                outsample[maskcol] = mask_patch
            else:
                outsample[imagecol] = skimage.transform.resize(img_patch.astype(np.uint8), pshape, order=0, preserve_range=True).astype('uint8')
                
                outsample[maskcol] = skimage.transform.resize(mask_patch.astype(np.uint8), pshape, order=0, preserve_range=True).astype('uint8')

            outsample[IRFcol] = np.int8(np.any(outsample[maskcol][patch_border:-patch_border, patch_border:-patch_border] == IRF_CODE))
            outsample[SRFcol] = np.int8(np.any(outsample[maskcol][patch_border:-patch_border, patch_border:-patch_border] == SRF_CODE))
            outsample[PEDcol] = np.int8(np.any(outsample[maskcol][patch_border:-patch_border, patch_border:-patch_border] == PED_CODE))

            yield tuple(outsample)


def load_oct_image(filepath, as_grey=False, dtype='uint8', no_alpha=True):
    """
    Load three consecative oct slices given the filepath.

    """
    if filepath.endswith('.npy'):  # image as numpy array
        print( "reading numpy OCT not yet implemented...")

    else:

        arr1 = np.expand_dims(sio.imread(filepath, as_gray=as_grey, img_num=0).astype(dtype), axis=-1)
        slice_num = int(filepath[:-5].split('_')[-1])
        if slice_num == 0:
            arr0 = arr1
        else:
            s0_filepath = filepath[:-8] + str(slice_num - 1).zfill(3) + '.tiff'
            arr0 = np.expand_dims(sio.imread(s0_filepath, as_gray=as_grey, img_num=0).astype(dtype), axis=-1)

        s2_filepath = filepath[:-8] + str(slice_num + 1).zfill(3) + '.tiff'
        if os.path.isfile(s2_filepath):
            arr2 = np.expand_dims(sio.imread(s2_filepath, as_gray=as_grey, img_num=0).astype(dtype), axis=-1)

        else:
            arr2 = arr1

        arr = np.concatenate([arr0, arr1, arr2], axis=-1)

    if arr.ndim == 3 and arr.shape[2] == 4 and no_alpha:
        arr = arr[..., :3]  # cut off alpha channel
    return arr


@nut_function
def ReadOCT(sample, columns, pathfunc=None, as_grey=False, dtype='uint8'):
    """
    Load OCT images for samples.

    Loads 3 consecative oct slices tif format.
    Images are returned as numpy arrays of shape (h, w, 3) for
    gray scale images.
    """

    def load(fileid):
        """Load image for given fileid"""
        if isinstance(pathfunc, str):
            filepath = pathfunc.replace('*', fileid)
        elif hasattr(pathfunc, '__call__'):
            filepath = pathfunc(sample)
        else:
            filepath = fileid
        return load_oct_image(filepath, as_grey=as_grey, dtype=dtype)

    if columns is None:
        return (load(sample),)  # image as tuple with one element

    colset = as_set(columns)
    elems = enumerate(sample)
    return tuple(load(e) if i in colset else e for i, e in elems)
