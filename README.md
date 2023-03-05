# retouch_practical_work
### Introduction
Optical coherence tomography (OCT) is a non-invasive imaging technology that allows for high-resolution visualization of biological tissues at the micrometer scale. OCT is a widely used imaging technique for the diagnosis and management of retinal diseases. However, manual segmentation of the retinal layers is a time-consuming and error-prone process and may vary between different clinicians. To overcome this limitation, automated segmentation algorithms have been developed to segment the retinal layers accurately and efficiently. The combination of machine learning and automated segmentation of retinal layers in retinal OCT has the potential to improve the diagnosis and management of retinal diseases, by providing accurate and reliable information about the structure and function of the retina.
This repository implements the method proposed by Tennakon et al. The method is an end-to-end trained deep learning based retinal fluid segmentation method that works across 3D-OCT images that are acquired using devices from three different vendors: Cirrus, Spectralis and Topcon. 
### Implementation
#### Prepare data 
This script is run first on the training dataset provided. This script contains three functions: preprocess_oct_images, create_test_train_set, and create_roi_masks. The function preprocess_oct_images loads OCT (optical coherence tomography) images and corresponding segmentation masks from a directory, preprocesses them by histogram matching and applying a median filter, and saves them to a new directory. The function also creates a CSV file that stores the image name, vendor, root, slice, and the presence or absence of intraretinal fluid (IRF), subretinal fluid (SRF), and pigment epithelial detachment (PED) in each slice.

The function create_test_train_set splits the preprocessed OCT images into training and testing sets, and saves the corresponding root paths of the directories containing the training and testing data to CSV files. The function create_roi_masks loads the preprocessed OCT images and segmentation masks, applies a morphological closing operation and entropy thresholding to the segmentation masks, and saves the resulting binary masks to a new directory.

The preprocessed data can be found [here](https://www.google.com/).

#### Config
Script with all hyperparameters needed for training and preprocessing. The fluids IRF, SRF and PED are coded as 1, 2 and 3, respectively. The total number of classes, including the background, is 4 and the patches produced are of a size (256,128). 

#### Utils
##### Base, losses.py, metrics.py
The metrics.py script in the segmentation_models library implements various evaluation metrics commonly used in image segmentation tasks. The metrics include Intersection over Union (IoU), Dice coefficient, Precision, Recall, F1 score, and Average Precision. These metrics can be used to measure the accuracy and performance of segmentation models. The metrics chosen in this script is the IOU Score and FScore (Dice coefficient). The IoU and Dice coefficient measure the overlap between the predicted and ground truth segmentation masks. Additionally, the metrics support multiple classes and can be used for both binary and multi-class segmentation tasks.

##### custom_nuts.py
The script consists of two functions: "sample_oct_patch_centers" and "sample_patches_entropy_mask". "sample_oct_patch_centers" takes as input a binary image roimask, a patch shape pshape, the number of positive samples to extract npos, and values to indicate a positive and a negative label, pos and neg respectively. It returns a list of coordinates for npos positive patches that are sampled randomly from roimask.

"sample_patches_entropy_mask" takes as input an OCT image, a corresponding segmentation ground truth mask, an OCT region of interest mask, a patch shape, the number of positive and negative patches to extract, and values to indicate a positive and a negative label. It yields patches of the OCT image, the corresponding mask patch, and labels indicating the presence or absence of IRF, SRF, and PED within the patch.

Both functions are used in a pipeline for generating image patches where patches are extracted from the OCT image according to a binary mask. The script defines the "ImagePatchesByMaskRetouch_resampled" function, which is a Nut processor (using nutsml library) that applies "sample_patches_retouch_mask" to an iterable of images, masks, and labels to yield image patches, mask patches, and labels for IRF, SRF, and PED.
