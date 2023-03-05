# retouch_practical_work
### Introduction
Optical coherence tomography (OCT) is a non-invasive imaging technology that allows for high-resolution visualization of biological tissues at the micrometer scale. OCT is a widely used imaging technique for the diagnosis and management of retinal diseases. However, manual segmentation of the retinal layers is a time-consuming and error-prone process and may vary between different clinicians. To overcome this limitation, automated segmentation algorithms have been developed to segment the retinal layers accurately and efficiently. The combination of machine learning and automated segmentation of retinal layers in retinal OCT has the potential to improve the diagnosis and management of retinal diseases, by providing accurate and reliable information about the structure and function of the retina.
This repository implements the method proposed by Tennakon et al. The method is an end-to-end trained deep learning based retinal fluid segmentation method that works across 3D-OCT images that are acquired using devices from three different vendors: Cirrus, Spectralis and Topcon. 
### Implementation
### Prepare data 
This script is run first on the training dataset provided. This script contains three functions: preprocess_oct_images, create_test_train_set, and create_roi_masks.

The function preprocess_oct_images loads OCT (optical coherence tomography) images and corresponding segmentation masks from a directory, preprocesses them by histogram matching and applying a median filter, and saves them to a new directory. The function also creates a CSV file that stores the image name, vendor, root, slice, and the presence or absence of intraretinal fluid (IRF), subretinal fluid (SRF), and pigment epithelial detachment (PED) in each slice.

The function create_test_train_set splits the preprocessed OCT images into training and testing sets, and saves the corresponding root paths of the directories containing the training and testing data to CSV files.

The function create_roi_masks loads the preprocessed OCT images and segmentation masks, applies a morphological closing operation and entropy thresholding to the segmentation masks, and saves the resulting binary masks to a new directory.
