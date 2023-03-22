# retouch_practical_work
### Introduction
Optical coherence tomography (OCT) is a non-invasive imaging technology that allows for high-resolution visualization of biological tissues at the micrometer scale. Manual segmentation of the retinal layers is a time-consuming and error-prone process and may vary between different clinicians. To overcome this limitation, automated segmentation algorithms have been developed to segment the retinal layers accurately and efficiently. This repository implements the method proposed by Tennakon et al. 

### Implementation
#### Prepare data 
This script is run first on the training dataset provided. This script contains three functions: preprocess_oct_images, create_test_train_set, and create_roi_masks. 
The preprocessed data can be found [here](https://drive.google.com/file/d/1zA6AP6OruucBSpQ2Aw7moJIlPpeBWQgE/view?usp=share_link).
 

##### custom_nuts.py
The script consists of two functions: "sample_oct_patch_centers" and "sample_patches_entropy_mask". "sample_oct_patch_centers" takes as input a binary image roimask, a patch shape pshape, the number of positive samples to extract npos, and values to indicate a positive and a negative label, pos and neg respectively. It returns a list of coordinates for npos positive patches that are sampled randomly from roimask.

"sample_patches_entropy_mask" takes as input an OCT image, a corresponding segmentation ground truth mask, an OCT region of interest mask, a patch shape, the number of positive and negative patches to extract, and values to indicate a positive and a negative label. It yields patches of the OCT image, the corresponding mask patch, and labels indicating the presence or absence of IRF, SRF, and PED within the patch.

Both functions are used in a pipeline for generating image patches where patches are extracted from the OCT image according to a binary mask. The script defines the "ImagePatchesByMaskRetouch_resampled" function, which is a Nut processor (using nutsml library) that applies "sample_patches_retouch_mask" to an iterable of images, masks, and labels to yield image patches, mask patches, and labels for IRF, SRF, and PED.

##### data_prepare.py
This script defines several functions to create a data iterator for training the model on OCT images. Data is rotated and patches are dropped randomly from the iterator if it does not contain any pathology. Images are preprocessed by subtracting the mean of the pixel values and dividing by the standard deviation. "get_iterator_nuts" function creates an iterator that reads OCT images, their corresponding masks, and their ROI masks from given directories. It also applies various augmentations to the input data and returns a batch of training data.

"get_data_iterator" function creates a data iterator that reads data from a CSV file, stratifies it according to the label distribution, shuffles it, applies various augmentations, randomly crops a patch from each image, preprocesses it, and filters it based on whether it contains any invalid pixels. The build_batch_train function is then used to construct batches of data for training.

#### Model
This script defines functions to create an instance of the U-Net model, a convolutional neural network used for image segmentation tasks. The script also defines the architecture of the descriminator to generate a binary classification. 

#### loss_functions.py
This script defines three functions to compute loss functions for a neural network. It defines the Dice loss for binary segmentation tasks. The function calculates true positive, false positive and false negative rates, and returns the Dice loss which is a measure of how similar the predicted segmentation map is to the true segmentation map.

#### train_model.py
This script carries out training of the model. The training is done for 100 epochs with the FScore (dice index) as a metric. The folder containing all model weights can be found [here](https://drive.google.com/file/d/1zA6AP6OruucBSpQ2Aw7moJIlPpeBWQgE/view?usp=share_link).

### Model Evaluation
The best evaluation scores obtained are: (preliminary results, not final)
loss: ### - f1-score: ###

Tensorboard logging is performed and the logs are shown below:

<p align="center">
<img src="https://user-images.githubusercontent.com/92387828/226880487-e688b12c-d0fa-40bd-a814-7c4f04be81e3.png" width=55% height=55%>>
</p>

### Model outputs 
#### test_model.py
The testing script takes in the OCT scan as input and outputs a segmentation mask for the input image. Image overlays are also produced with red color indicating abnormal areas. Some overlays are seen below: 

<p align="center">
  <img src="https://user-images.githubusercontent.com/92387828/226880487-e688b12c-d0fa-40bd-a814-7c4f04be81e3.png" width=30% height=30%>
</p>

### Results
The results will be shown later. 
