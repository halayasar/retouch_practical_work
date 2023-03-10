# retouch_practical_work
### Introduction
Optical coherence tomography (OCT) is a non-invasive imaging technology that allows for high-resolution visualization of biological tissues at the micrometer scale. OCT is a widely used imaging technique for the diagnosis and management of retinal diseases. However, manual segmentation of the retinal layers is a time-consuming and error-prone process and may vary between different clinicians. To overcome this limitation, automated segmentation algorithms have been developed to segment the retinal layers accurately and efficiently. The combination of machine learning and automated segmentation of retinal layers in retinal OCT has the potential to improve the diagnosis and management of retinal diseases, by providing accurate and reliable information about the structure and function of the retina.
This repository implements the method proposed by Tennakon et al. The method is an end-to-end trained deep learning based retinal fluid segmentation method that works across 3D-OCT images that are acquired using devices from three different vendors: Cirrus, Spectralis and Topcon. 

### Implementation
#### Prepare data 
This script is run first on the training dataset provided. This script contains three functions: preprocess_oct_images, create_test_train_set, and create_roi_masks. The function preprocess_oct_images loads OCT (optical coherence tomography) images and corresponding segmentation masks from a directory, preprocesses them by histogram matching and applying a median filter, and saves them to a new directory. The function also creates a CSV file that stores the image name, vendor, root, slice, and the presence or absence of intraretinal fluid (IRF), subretinal fluid (SRF), and pigment epithelial detachment (PED) in each slice.

The function create_test_train_set splits the preprocessed OCT images into training and testing sets, and saves the corresponding root paths of the directories containing the training and testing data to CSV files. The function create_roi_masks loads the preprocessed OCT images and segmentation masks, applies a morphological closing operation and entropy thresholding to the segmentation masks, and saves the resulting binary masks to a new directory.

The preprocessed data can be found [here](https://drive.google.com/file/d/1zA6AP6OruucBSpQ2Aw7moJIlPpeBWQgE/view?usp=share_link).

#### Config
Script with all hyperparameters needed for training and preprocessing. The fluids IRF, SRF and PED are coded as 1, 2 and 3, respectively. The total number of classes, including the background, is 4 and the patches produced are of a size (256,128). 

#### Utils folder
##### Base, losses.py, metrics.py
The metrics.py script in the segmentation_models library implements various evaluation metrics commonly used in image segmentation tasks. The metrics include Intersection over Union (IoU), Dice coefficient, Precision, Recall, F1 score, and Average Precision. These metrics can be used to measure the accuracy and performance of segmentation models. The metrics chosen in this script is the IOU Score and FScore (Dice coefficient). The IoU and Dice coefficient measure the overlap between the predicted and ground truth segmentation masks. Additionally, the metrics support multiple classes and can be used for both binary and multi-class segmentation tasks.

##### custom_nuts.py
The script consists of two functions: "sample_oct_patch_centers" and "sample_patches_entropy_mask". "sample_oct_patch_centers" takes as input a binary image roimask, a patch shape pshape, the number of positive samples to extract npos, and values to indicate a positive and a negative label, pos and neg respectively. It returns a list of coordinates for npos positive patches that are sampled randomly from roimask.

"sample_patches_entropy_mask" takes as input an OCT image, a corresponding segmentation ground truth mask, an OCT region of interest mask, a patch shape, the number of positive and negative patches to extract, and values to indicate a positive and a negative label. It yields patches of the OCT image, the corresponding mask patch, and labels indicating the presence or absence of IRF, SRF, and PED within the patch.

Both functions are used in a pipeline for generating image patches where patches are extracted from the OCT image according to a binary mask. The script defines the "ImagePatchesByMaskRetouch_resampled" function, which is a Nut processor (using nutsml library) that applies "sample_patches_retouch_mask" to an iterable of images, masks, and labels to yield image patches, mask patches, and labels for IRF, SRF, and PED.

##### data_prepare.py
This script defines several functions to create a data iterator for training the model on OCT images. Data is rotated and patches are dropped randomly from the iterator if it does not contain any pathology. Images are preprocessed by subtracting the mean of the pixel values and dividing by the standard deviation. "get_iterator_nuts" function creates an iterator that reads OCT images, their corresponding masks, and their ROI masks from given directories. It also applies various augmentations to the input data and returns a batch of training data.

"get_data_iterator" function creates a data iterator that reads data from a CSV file, stratifies it according to the label distribution, shuffles it, applies various augmentations, randomly crops a patch from each image, preprocesses it, and filters it based on whether it contains any invalid pixels. The build_batch_train function is then used to construct batches of data for training.

### Model_utils folder
#### Model
This script defines functions to create an instance of the U-Net model, a convolutional neural network used for image segmentation tasks. The model consists of an encoder, which downsamples the input image to extract features, and a decoder, which upsamples the feature map to generate a segmentation map with the same size as the input image. A plot of the model can be seen in the plots folder in the repository. 

The "create_encoder_layer" function defines a single layer of the encoder, which includes convolutional and batch normalization layers. If it is the first layer of the encoder, a 7x7 convolutional layer is used to process the input image. Otherwise, a max pooling layer is used to downsample the feature map.

The "create_decoder_layer" function defines a single layer of the decoder, which includes a deconvolutional layer, a concatenation layer that merges the feature map from the corresponding encoder layer, and a convolutional layer. Spatial dropout and batch normalization layers can be included in the decoder. Cropping is performed on the encoder feature map to ensure that it has the same size as the upsampled decoder feature map.

The "create_unet_model" function creates an instance of the U-Net model with a specific architecture. The model includes four encoder layers and three decoder layers. The first decoder layer has the smallest size, while the last decoder layer has the same size as the input image. The decoder layers also include additional upsampling and cropping layers to match the size of the encoder feature map. The model takes an input image of size (224, 224, 3) and produces a segmentation map with the same size.

#### loss_functions.py
This script defines three functions to compute loss functions for a neural network. "get_dice_loss" function returns the Dice loss for binary segmentation tasks. The function calculates true positive, false positive and false negative rates, and returns the Dice loss which is a measure of how similar the predicted segmentation map is to the true segmentation map.

"get_balanced_cross_entropy_loss_function" is a function that calculates the cross-entropy loss, which is a commonly used loss function for classification tasks. The function returns the average loss over all classes. "get_combined_cross_entropy_and_dice_loss_function" is a function that computes the combined loss of Dice loss and cross-entropy loss. 

#### train_model.py
This script carries out training of the segmentation model. The training is done for 100 epochs with the IOU Score and F Score as metrics. All model weights are saved and the best weight is named appropriately. The folder containing all the weights can be found [here](https://drive.google.com/file/d/1zA6AP6OruucBSpQ2Aw7moJIlPpeBWQgE/view?usp=share_link).

#### train_adversarial.py
This script carries out GAN training. The best segmentation model weight is loaded and used in the training. The GAN model is trained with each epoch involving training the generator and discriminator alternately, with batches of real and fake images. The generator attempts to create realistic fake images, and the discriminator tries to distinguish between the real and fake images. This process is repeated for multiple epochs until the discriminator can no longer distinguish between the real and fake images. As the segmentation model (generator) is trained, the weights are also saved in a separate folder for the adversarial weights. All generator weights can be found [here](https://drive.google.com/file/d/1zA6AP6OruucBSpQ2Aw7moJIlPpeBWQgE/view?usp=share_link). A plot of the discriminator model used is in the plots folder in the repository.

### Model Evaluation
Again, the best segmentation model weight after adversarial training is loaded and used to evaluate the model. The evaluation, as mentioned, is done with the IOU Score and F Score as metrics. There is an evaluation script for before and after the adversarial training to see the effect of adversarial training on the model. 

So far, the best evaluation scores obtained are: (preliminary results, not final)
- before adversarial training:
loss: 0.1652 - iou_score: 0.4546 - f1-score: 0.5276

- after adversarial training:
loss: 0.3532 - iou_score: 0.6069 - f1-score: 0.667

Tensorboard logging is performed and all logging folders can be found [here](https://drive.google.com/file/d/12TeeCgwxXRfMW-IJ-SijKOdwZQhV1yo-/view?usp=share_link). The training and validation logs are shown below:

<p align="center">
<img src="https://user-images.githubusercontent.com/92387828/224982269-808c86e4-dee1-42a9-9985-24b182493704.PNG" width=35% height=35%> <img src="https://user-images.githubusercontent.com/92387828/224982141-db3de048-7fc1-49a7-af16-6e9c19c9ba86.PNG" width=35% height=35%>
</p>

### Model outputs 
#### test_model.py
The testing script takes in the OCT scan as input and outputs a segmentation mask for the input image. The radius of the circular crop of the image is found and preprocessing is performed on the image (similar to that done on training dataset). Afterwards, the output of the segmentation model is used to set the class with the highest probability as the predicted class. Then, thresholding is performed to the probabilities based on a threshold value of 0.5 and returns a binary mask.
The script then uses an image, crops it into smaller patches, and passes each patch through prediction function to get a binary mask for that patch. It then combines the binary masks for each patch to get the final segmentation mask for the whole image. Image overlays are also produced. Some overlays are seen below: 

<p align="center">
  <img src="https://user-images.githubusercontent.com/92387828/224986209-511d9549-2a92-4469-b555-c13bc44fdc84.PNG" width=30% height=30%> <img src="https://user-images.githubusercontent.com/92387828/224986316-20e40e06-ad22-4e28-b6f6-f47aee5125c8.PNG" width=30% height=30%> <img src="https://user-images.githubusercontent.com/92387828/224986377-0bd3ca42-4483-4fc7-bb22-ae36643023e7.PNG" width=30% height=30%>
</p>

All predicted outputs can be found [here](https://drive.google.com/file/d/1zA6AP6OruucBSpQ2Aw7moJIlPpeBWQgE/view?usp=share_link).

### Results
The results will be shown later. 
