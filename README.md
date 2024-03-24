# Skin Lesion Characterization using Conditional GANs and EfficientNet.

Analysis of **Skin Lesion Images** to segment lesion regions and characterize the lesion type using deep adversarial learning.

## Environment Setup

Set up the execution environment using the requirement files.
- Requirements for setting up **conda environment** are contained in `dep-file-conda.yml`.
- Requirements for setting up using **pip installations** (not recommended) are contained in `dep-file-pip.txt`.

## Running Experiments for Dataset Balancing and Lesion Segmentation. 

The dataset balancing analysis and lesion segmentation network is contained in [`notebooks`](./notebooks) as sequentially numbered python notebooks.

## Running Scaling Experiments for Classification

The classification architectures for classifiers are scripted in [`src/classifiers`](./src/classifiers). The preprocessing workflow used to prepare the dataset is in [`model-building`](.//model-building).

- Load the appropriate classifier drive function, say `experiment_effnetb6` in [`src/run.py`](./src/run.py) by importing them.
- Set up the data path.
- Call the driver function in [`src/run.py`](./src/run.py), and execute `python run.py`.

> [!Note]
> The dataset will be released during manuscript publication.

## Data Directory (*for internal reference only*)

1. Unsegmented Classification
  - Preprocess
    - `Unbalanced_train\ISIC_2019_Training_Input(input)`: 25331 images
    - `training/Processed`: 25331 images
  - Unbalanced
    - `training/Processed`: 25331 images
  - Re-weighted
    - `training/Processed`: 25331 images
  - SMOTE
    - `training/Processed(train)`: 25331 images
    - `training_balanced(resampled)`: 61875 images
  - Balanced    
    - `training_balanced`: 61875 images

2. Conditional GAN Model

   - Preprocess
     - `Segmentation/train/ISIC2018_Task1-2_Training_Input(input)`: 2594 images
     - `Segmentation/Processed_train`: 2594 images
   - Training Images
     - `GAN/train/images`: 2594 images 
   - Validation Images
     - `GAN/val/images`: 100 images
   - Test           
     - `GAN/test/images`: 11720 images

3. Segmentation using BCD UNet

   - Training Images
     - `Segmentation`: 14314 images 
   - Validation Images 
     - `Segmentation/Data`: 100 images
   - Test            
     - `Classification/training/Processed`: 25331 images

4. Segmented Classification

   - Unbalanced
    - `Segmented_train_120epochs`: 25331 images
   - Re-weights
    - `Segmented_train_120epochs`: 25331 images
   - SMOTE 
    - `Segmented_train_120epochs(train)`: 25331 images
    - `segmented_smote(resampled)`: 61875 images
   - Balanced  
    - `segmented_smote`: 61875 images  
