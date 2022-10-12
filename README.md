# Skin-Lesion-Analysis

Analysis of Skin Lesion Dermoscopic Images to segment out lesion regions, and classify the lesion type using Deep Learning.

## Data Directory

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
     - `\GAN\train\images`: 2594 images 
   - Validation Images
     - `\GAN\val\images`: 100 images
   - Test           
     - `\GAN\test\images`: 11720 images

3. Segmentation using BCD UNet

   - Training Images
     - `\Segmentation`: 14314 images 
   - Validation Images 
     - `\Segmentation\Data`: 100 images
   - Test            
     - `\Classification\training/Processed`: 25331 images

4. Segmented Classification

   - Unbalanced --- Segmented_train_120epochs --- 25331 images
   - Re-weights --- Segmented_train_120epochs --- 25331 images
   - SMOTE     --- Segmented_train_120epochs(train) --- 25331 images
                 --- segmented_smote(resampled) --- 61875 images
   - Balanced  --- segmented_smote           --- 61875 images  
