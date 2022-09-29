# Skin-Lesion-Analysis

Analysis of Skin Lesion Dermoscopic Images to segment out, and subtype lesions using Deep Learning.

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

2. CGAN model

   6. Pre-process     --- Segmentation/train/ISIC2018_Task1-2_Training_Input(input) --- 2594 images
                      --- Segmentation/Processed_train --- 2594 images
   7. Training images --- \GAN\train\images                 --- 2594 images 
      Validat  images --- \GAN\val\images                 --- 100 images
      Test            --- \GAN\test\images                --- 11720 images

3. Segmentation -- BCDU-Net

   8. Training images --- \Segmentation\Data                 --- 14314 images 
      Validat  images --- \Segmentation\Data                 --- 100 images
      Test            --- \Classification\training/Processed --- 25331 images

4. Segmented Classification

   9. Unbalanced --- Segmented_train_120epochs --- 25331 images
   10. re-weights --- Segmented_train_120epochs --- 25331 images
   11. smote      --- Segmented_train_120epochs(train) --- 25331 images
                 --- segmented_smote(resampled) --- 61875 images
   12. balanced   --- segmented_smote           --- 61875 images  
