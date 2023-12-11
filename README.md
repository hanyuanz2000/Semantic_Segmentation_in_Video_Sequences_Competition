# Semantic Segmentation in Video Sequences Competition
## Introduction
Welcome to our NYU cutting-edge deep learning project developed for the NYU Deep Learning 2023 Fall Final Competition. The primary goal of this project is to perform semantic segmentation in video sequences using a blend of labeled and unlabeled data. Our focus is on synthetic videos depicting simple 3D shapes governed by basic physics principles. Each video frame showcases diverse combinations of shapes, materials, and colors. The core task involves using the initial 11 frames of a video to predict the semantic segmentation mask for the final frame (the 22nd frame).

The dataset includes:
- Unlabeled Dataset: 13,000 videos, each containing 22 frames.
- Labeled Training Dataset: 1,000 videos with 22 frames each, accompanied by ground truth segmentation masks (in .npy format)
- Labeled Validation Dataset: 1,000 videos with 22 frames each, also with ground truth masks.

Our goal is to utilize this dataset to create a model that accurately predicts the semantic segmentation mask for the 22nd frame of each video, based on the first 11 frames.

## Approaches
We are exploring three main approaches to address this challenge:

### 1. Baseline Direct Prediction Using U-Net (Supervised Learning)
- **Description**: This approach involves using only the labeled training data with a U-Net architecture. We diretly train a Unet with the labeled data, and use the segmenatation result of the 11th frame as our mask prediction on the 22nd frame.
- **Rationale**: U-Net is known for its effectiveness in semantic segmentation, especially when dealing with limited data. 

### 2. Dual-Phase Training Integrating Self-supervised and Supervised Learning

- **Brief Description**: This method employs a U-Net architecture solely utilizing labeled training data. The process involves directly training a U-Net model with labeled data, aiming to predict the segmentation mask for the 22nd frame based on the ground truth 11th frame.
- **Rationale**: This methodology aims to harness both labeled and unlabeled data effectively, thereby potentially enhancing the model's performance through a comprehensive learning approach.

- **Detailed Description**:

1. **Self-Supervised Learning Phase:**
   - **Task:** Predict the 22nd frame using the first 11 frames (train with unlabeled videos).
   - **Simplified Encoder:** A streamlined encoder is employed to process each of the first 11 frames, focusing on efficient feature extraction.
   - **Temporal Dynamics Analysis:** The ConvLSTM layer is used to analyze temporal evolution across frames, integrating the extracted features into a coherent temporal sequence.
   - **Reconstruction Output:** The output from the ConvLSTM layer is then utilized to reconstruct the appearance of the 22nd frame. The output is formatted as (batch size, channels, height, width).
   - **Efficiency Focus:** This phase emphasizes computational efficiency and effective temporal feature capture without the complexity of U-Net.

2. **Supervised Learning Phase:**
   - **Task:** Develop a segmentation mask for the 22nd frame, using the model's understanding from the first phase.
   - **U-Net Architecture:** Leveraging U-Net's strengths in image segmentation, the model focuses on accurately predicting the segmentation mask of the 22nd frame.
   - **Integration of Phases:** The spatial understanding gained in the first phase is combined with U-Net's capabilities to enhance mask prediction accuracy.
   - **Segmentation Head:** A specialized segmentation head is used to refine and finalize the mask prediction.

## Evaluation
Our models' efficacy will be rigorously assessed utilizing the Jaccard Index, which compares the congruence between the predicted segmentation masks and the actual ground truth within the validation set and another concealed hidden test set.

## Results
The Dual-Phase model registers a validation Jaccard Index of 0.16, in contrast to the baseline model, which attains a validation Jaccard Index of 0.24. This notable disparity predominantly stems from the inadequate quality of the reconstruction of the 22nd frame.