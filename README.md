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

### 1. Direct Prediction Using U-Net (Supervised Learning)
- **Description**: This approach involves using only the labeled training data with a U-Net architecture. Considering the limited training data, we concatenate the first 11 frames along the channel dimension to get a 33x240x160 tensor as input.
- **Rationale**: U-Net is known for its effectiveness in semantic segmentation, especially when dealing with limited data.

### 2. Dual-Phase Training Integrating Self-supervised and Supervised Learning

- **Brief Description**: The strategy involves two distinct phases. In the first phase, the goal is to predict the 22nd frame from the initial 11 frames using both unlabeled and labeled datasets. In the second phase, the focus shifts to predicting the segmentation mask of the 22nd frame, building upon the work of the first phase.
- **Rationale**: This methodology aims to harness both labeled and unlabeled data effectively, thereby potentially enhancing the model's performance through a comprehensive learning approach.

- **Detailed Description**:

1. **Self-Supervised Learning Phase:**
   - **Task:** Predict the 22nd frame using the first 11 frames from both unlabeled and labeled training videos.
   - **Simplified Encoder:** A streamlined encoder is employed to process each of the first 11 frames, focusing on efficient feature extraction.
   - **Temporal Dynamics Analysis:** The ConvLSTM layer is used to analyze temporal evolution across frames, integrating the extracted features into a coherent temporal sequence.
   - **Reconstruction Output:** The output from the GRU layer is then utilized to reconstruct the appearance of the 22nd frame. The output is formatted as (batch size, channels, height, width).
   - **Efficiency Focus:** This phase emphasizes computational efficiency and effective temporal feature capture without the complexity of U-Net.

2. **Supervised Learning Phase:**
   - **Task:** Develop a segmentation mask for the 22nd frame, using the model's understanding from the first phase.
   - **U-Net Architecture:** Leveraging U-Net's strengths in image segmentation, the model focuses on accurately predicting the segmentation mask of the 22nd frame.
   - **Integration of Phases:** The spatial understanding gained in the first phase is combined with U-Net's capabilities to enhance mask prediction accuracy.
   - **Segmentation Head:** A specialized segmentation head is used to refine and finalize the mask prediction.

### Conclusion

This two-phase approach strategically aligns the model's strengths with each phase's objectives. By balancing computational efficiency and the ability to capture both spatial and temporal dynamics, this strategy is poised to optimize overall performance in semantic segmentation of video sequences.

## Evaluation
The performance of our models will be evaluated based on the Intersection over Union (IoU) between the predicted segmentation masks and the ground truth for the hidden test set.

## Repository Structure
