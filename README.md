# Semantic Segmentation in Video Sequences Competition
## Introduction
Welcome to our NYU cutting-edge deep learning project developed for the NYU Deep Learning 2023 Fall Final Competition. The primary goal of this project is to perform semantic segmentation in video sequences using a blend of labeled and unlabeled data. Our focus is on synthetic videos depicting simple 3D shapes governed by basic physics principles. Each video frame showcases diverse combinations of shapes, materials, and colors. The core task involves using the initial 11 frames of a video to predict the semantic segmentation mask for the final frame (the 22nd frame).

The dataset includes:
- Unlabeled Dataset: 13,000 videos, each containing 22 frames.
- Labeled Training Dataset: 1,000 videos with 22 frames each, accompanied by ground truth segmentation masks (in .npy format)
- Labeled Validation Dataset: 1,000 videos with 22 frames each, also with ground truth masks.

The data is avaiable [here](https://drive.google.com/file/d/1EFVaLyLNySimdLd-x-EHrh624jj3HA8y/view?usp=drive_link).

Our goal is to utilize this dataset to create a model that accurately predicts the semantic segmentation mask for the 22nd frame of each video, based on the first 11 frames.

## Approaches
We are exploring three main approaches to address this challenge:

### 1. Baseline Direct Prediction Using U-Net (Supervised Learning)
- **Description**: This approach involves using only the labeled training data with a U-Net architecture. We diretly train a Unet with the labeled data. For inference, the 11th frame is passed through our Unet model for segmentation.
- **Rationale**: U-Net is known for its effectiveness in semantic segmentation, especially when dealing with limited data. 

### 2. Dual-Phase Training Integrating Self-supervised and Supervised Learning

- **Brief Description**: he strategy involves two distinct phases. In the first phase, the goal is to use a reconstruction model to predict the 22nd frame from the initial 11 frames. The reconstruction model is trained with unlabeled datasets. In the second phase, the focus shifts to predicting the segmentation mask of the 22nd frame with Unet, building upon the work of the first phase.
- **Rationale**: This methodology aims to harness both labeled and unlabeled data effectively, thereby potentially enhancing the model's performance through a comprehensive learning approach.

- **Detailed Description**:

1. **Self-Supervised Learning Phase (Reconstruction Model):**
   - **Task:** Predict the 22nd frame using the first 11 frames (train with unlabeled videos).
   - **Encoder:** An encoder composed of Conv layers and Maxpooling layers is employed to process each of the first 11 frames, focusing on efficient feature extraction.
   - **Temporal Dynamics Analysis:** The ConvLSTM layer is used to analyze temporal evolution across frames, integrating the extracted features into a coherent temporal sequence.
   - **Decoder:** The last hiden layer generated from the ConvLSTM is then pass to decoder, which is composed of A bilinear upsampling and few convs. In this way, we can generate the reconstructed 22nd frame.

2. **Supervised Learning Phase (Segmentation Model):**
   - **Task:** Predict the semantic segmentation mask
   - **Traning Data:** Training Data: 22,000 frame and their corresponding Mask from labeled training set (1000 videos and 22 frames for each videp)
   - **U-Net Architecture:** Utilizes a Unet segmentation model, currently designed to process individual frames without considering object movement

## How to run the code (train and inference)
Your GitHub README section for running the code is quite detailed but could benefit from a few refinements for clarity and professionalism. Here are some suggestions:

1. **Title Refinement**: Use a clearer title. Instead of "How to run the code (train and inference)," consider something more descriptive, like "Training and Inference Instructions."

2. **Formatting and Structure**: Use consistent formatting for directory names and script paths. Also, consider using bullet points for steps and organizing the content for better readability.

3. **Typos and Grammar**: Correct spelling errors (e.g., "optinal" should be "optional," "diretory" should be "directory") and improve the grammar for professionalism.

4. **Clarification of Instructions**: Provide a bit more detail in the optional arguments section, like a brief mention of what kind of arguments one can expect.

5. **Consistency in Naming**: Ensure consistency in script names (e.g., the mismatch between `inference_val_with_reconstuctor.py` and `python inference_hidden_without_reconstuctor.py`).

Here's a revised version incorporating these suggestions:

---

## Training and Inference Instructions

This section guides you through the process of training and conducting inference using various scripts in our repository.

### Training Scripts

- **Reconstructor Model Training**:
  - Script: [train_reconstructor.py](SSL_convLSTM_Reconstructor/train_reconstructor.py)
  - Directory: `SSL_convLSTM_Reconstructor`
  - Command: `python train_reconstructor.py`
  - Note: For optional arguments, refer to the script documentation.

- **Segmentation Model Training**:
  - Script: [train.py](Unets/train.py)
  - Directory: `Unets`
  - Command: `python train.py`
  - Note: For optional arguments, refer to the script documentation.

### Inference Scripts

- **Inference with Reconstructor (Validation Set)**:
  - Script: [inference_val_with_reconstructor.py](Unets/inference_val_with_reconstructor.py)
  - Description: This script performs inference on the validation set using both the Reconstructor and Segmentation models. It uses the first 11 frames to predict the 22nd frame, which is then used for segmentation.
  - Directory: `Unets`
  - Command: `python inference_val_with_reconstructor.py`

- **Inference without Reconstructor (Validation Set)**:
  - Script: [inference_val_without_reconstructor.py](Unets/inference_val_without_reconstructor.py)
  - Description: This script performs inference on the validation set using only the Segmentation model. It directly uses the 11th frame for inference, bypassing the Reconstructor due to its reconstruction quality limitations.
  - Directory: `Unets`
  - Command: `python inference_val_without_reconstructor.py`

---

This revised version aims to make your README section more user-friendly, clear, and professional.

## Evaluation
Our models' efficacy will be rigorously assessed utilizing the Jaccard Index, which compares the congruence between the predicted segmentation masks and the actual ground truth within the validation set and another concealed hidden test set.

## Results
The Dual-Phase model registers a validation Jaccard Index of 0.16, in contrast to the baseline model, which attains a validation Jaccard Index of 0.24. This notable disparity predominantly stems from the inadequate quality of the reconstruction of the 22nd frame.

## Reference Repo
- https://github.com/milesial/Pytorch-UNet
- https://github.com/ndrplz/ConvLSTM_pytorch