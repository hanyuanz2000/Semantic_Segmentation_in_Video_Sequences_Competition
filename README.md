# Semantic Segmentation in Video Sequences Competition
## Introduction
Welcome to our NYU cutting-edge deep learning project developed for the NYU Deep Learning 2023 Fall Final Competition. The challenge involves semantic segmentation in video sequences using a mix of labeled and unlabeled data. Our dataset features synthetic videos with simple 3D shapes interacting according to basic physics principles, where each frame presents unique combinations of shapes, materials, and colors. The task is to use the first 11 frames of a video to generate the semantic segmentation mask of the last frame (the 22nd frame).


## Problem Description
The dataset includes:
- 13,000 unlabeled videos with 22 frames each.
- 1,000 labeled training videos with 22 frames each.
- 1,000 labeled validation videos with 22 frames each.

Our goal is to utilize this dataset to create a model that accurately predicts the semantic segmentation mask for the 22nd frame of each video, based on the first 11 frames.

## Approaches
We are exploring three main approaches to address this challenge:

### 1. Direct Prediction Using U-Net
- **Description**: This approach involves using only the labeled training data with a U-Net architecture. Considering the limited training data, we concatenate the first 11 frames along the channel dimension to get a 33x240x160 tensor as input.
- **Rationale**: U-Net is known for its effectiveness in semantic segmentation, especially when dealing with limited data.

### 2. U-Net as Encoder with Unlabeled Data Utilization
- **Description**: In this method, we use U-Net as an encoder. We train the encoder using both labeled and unlabeled data. The encoder tries to predict the 22nd frame from the first 11 frames. This approach allows us to leverage the unlabeled data. Once the encoder is trained, we use the labeled training data to train a predictor.
- **Rationale**: This method aims to effectively utilize both labeled and unlabeled data, potentially enhancing the model's performance through a broader learning scope.

### 3. JEPA
- **Description**: 
- **Rationale**: 

## Evaluation
The performance of our models will be evaluated based on the Intersection over Union (IoU) between the predicted segmentation masks and the ground truth for the hidden test set.

## Repository Structure
