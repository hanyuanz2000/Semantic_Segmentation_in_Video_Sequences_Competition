import torch
import torch.nn as nn
from Reconstructor import Encoder, VideoFrameReconstructor

# Your DoubleConv, Down, Up, OutConv, and Encoder classes go here

# -----------------Test Encoder-----------------
# print("-----------------Test Encoder-----------------")
# # Initialize the encoder
# n_channels = 3  # Number of input channels
# n_classes = 10  # Just an example, set the number of output classes as required
# bilinear = False  # Set to True or False based on your requirement

# encoder = Encoder(n_channels, n_classes, bilinear)

# # Create a dummy input tensor of size (batch, C, H, W) = (2, 3, 240, 160)
# x = torch.randn(2, 3, 240, 160)
# print(f"Input size: {x.size()}")

# # Pass the input through the encoder
# output = encoder(x)

# # Print the size of the output
# print(f"Output size: {output.size()}")

# -----------------Test final_hidden of VideoFrameReconstructor-----------------
print("-----------------Test final_hidden of VideoFrameReconstructor-----------------")

# Initialize the VideoFrameReconstructor
# The parameters should be adjusted according to your model's requirements
num_frames = 11  # Number of frames in the sequence
num_layers = 2  # Number of layers in ConvLSTM
hidden_size = 512  # Hidden size of ConvLSTM
C, H, W = 3, 240, 160  # Input size
batch_size = 2  # Batch size
print(f'Input size: [{batch_size}, {num_frames}, {C}, {H}, {W}]')

model = VideoFrameReconstructor(num_frames, hidden_size, num_layers, C, H, W)

# Create a test input tensor with shape [B, T, C, H, W]
x = torch.rand(batch_size, num_frames, C, H, W)  # Batch size is 2

# Pass the test input through the model
output = model(x)
