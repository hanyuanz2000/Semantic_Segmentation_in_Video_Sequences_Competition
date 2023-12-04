from UNet_GRU_Reconsructor import UNet_GRU_Reconsructor
import torch

# Define the network parameters
n_channels = 3   # number of channels in input frames
n_classes = 3    # assuming the number of output channels you want is same as input
gru_hidden_size = 512  # example hidden size for GRU
bilinear = False       # example setting for bilinear

# Initialize the network
model = UNet_GRU_Reconsructor(n_channels, n_classes, gru_hidden_size, bilinear)

# Create a dummy input tensor of shape (batch_size, sequence length, n_channel, H, W)
# Example: (4, 11, 3, 240, 160)
dummy_input = torch.randn(4, 11, 3, 240, 160)

# Pass the dummy input through the network
output = model(dummy_input)

# Print the shape of the output
print("Output shape:", output.shape)

# Verify if the output shape is as expected (batch_size, n_channels, H, W)
assert output.shape == (4, n_channels, 240, 160), "Output shape does not match expected shape"