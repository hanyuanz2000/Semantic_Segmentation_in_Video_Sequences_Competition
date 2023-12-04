from video_frame_reconstructor import VideoFrameReconstructor
import torch

# Define the network parameters
# Parameters
num_frames = 11
hidden_size = 512
num_layers = 1
height = 30  # The height of the feature map before flattening
width = 20   # The width of the feature map before flattening

# Initialize the model
model = VideoFrameReconstructor(num_frames, hidden_size, num_layers, height, width)

# Test with a dummy input
dummy_input = torch.randn(1, num_frames, 3, 240, 160)  # Batch size, timesteps, channels, height, width
output = model(dummy_input)
print("Output shape:", output.shape)