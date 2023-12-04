import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the full model that combines the encoder and reconstructor
class VideoFrameReconstructor(nn.Module):
    def __init__(self, num_frames, hidden_size, num_layers, height, width):
        super(VideoFrameReconstructor, self).__init__()
        self.encoder = Encoder()
        self.reconstructor = Reconstructor(30*20*128, hidden_size, num_layers)
        # Define the decoder structure (transposed convolutions)
        self.decoder = Decoder()

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        # Flatten out the time dimension
        x = x.view(batch_size * timesteps, C, H, W)
        # Pass each frame through the encoder
        x = self.encoder(x)
        # Reshape for the GRU
        x = x.view(batch_size, timesteps, -1)
        # Reconstruct the final frame
        x = self.reconstructor(x)
        # Reshape and pass through the decoder
        x = x.view(batch_size, 128, 30, 20)  # Reshape to the expected dimensions before the decoder
        x = self.decoder(x)
        return x


# Define the encoder structure
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Define convolutional layers here
        self.conv_layers = nn.Sequential(
            # Convolutional layer followed by BatchNorm and ReLU
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolutional layer followed by BatchNorm and ReLU
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Last convolutional layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv_layers(x)

# Define the reconstructor with GRU
class Reconstructor(nn.Module):
    def __init__(self, encoded_size, hidden_size, num_layers):
        super(Reconstructor, self).__init__()
        self.gru = nn.GRU(encoded_size, hidden_size, num_layers, batch_first=True)
        # Map the final GRU output back to the encoded size
        self.fc = nn.Linear(hidden_size, encoded_size)

    def forward(self, x):
        x, _ = self.gru(x)
        # We take the last GRU output for reconstruction
        x = self.fc(x[:, -1, :])
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Start with the output from the GRU, which we'll assume has been flattened
        # You'll need to adjust the sizes here to match the output of your GRU
        
        # Assuming we're starting with a feature map of size [batch, 128, 15, 10]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Final layer to get back to 3 channels; use kernel_size=2, stride=2 to double the spatial dimensions
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.Tanh()  # Use Tanh or Sigmoid depending on your image normalization
        )
    
    def forward(self, x):
        return self.decoder(x)
