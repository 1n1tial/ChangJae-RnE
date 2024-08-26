import torch
import torch.nn as nn
from diffusers import UNet2DModel
from .config import TrainingConfig

model = UNet2DModel(
    sample_size=TrainingConfig.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ), 
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
      ),
)

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1),  # Downsample to latent dim
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # Upsample to output dim
            nn.Sigmoid(),  # Assuming image output
        )

    def forward(self, x):
        return self.decoder(x)

class EncoderDecoder(nn.Module):
    def __init__(self, in_channels, latent_dim, out_channels):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, out_channels)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
latent_dim = 256
encoder = Encoder(in_channels=3, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, out_channels=3)
ed_model = EncoderDecoder(in_channels=3, latent_dim=latent_dim, out_channels=3)

class CustomUNet2DModel(UNet2DModel):
    def __init__(self, encoder, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, timesteps, *args, **kwargs):
        # Encode the input
        encoded_x = self.encoder(x)
        
        # Apply the diffusion process using the parent class
        diffusion_output = super().forward(encoded_x, timesteps, *args, **kwargs)
        
        # Decode the output back to the original size
        decoded_output = self.decoder(diffusion_output[0])
        
        return decoded_output, 0

# Instantiate the encoder, UNet model (with appropriate params), and decoder
latent_dim = 256
encoder = Encoder(in_channels=3, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, out_channels=3)

# Define the UNet2D model with appropriate parameters (e.g., resolution, number of channels, etc.)
model2 = CustomUNet2DModel(
    encoder=encoder,
    decoder=decoder,
    sample_size=TrainingConfig.image_size,  # or whatever your input size is
    in_channels=latent_dim,  # should match the latent_dim from encoder
    out_channels=latent_dim,  # typically match latent_dim
    layers_per_block=2,
    block_out_channels=(64, 64, 128, 128, 256),  # the number of output channes for each UNet block
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    ), 
    up_block_types=(
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
      ),
)


from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

optimizer = torch.optim.AdamW(model2.parameters(), lr=TrainingConfig.learning_rate)

print(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model2.parameters()))