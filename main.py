import torch
import torch.nn as nn
import torch.optim as optim

# Define the encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (B, 128, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # (B, 256, H/8, W/8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*8*8, latent_dim) # (B, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

# Define the decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (B, 128, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),    # (B, 3, H, W)
            nn.Sigmoid()  # Output pixel values between 0 and 1
        )
    
    def forward(self, x):
        return self.decoder(x)

class DiffusionModel(nn.Module):
    def __init__(self, latent_dim):
        super(DiffusionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x, t):
        # x is the latent representation
        # t is the diffusion step (can be used for controlling noise level)
        noise = torch.randn_like(x) * (1 / (t + 1))
        return self.model(x + noise)  # simple noise + denoise step

class ImageToImageDiffusion(nn.Module):
    def __init__(self, encoder, decoder, diffusion_model):
        super(ImageToImageDiffusion, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion_model = diffusion_model
    
    def forward(self, x, diffusion_steps=10):
        # Step 1: Encode the image to latent space
        z = self.encoder(x)
        
        # Step 2: Apply diffusion process in latent space
        for t in range(diffusion_steps):
            z = self.diffusion_model(z, t)
        
        # Step 3: Decode the latent representation back to image space
        x_hat = self.decoder(z)
        return x_hat

# Example: initialize the autoencoder
latent_dim = 128
encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)

# Initialize the full model
diffusion_steps = 10
image_to_image_diffusion = ImageToImageDiffusion(encoder, decoder, DiffusionModel(latent_dim))

# Example usage with dummy data
input_image = torch.rand((1, 3, 64, 64))  # A random 64x64 image
output_image = image_to_image_diffusion(input_image, diffusion_steps)

# Initialize the full model
diffusion_steps = 10
image_to_image_diffusion = ImageToImageDiffusion(encoder, decoder, DiffusionModel(latent_dim))

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(image_to_image_diffusion.parameters(), lr=1e-3)

# Dummy training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output_image = image_to_image_diffusion(input_image, diffusion_steps)
    loss = criterion(output_image, input_image)  # Comparing output and input
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

