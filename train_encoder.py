import torch
from .models import Encoder, Decoder, EncoderDecoder
from .preprocess import train_loader
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Hyperparameters
latent_dim = 128
learning_rate = 1e-3
num_epochs = 10
in_channels = 3
out_channels = 3

# Initialize the model, loss function, and optimizer
model = EncoderDecoder(in_channels=in_channels, latent_dim=latent_dim, out_channels=out_channels)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        images = batch['images'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    # Calculate average loss over the epoch
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # sample images
    sample_images = images[:8]  # Take the first 8 images
    sample_outputs = outputs[:8]  # Corresponding reconstructed images
    
    save_image(sample_images, f'./ChangJae-RnE/sample_images/original_epoch_{epoch + 1}.png')
    save_image(sample_outputs, f'./ChangJae-RnE/sample_images/reconstructed_epoch_{epoch + 1}.png')


    
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')

# Save the model parameters
torch.save(model.state_dict(), "./ChangJae-RnE/encoder_decoder.pth")


# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./ChangJae-RnE/ed.png')