import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

from models import VAE
from data_loader import get_mnist_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 20
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

def vae_loss_function(recon_x, x, mu, log_var):
    """
    Calculates the VAE loss: Reconstruction Loss + KL Divergence.
    """

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
  
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return (BCE + KLD) / x.size(0)

def train_vae(model, train_loader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(1, num_epochs + 1):
        overall_loss = 0
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=True)
        
        for batch_idx, (data, _) in enumerate(loop):

            data = data.to(device)
            
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(data)

            loss = vae_loss_function(recon_batch, data, mu, log_var)
            
            loss.backward()
            optimizer.step()
            
            overall_loss += loss.item()
            loop.set_postfix(loss=(overall_loss / (batch_idx + 1)))

        avg_loss = overall_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

def generate_and_save_images(model, device, num_images=10, filename="results/generated_sample.png"):
    """Generates images by sampling from the latent space."""
    model.eval()
    with torch.no_grad():
       
        z = torch.randn(num_images, LATENT_DIM).to(device)
    
        h_decode = model.decoder_input(z)
        h_decode = h_decode.view(-1, 64, 7, 7)
    
        generated_images = model.decoder_body(h_decode).cpu()
    
        fig, axes = plt.subplots(1, num_images, figsize=(10, 1))
        for i, ax in enumerate(axes.flat):
            img = generated_images[i].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        
        os.makedirs('results', exist_ok=True)
        plt.savefig(filename)
        plt.close(fig)
        print(f"Generated images saved to {filename}")


if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    train_loader, _ = get_mnist_dataloaders(BATCH_SIZE)

    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_vae(model, train_loader, optimizer, DEVICE, NUM_EPOCHS)

    os.makedirs('results/checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'results/checkpoints/vae_mnist_final.pth')
    print("Model saved.")
    
    generate_and_save_images(model, DEVICE)
