import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Convolutional Variational Autoencoder (VAE) for 28x28 images (MNIST/Fashion).
    """
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder_body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # -> 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 64x7x7
            nn.ReLU(),
            nn.Flatten()
        )
        self.flat_size = 64 * 7 * 7
        
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flat_size)
        
        self.decoder_body = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  
            nn.Sigmoid() 
        )

    def reparameterize(self, mu, log_var):
        """Reparameterization Trick: z = mu + epsilon * sigma"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        z = self.reparameterize(mu, log_var)
        
        h_decode = self.decoder_input(z)
        h_decode = h_decode.view(-1, 64, 7, 7)
        reconstruction = self.decoder_body(h_decode)
        
        return reconstruction, mu, log_var


class ED_VAE(VAE):
    """
    Placeholder for the upgraded model (ED-VAE or VAE-GAN).
    Implement the new architecture here.
    """
    def __init__(self, latent_dim=20, img_size=28):
        super().__init__(latent_dim)
        pass

class Discriminator(nn.Module):
    def __init__(self, img_size=28):
        super().__init__()
        pass
