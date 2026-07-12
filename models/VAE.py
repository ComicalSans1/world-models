import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, img_channels=3, n_hidden=32):
        super().__init__()
        self.n_hidden = n_hidden
        self.img_channels = img_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),          
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),       
            nn.ReLU()
        )
        
        self.mu = nn.Linear(4096, n_hidden)
        self.logstd = nn.Linear(4096, n_hidden)
         
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)

        mu = self.mu(x)
        logstd = self.logstd(x)
        return mu, logstd

class Decoder(nn.Module):
    def __init__(self, img_channels=3, n_hidden=32):
        super().__init__()
        self.img_channels = img_channels
        self.fc = nn.Linear(n_hidden, 4096)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),   
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=6, stride=2), 
        )
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 256, 4, 4)
        x = self.deconv(x)

        x = x.permute(0, 2, 3, 1)
        return x

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = Encoder(3, n_hidden = 32)
        self.Decoder = Decoder(3, n_hidden = 32)

    def forward(self, x):
        mu, logstd = self.Encoder(x)
        
        std = logstd.exp()
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std

        decoded = self.Decoder(z)
        return mu, logstd, decoded