import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        # print(tensor.size())
        return tensor.view(self.size)


class AE_MNIST(nn.Module):
    """Encoder-Decoder architecture for a typical autoencoder."""
    def __init__(self, z_dim=8, nc=1, model_type='probabilistic'):
        super(AE_MNIST, self).__init__()
        self.type = model_type
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024 * 1 * 1))         # B, 1024*4*4
        )

        if self.type == 'probabilistic':
            self.fc1 = nn.Linear(1024 * 1 * 1, z_dim)    # B, z_dim
            self.fc2 = nn.Linear(1024 * 1 * 1, z_dim)    # B, z_dim
        else:
            self.fc = nn.Linear(1024 * 1 * 1, z_dim)    # B, z_dim

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024 * 7 * 7),   # B, 1024*8*8
            View((-1, 1024, 7, 7)),   # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            # nn.BatchNorm2d(128),
            # nn.ReLU(True),
            nn.ConvTranspose2d(256, nc, 1)                       # B,   nc, 64, 64
            # nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        if self.type == 'probabilistic':
            z, mu, logvar = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z, mu, logvar
        else:
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z

    def encode(self, x):
        z = self.encoder(x)
        if self.type == 'probabilistic':
            mu = self.fc1(z)
            logvar = self.fc2(z)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            z = self.fc(z)
            return z

    def decode(self, z):
        return self.decoder(z)


class AE_CelebA(nn.Module):
    """Encoder-Decoder architecture for a typical autoencoder."""
    def __init__(self, z_dim=10, nc=3, model_type='probabilistic'):
        super(AE_CelebA, self).__init__()
        self.type = model_type
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024 * 4 * 4))         # B, 1024*4*4
        )

        if self.type == 'probabilistic':
            self.fc1 = nn.Linear(1024 * 4 * 4, z_dim)    # B, z_dim
            self.fc2 = nn.Linear(1024 * 4 * 4, z_dim)    # B, z_dim
        else:
            self.fc = nn.Linear(1024 * 4 * 4, z_dim)    # B, z_dim

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024 * 8 * 8),   # B, 1024*8*8
            View((-1, 1024, 8, 8)),   # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        if self.type == 'probabilistic':
            z, mu, logvar = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z, mu, logvar
        else:
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z

    def encode(self, x):
        z = self.encoder(x)
        if self.type == 'probabilistic':
            mu = self.fc1(z)
            logvar = self.fc2(z)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            z = self.fc(z)
            return z

    def decode(self, z):
        return self.decoder(z)


class AE_MLP(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, model_type: str = 'probabilistic'):
        super(AE_MLP, self).__init__()
        self.type = model_type
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.fc1 = nn.Linear(x_dim, 50)
        if self.type == 'probabilistic':
            self.fc21 = nn.Linear(50, z_dim)
            self.fc22 = nn.Linear(50, z_dim)
        else:
            self.fc2 = nn.Linear(50, z_dim)
        self.fc3 = nn.Linear(z_dim, 50)
        self.fc4 = nn.Linear(50, x_dim)

    def encode(self, x):
        z = F.relu(self.fc1(x))
        if self.type == 'probabilistic':
            mu = self.fc21(z)
            logvar = self.fc22(z)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            z = self.fc2(z)
            return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        if self.type == 'probabilistic':
            z, mu, logvar = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z, mu, logvar
        else:
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, z


class SimpleVAE(nn.Module):
    def __init__(self, x_dim: int, z_dim: int):
        super(SimpleVAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, z_dim)
        self.fc32 = nn.Linear(256, z_dim)
        self.fc4 = nn.Linear(z_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, x_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        h4 = F.relu(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def load_datasets(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    name = args.source_data
    pathname = args.datapath
    data_set = getattr(datasets, name)
    data_loaders = {}
    print(name)
    if name == 'MNIST' or name == 'FashionMNIST' or name == 'KMNIST' or name == 'USPS':
        data_loaders['train'] = torch.utils.data.DataLoader(
            data_set(pathname, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        data_loaders['val'] = torch.utils.data.DataLoader(
            data_set(pathname, train=False, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif name == 'EMNIST':
        data_loaders['train'] = torch.utils.data.DataLoader(
            data_set(pathname, split='letters', train=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        data_loaders['val'] = torch.utils.data.DataLoader(
            data_set(pathname, split='letters', train=False, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif name == 'SVHN' or name == 'CelebA':
        transform = transforms.Compose([
            transforms.CenterCrop((140, 140)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(), ])
        data_loaders['train'] = torch.utils.data.DataLoader(
            data_set(pathname, split='train', download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        data_loaders['val'] = torch.utils.data.DataLoader(
            data_set(pathname, split='test', download=True, transform=transform),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif name == 'Omniglot':
        data_loaders['train'] = torch.utils.data.DataLoader(
            data_set(pathname, background=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        data_loaders['val'] = torch.utils.data.DataLoader(
            data_set(pathname, background=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        print('Unknown data class!')

    return data_loaders


def loss_function(recon_x, x, rec_type):
    if rec_type == 'BCE':
        # print(recon_x.size())
        # print(x.size())
        reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    elif rec_type == 'MSE':
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
    elif rec_type == 'MAE':
        reconstruction_loss = F.l1_loss(recon_x, x, reduction='sum')
    else:
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum') + F.l1_loss(recon_x, x, reduction='sum')
    return reconstruction_loss

