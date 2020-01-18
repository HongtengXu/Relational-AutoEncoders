import torch
import torch.nn.init as init
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        print(tensor.size())
        return tensor.view(self.size)


class AE_MNIST(nn.Module):
    """Encoder-Decoder architecture for a typical autoencoder."""
    def __init__(self, z_dim=2, nc=1, type='variational'):
        super(AE_MNIST, self).__init__()
        self.type = type
        self.z_dim = z_dim
        self.nc = nc
        self.init_num_filters_ = 16
        self.lrelu_slope_ = 0.2
        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, self.init_num_filters_, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 1, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.AvgPool2d(kernel_size=2, padding=0),
            nn.Conv2d(self.init_num_filters_ * 1, self.init_num_filters_ * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.AvgPool2d(kernel_size=2, padding=0),
            nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            nn.AvgPool2d(kernel_size=2, padding=1),
            View((-1, self.init_num_filters_ * 4 * 4 * 4))         # B, 1024*4*4
        )

        if self.type == 'variational':
            self.fc1 = nn.Linear(self.init_num_filters_ * 4 * 4 * 4, z_dim)    # B, z_dim
            self.fc2 = nn.Linear(self.init_num_filters_ * 4 * 4 * 4, z_dim)    # B, z_dim
        else:
            self.fc = nn.Linear(self.init_num_filters_ * 4 * 4 * 4, z_dim)    # B, z_dim

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, self.init_num_filters_ * 4 * 4 * 4),   # B, 1024*8*8
            View((-1, 4 * self.init_num_filters_, 4, 4)),   # B, 1024,  8,  8
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=0),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.init_num_filters_ * 4, self.init_num_filters_ * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 2, self.init_num_filters_ * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(self.lrelu_slope_, inplace=True),

            nn.Conv2d(self.init_num_filters_ * 2, nc, kernel_size=3, padding=1)
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        z = self._encode(x)
        if self.type == 'variational':
            mu = self.fc1(z)
            logvar = self.fc2(z)
            z = self.reparameterize(mu, logvar)
            x_recon = self._decode(z)
            return x_recon, z, mu, logvar
        else:
            z = self.fc(z)
            x_recon = self._decode(z)
            return x_recon, z

    def _encode(self, x):
        z = self._encode(x)
        if self.type == 'variational':
            mu = self.fc1(z)
            logvar = self.fc2(z)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            z = self.fc(z)
            return z

    def _decode(self, z):
        return self.decoder(z)


class AE_CelebA(nn.Module):
    """Encoder-Decoder architecture for a typical autoencoder."""
    def __init__(self, z_dim=10, nc=3, type='variational'):
        super(AE_CelebA, self).__init__()
        self.type = type
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

        if self.type == 'variational':
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
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        z = self._encode(x)
        if self.type == 'variational':
            mu = self.fc1(z)
            logvar = self.fc2(z)
            z = self.reparameterize(mu, logvar)
            x_recon = self._decode(z)
            return x_recon, z, mu, logvar
        else:
            z = self.fc(z)
            x_recon = self._decode(z)
            return x_recon, z

    def _encode(self, x):
        z = self._encode(x)
        if self.type == 'variational':
            mu = self.fc1(z)
            logvar = self.fc2(z)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            z = self.fc(z)
            return z

    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


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


def load_datasets(args, pathname, src: bool):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if src is True:
        name = args.source_data
    else:
        name = args.target_data

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
        reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    elif rec_type == 'MSE':
        reconstruction_loss = F.mse_loss(recon_x, x, size_average=False)
    elif rec_type == 'MAE':
        reconstruction_loss = F.l1_loss(recon_x, x, size_average=False)
    else:
        reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') + \
                              F.mse_loss(recon_x, x, size_average=False) + F.l1_loss(recon_x, x, size_average=False)
    return reconstruction_loss
