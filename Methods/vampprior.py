import math
import numpy as np
import torch
from torch import nn
from torch import optim
import Methods.evaluation as evaluation
import Methods.models as method


# regularizer used in VampPrior
class Prior(nn.Module):
    def __init__(self, data_size: list, device):
        super(Prior, self).__init__()
        # data_size = [num_component, num_channels, num_dim, ...]
        self.data_size = data_size
        self.num_components = data_size[0]
        self.output_size = int(np.prod(data_size) / data_size[0])
        self.nonlinearity = nn.Sigmoid()  # nn.Hardtanh(min_val=0.0, max_val=1.0)
        # self.linear = nn.Linear(self.num_components, self.output_size, bias=False)
        # # self.linear.weight.data.normal_(0.05, 0.01)
        # self.idle_input = torch.eye(self.num_components, self.num_components, requires_grad=False).to(device)
        self.basis = nn.Parameter(torch.randn(data_size), requires_grad=True)

    def forward(self):
        # h = self.linear(self.idle_input)
        # return self.nonlinearity(h).reshape(self.data_size)
        return self.nonlinearity(self.basis)


def log_normal_diag(x, mean, log_var, dim=None):
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    return torch.sum(log_normal, dim)


def log_normal_standard(x, dim=None):
    log_normal = -0.5 * torch.pow(x, 2)
    return torch.sum(log_normal, dim)


def log_gmm(z, vae_model, prior_model):
    x = prior_model()
    _, _, z_p_mean, z_p_logvar = vae_model(x)
    # expand z
    z_expand = z.unsqueeze(1)
    means = z_p_mean.unsqueeze(0)
    logvars = z_p_logvar.unsqueeze(0)
    a = log_normal_diag(z_expand, means, logvars, dim=2) - math.log(prior_model.num_components)  # MB x C
    a_max, _ = torch.max(a, 1)  # MB x 1
    # calculte log-sum-exp
    log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1

    return log_prior


def KL_divergence_gmm(z_q, z_q_mean, z_q_logvar, vae_model, prior_model):
    log_p_z = log_gmm(z_q, vae_model, prior_model)
    log_q_z = log_normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
    return -(log_p_z - log_q_z).sum()


def train(model, prior, train_loader, optimizer, device, epoch, args):
    model.train()
    prior.train()
    train_rec_loss = 0
    train_reg_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, z, mu, logvar = model(data)
        rec_loss = method.loss_function(0.95 * recon_batch, data, args.loss_type)
        reg_loss = args.gamma * KL_divergence_gmm(z, mu, logvar, model, prior)
        loss = rec_loss + reg_loss
        loss.backward()
        train_rec_loss += rec_loss.item()
        train_reg_loss += reg_loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                rec_loss.item() / len(data),
                reg_loss.item() / len(data),
                loss.item() / len(data)))

    print('====> Epoch: {} Average RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
        epoch, train_rec_loss / len(train_loader.dataset), train_reg_loss / len(train_loader.dataset),
        (train_rec_loss + train_reg_loss) / len(train_loader.dataset)))


def test(model, prior, test_loader, device, args):
    model.eval()
    prior.eval()
    test_rec_loss = 0
    test_reg_loss = 0
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, z, mu, logvar = model(data)
            rec_loss = method.loss_function(0.95 * recon_batch, data, args.loss_type)
            reg_loss = args.gamma * KL_divergence_gmm(z, mu, logvar, model, prior)
            test_rec_loss += rec_loss.item()
            test_reg_loss += reg_loss.item()
            test_loss += (rec_loss.item() + reg_loss.item())

    test_rec_loss /= len(test_loader.dataset)
    test_reg_loss /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('====> Test set RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
        test_rec_loss, test_reg_loss, test_loss))
    return test_rec_loss, test_reg_loss, test_loss


def train_model(model, prior, train_loader, test_loader, device, args):
    model = model.to(device)
    prior = prior.to(device)
    loss_list = []
    optimizer = optim.Adam(list(model.parameters()) + list(prior.parameters()), lr=1e-4)
    for epoch in range(1, args.epochs + 1):
        train(model, prior, train_loader, optimizer, device, epoch, args)
        test_rec_loss, test_reg_loss, test_loss = test(model, prior, test_loader, device, args)
        loss_list.append([test_rec_loss, test_reg_loss, test_loss])
        if epoch % args.landmark_interval == 0:
            evaluation.interpolation_2d(model, test_loader, device, epoch, args, prefix='vampprior')
            prior.eval()
            model.eval()
            x = prior()
            _, _, z_p_mean, z_p_logvar = model(x)
            print(z_p_mean.size())
            evaluation.sampling(model, device, epoch, args, prior=[z_p_mean, z_p_logvar], prefix='vampprior')
            evaluation.reconstruction(model, test_loader, device, epoch, args, prefix='vampprior')
    return loss_list
