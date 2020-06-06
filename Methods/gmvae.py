import numpy as np
import torch
from torch import nn
from torch import optim
import Methods.evaluation as evaluation
import Methods.models as method


# regularizer used in proximal-relational autoencoder
class Prior(nn.Module):
    def __init__(self, data_size: list):
        super(Prior, self).__init__()
        # data_size = [num_component, z_dim]
        self.data_size = data_size
        self.number_components = data_size[0]
        self.output_size = data_size[1]
        self.mu = nn.Parameter(torch.randn(data_size), requires_grad=True)
        self.logvar = nn.Parameter(torch.randn(data_size), requires_grad=True)
        self.pi = nn.Parameter(torch.ones(self.number_components) / self.number_components, requires_grad=False)

    def forward(self):
        return self.mu, self.logvar


def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.
    Args:
        x: tensor: (batch, ..., dim): Observation
        m: tensor: (batch, ..., dim): Mean
        v: tensor: (batch, ..., dim): Variance
    Return:
        kl: tensor: (batch1, batch2, ...): log probability of each sample. Note
            that the summation dimension (dim=-1) is not kept
    """
    # print("q_m", m.size())
    # print("q_v", v.size())
    const = -0.5 * x.size(-1) * torch.log(2 * torch.tensor(np.pi))
    # print(const.size())
    log_det = -0.5 * torch.sum(torch.log(v), dim=-1)
    # print("log_det", log_det.size())
    log_exp = -0.5 * torch.sum((x - m) ** 2 / v, dim=-1)
    log_prob = const + log_det + log_exp
    return log_prob


def log_normal_mixture(z, m, v):
    """
    Computes log probability of a uniformly-weighted Gaussian mixture.
    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances
    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    z = z.unsqueeze(1)
    log_probs = log_normal(z, m, v)
    # print("log_probs_mix", log_probs.shape)

    log_prob = log_mean_exp(log_probs, 1)
    # print("log_prob_mix", log_prob.size())

    ################################################################################
    # End of code modification
    ################################################################################
    return log_prob


def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner
    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed
    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner
    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed
    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def KL_divergence_gmm(z_given_x, q_m, q_logv, p_m, p_logv):
    """
    Computes the Evidence Lower Bound, KL and, Reconstruction costs
    Returns:
        kld: tensor: (): ELBO KL divergence to prior
    """
    # Compute the mixture of Gaussian prior
    p_m = p_m.unsqueeze(0)
    p_v = torch.exp(p_logv.unsqueeze(0))
    q_v = torch.exp(q_logv)

    # terms for KL divergence
    log_q_phi = log_normal(z_given_x, q_m, q_v)
    # print("log_q_phi", log_q_phi.size())
    log_p_theta = log_normal_mixture(z_given_x, p_m, p_v)
    # print("log_p_theta", log_p_theta.size())
    kl = log_q_phi - log_p_theta
    # print("kl", kl.size())

    kld = torch.sum(kl)
    return kld


def train(model, prior, train_loader, optimizer, device, epoch, args):
    model.train()
    prior.train()
    train_rec_loss = 0
    train_reg_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, z, mu, logvar = model(data)
        p_mu, p_logv = prior()
        rec_loss = method.loss_function(recon_batch, data, args.loss_type)
        reg_loss = args.gamma * KL_divergence_gmm(z, mu, logvar, p_mu, p_logv)
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
            p_mu, p_logv = prior()
            rec_loss = method.loss_function(recon_batch, data, args.loss_type)
            reg_loss = args.gamma * KL_divergence_gmm(z, mu, logvar, p_mu, p_logv)
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
    optimizer = optim.Adam(list(model.parameters()) + list(prior.parameters()), lr=args.lr, betas=(0.5, 0.999))
    for epoch in range(1, args.epochs + 1):
        train(model, prior, train_loader, optimizer, device, epoch, args)
        test_rec_loss, test_reg_loss, test_loss = test(model, prior, test_loader, device, args)
        loss_list.append([test_rec_loss, test_reg_loss, test_loss])
        if epoch % args.landmark_interval == 0:
            evaluation.interpolation_2d(model, test_loader, device, epoch, args, prefix='gmvae')
            prior.eval()
            model.eval()
            z_p_mean, z_p_logvar = prior()
            print(z_p_mean.size())
            evaluation.sampling(model, device, epoch, args, prior=[z_p_mean, z_p_logvar], prefix='gmvae')
            evaluation.reconstruction(model, test_loader, device, epoch, args, prefix='gmvae')
    return loss_list
