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

    def forward(self):
        return self.mu, self.logvar


def sampling_gaussian(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps*std


def sampling_gmm(mu, logvar, num_sample):
    std = torch.exp(0.5 * logvar)
    n = int(num_sample / mu.size(0)) + 1
    for i in range(n):
        eps = torch.randn_like(std)
        if i == 0:
            samples = mu + eps * std
        else:
            samples = torch.cat((samples, mu + eps * std), dim=0)
    return samples[:num_sample, :]


def distance_tensor(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C, D] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.abs(x_col - y_row) ** p
    return distance


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance


def sliced_fgw_distance(posterior_samples, prior_samples, num_projections=50, p=2, beta=0.1, device='cpu'):
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = prior_samples.size(1)
    # generate random projections in latent space
    projections = torch.randn(size=(embedding_dim, num_projections)).to(device)
    # calculate projections through the encoded samples
    posterior_projections = posterior_samples.matmul(projections)  # batch size x #projections
    prior_projections = prior_samples.matmul(projections)  # batch size x #projections
    posterior_projections = torch.sort(posterior_projections, dim=0)[0]
    prior_projections1 = torch.sort(prior_projections, dim=0)[0]
    prior_projections2 = torch.sort(prior_projections, dim=0, descending=True)[0]
    posterior_diff = distance_tensor(posterior_projections, posterior_projections, p=p)
    prior_diff1 = distance_tensor(prior_projections1, prior_projections1, p=p)
    prior_diff2 = distance_tensor(prior_projections2, prior_projections2, p=p)
    # print(posterior_projections.size(), prior_projections1.size())
    # print(posterior_diff.size(), prior_diff1.size())
    w1 = torch.sum((posterior_projections - prior_projections1) ** p, dim=0)
    w2 = torch.sum((posterior_projections - prior_projections2) ** p, dim=0)
    # print(w1.size(), torch.sum(w1))
    gw1 = torch.mean(torch.mean((posterior_diff - prior_diff1) ** p, dim=0), dim=0)
    gw2 = torch.mean(torch.mean((posterior_diff - prior_diff2) ** p, dim=0), dim=0)
    # print(gw1.size(), torch.sum(gw1))
    fgw1 = (1 - beta) * w1 + beta * gw1
    fgw2 = (1 - beta) * w2 + beta * gw2
    return torch.sum(torch.min(fgw1, fgw2))


def sliced_gw_distance(posterior_samples, prior_samples, num_projections=50, p=2, device='cpu'):
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = prior_samples.size(1)
    # generate random projections in latent space
    projections = torch.randn(size=(embedding_dim, num_projections)).to(device)
    # calculate projections through the encoded samples
    posterior_projections = posterior_samples.matmul(projections)  # batch size x #projections
    prior_projections = prior_samples.matmul(projections)  # batch size x #projections
    posterior_projections = torch.sort(posterior_projections, dim=0)[0]
    prior_projections1 = torch.sort(prior_projections, dim=0)[0]
    prior_projections2 = torch.sort(prior_projections, dim=0, descending=True)[0]
    posterior_diff = distance_tensor(posterior_projections, posterior_projections, p=p)
    prior_diff1 = distance_tensor(prior_projections1, prior_projections1, p=p)
    prior_diff2 = distance_tensor(prior_projections2, prior_projections2, p=p)

    out1 = torch.sum(torch.sum((posterior_diff - prior_diff1) ** p, dim=0), dim=1)
    out2 = torch.sum(torch.sum((posterior_diff - prior_diff2) ** p, dim=0), dim=1)
    return torch.sum(torch.min(out1, out2))


def sliced_wasserstein_distance(encoded_samples, num_projections=50, p=2, device='cpu'):
    """
    Sliced Wasserstein Distance between encoded samples and drawn distribution samples.
        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')
        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    # print(encoded_samples.size())
    embedding_dim = encoded_samples.size(1)
    distribution_samples = torch.randn(size=encoded_samples.size()).to(device)
    # generate random projections in latent space
    projections = torch.randn(size=(num_projections, embedding_dim)).to(device)
    # print(projections.size())
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.sum()


def train(model, prior, train_loader, optimizer, device, epoch, args):
    model.train()
    prior.train()
    train_rec_loss = 0
    train_reg_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, z = model(data)
        z_mu, z_logvar = prior()
        z_samples = sampling_gmm(z_mu, z_logvar, num_sample=z.size(0))
        rec_loss = method.loss_function(recon_batch, data, args.loss_type)
        reg_loss = args.gamma * sliced_fgw_distance(z, z_samples, beta=args.beta, device=device)
        loss = rec_loss + reg_loss
        loss.backward()
        train_rec_loss += rec_loss.item()
        train_reg_loss += reg_loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Model Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))

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
            recon_batch, z = model(data)
            z_mu, z_logvar = prior()
            z_samples = sampling_gmm(z_mu, z_logvar, num_sample=z.size(0))
            rec_loss = method.loss_function(recon_batch, data, args.loss_type)
            reg_loss = args.gamma * sliced_fgw_distance(z, z_samples, beta=args.beta, device=device)
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
    optimizer = optim.Adam(list(model.parameters()) + list(prior.parameters()), lr=args.lr, betas=(0.9, 0.999))
    for epoch in range(1, args.epochs + 1):
        train(model, prior, train_loader, optimizer, device, epoch, args)
        test_rec_loss, test_reg_loss, test_loss = test(model, prior, test_loader, device, args)
        loss_list.append([test_rec_loss, test_reg_loss, test_loss])
        if epoch % args.landmark_interval == 0:
            evaluation.interpolation_2d(model, test_loader, device, epoch, args)
            prior.eval()
            z_p_mean, z_p_logvar = prior()
            evaluation.sampling(model, device, epoch, args, prior=[z_p_mean, z_p_logvar])
            evaluation.reconstruction(model, test_loader, device, epoch, args)
    return loss_list
