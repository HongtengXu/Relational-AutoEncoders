import time
import torch
from torch import optim
from torchvision.utils import save_image
import Methods.evaluation as evaluation
import Methods.models as method


# regularizer used in WAE-MMD
def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    """Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2*z_dim*z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum


def maximum_mean_discrepancy(z_tilde, device):
    """Calculate maximum mean discrepancy described in the WAE paper.
    Args:
        z_tilde (Tensor): samples from deterministic non-random encoder Q(Z|X).
            2D Tensor(batch_size x dimension).
        device: samples from prior distributions. same shape with z_tilde.
    """
    assert z_tilde.ndimension() == 2
    z = torch.randn(z_tilde.size()).to(device)
    n = z.size(0)
    out = im_kernel_sum(z, z, 1, exclude_diag=True).div(n*(n-1)) + \
        im_kernel_sum(z_tilde, z_tilde, 1, exclude_diag=True).div(n*(n-1)) - \
        im_kernel_sum(z, z_tilde, 1, exclude_diag=False).div(n*n).mul(2)
    return out


def train(model, train_loader, optimizer, device, epoch, args):
    model.train()
    train_rec_loss = 0
    train_reg_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        since = time.time()
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, z = model(data)
        rec_loss = method.loss_function(recon_batch, data, args.loss_type)
        reg_loss = args.gamma * maximum_mean_discrepancy(z, device)
        loss = rec_loss + reg_loss
        loss.backward()
        train_rec_loss += rec_loss.item()
        train_reg_loss += reg_loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
            print('Time = {:.2f}sec'.format(time.time()-since))

    print('====> Epoch: {} Average RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
        epoch, train_rec_loss / len(train_loader.dataset), train_reg_loss / len(train_loader.dataset),
        (train_rec_loss + train_reg_loss) / len(train_loader.dataset)))


def test(model, test_loader, device, args):
    model.eval()
    test_rec_loss = 0
    test_reg_loss = 0
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, z = model(data)
            rec_loss = method.loss_function(recon_batch, data, args.loss_type)
            reg_loss = args.gamma * maximum_mean_discrepancy(z, device)
            test_rec_loss += rec_loss.item()
            test_reg_loss += reg_loss.item()
            test_loss += (rec_loss.item() + reg_loss.item())

    test_rec_loss /= len(test_loader.dataset)
    test_reg_loss /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('====> Test set RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
        test_rec_loss, test_reg_loss, test_loss))
    return test_rec_loss, test_reg_loss, test_loss


def train_model(model, train_loader, test_loader, device, args):
    model = model.to(device)
    loss_list = []
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, device, epoch, args)
        test_rec_loss, test_reg_loss, test_loss = test(model, test_loader, device, args)
        loss_list.append([test_rec_loss, test_reg_loss, test_loss])
        if epoch % args.landmark_interval == 0:
            evaluation.interpolation_2d(model, test_loader, device, epoch, args, prefix='wae')
            evaluation.sampling(model, device, epoch, args, prior=None, prefix='wae')
            evaluation.reconstruction(model, test_loader, device, epoch, args, prefix='wae')
    return loss_list
