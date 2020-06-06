import torch
from torch import optim
import Methods.evaluation as evaluation
import Methods.models as method


# regularizer used in VAE
def KL_divergence(mu, logvar):
    """
    The KL divergence between the posterior p(z|x) and the prior p(z)
    Both the prior and the posterior are Gaussian
    :param mu:
    :param logvar:
    :return:
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def train(model, train_loader, optimizer, device, epoch, args):
    model.train()
    train_rec_loss = 0
    train_reg_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, _, mu, logvar = model(data)
        rec_loss = method.loss_function(recon_batch, data, args.loss_type)
        reg_loss = args.gamma * KL_divergence(mu, logvar)
        loss = rec_loss + reg_loss
        loss.backward()
        train_rec_loss += rec_loss.item()
        train_reg_loss += reg_loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))

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
            recon_batch, _, mu, logvar = model(data)
            rec_loss = method.loss_function(recon_batch, data, args.loss_type)
            reg_loss = args.gamma * KL_divergence(mu, logvar)
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
            evaluation.interpolation_2d(model, test_loader, device, epoch, args, prefix='vae')
            evaluation.sampling(model, device, epoch, args, prior=None, prefix='vae')
            evaluation.reconstruction(model, test_loader, device, epoch, args, prefix='vae')
    return loss_list
