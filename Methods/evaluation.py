import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torchvision.utils import save_image


def visualization_tsne(model, test_loader, device, args, prefix, prior=None):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            if args.model_type == 'probabilistic':
                recon_batch, z, mu, logvar = model(data)
            else:
                recon_batch, mu = model(data)

            if i == 0:
                # zs = recon_batch.cpu().numpy()
                zs = mu.cpu().numpy()
                labels = label.cpu().numpy()
            else:
                # zs = np.concatenate((zs, recon_batch.cpu().numpy()), axis=0)
                zs = np.concatenate((zs, mu.cpu().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().numpy()), axis=0)

    num_sample = min([zs.shape[0], 2500])
    zs = zs[:num_sample, :]
    print(zs.shape)
    labels = labels[:num_sample]
    if prior is None:
        x_embedded = TSNE(n_components=2).fit_transform(zs)
    else:
        landmarks = prior[0].cpu().data.numpy()
        x_embedded = TSNE(n_components=2).fit_transform(np.concatenate((zs, landmarks), axis=0))
        landmarks = x_embedded[num_sample:, :]
        x_embedded = x_embedded[:num_sample, :]
    plt.figure(figsize=(5, 5))
    for i in range(10):
        plt.scatter(x_embedded[labels == i, 0], x_embedded[labels == i, 1], s=1, label='{}'.format(i))
    if prior is not None:
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=50, c='k', marker='x', label=r'$\mu_k$')
    plt.legend()
    plt.savefig('{}/{}_tsne_{}_{}.pdf'.format(args.resultpath, prefix, args.model_type, args.source_data))
    plt.close('all')


def visualization_tsne2(model, test_loader, device, args, prefix, prior=None):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            if args.model_type == 'probabilistic':
                recon_batch, z, mu, logvar = model(data)
            else:
                recon_batch, mu = model(data)

            if i == 0:
                # zs = recon_batch.cpu().numpy()
                zs = mu.cpu().numpy()
                # labels = label.cpu().numpy()
            else:
                # zs = np.concatenate((zs, recon_batch.cpu().numpy()), axis=0)
                zs = np.concatenate((zs, mu.cpu().numpy()), axis=0)
                # labels = np.concatenate((labels, label.cpu().numpy()), axis=0)

    num_sample = min([zs.shape[0], 2500])
    zs = zs[:num_sample, :]
    print(zs.shape)
    # labels = labels[:num_sample]
    if prior is None:
        x_embedded = TSNE(n_components=2).fit_transform(zs)
    else:
        landmarks = prior[0].cpu().data.numpy()
        x_embedded = TSNE(n_components=2).fit_transform(np.concatenate((zs, landmarks), axis=0))
        landmarks = x_embedded[num_sample:, :]
        x_embedded = x_embedded[:num_sample, :]
    plt.figure(figsize=(5, 5))
    # for i in range(10):
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=1)
    if prior is not None:
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=50, c='k', marker='x', label=r'$\mu_k$')
    plt.legend()
    plt.savefig('{}/{}_tsne_{}_{}.pdf'.format(args.resultpath, prefix, args.model_type, args.source_data))
    plt.close('all')


def interpolation_2d(model, data_loader, device, epoch, args, prefix, nrow=14):
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            if i == 0:
                data = data.to(device)
                if args.model_type == 'probabilistic':
                    recon_batch, z, mu, logvar = model(data)
                else:
                    recon_batch, mu = model(data)
                mu = mu[:4, :]
            else:
                break

        latents = torch.randn(int(nrow ** 2), mu.size(1)).to(device)
        for i in range(nrow):
            for j in range(nrow):
                x1 = (nrow - 1 - i) / (nrow - 1)
                x2 = i / (nrow - 1)
                y1 = (nrow - 1 - j) / (nrow - 1)
                y2 = j / (nrow - 1)
                n = nrow * i + j
                latents[n, :] = y1 * (x1 * mu[0, :] + x2 * mu[1, :]) + y2 * (x1 * mu[2, :] + x2 * mu[3, :])

        samples = model.decode(latents).cpu()
        s = int(args.x_dim ** 0.5)
        save_image(samples.view(int(nrow ** 2), args.nc, s, s),
                   '{}/{}_interp2d_{}_{}_{}.png'.format(
                       args.resultpath, prefix, args.model_type, args.source_data, epoch), nrow=nrow)


def sampling(model, device, epoch, args, prefix, prior=None, nrow=14):
    model.eval()
    n_samples = int(nrow ** 2)
    with torch.no_grad():
        if prior is None:  # normal prior
            sample = torch.randn(n_samples, args.z_dim)
        else:
            mu = prior[0]
            logvar = prior[1]
            n_components = prior[0].size(0)
            sample = torch.randn(n_samples, args.z_dim)
            for i in range(n_samples):
                idx = int(n_components * np.random.rand())
                std = torch.exp(0.5 * logvar[idx, :])
                eps = torch.randn_like(std)
                # print(idx)
                sample[i, :] = mu[idx, :] + eps * std

        sample = model.decode(sample.to(device)).cpu()
        s = int(args.x_dim ** 0.5)
        pathname = '{}/{}_samples_{}_{}_{}.png'.format(args.resultpath, prefix, args.model_type, args.source_data, epoch)
        print(pathname)
        save_image(sample.view(n_samples, args.nc, s, s), pathname, nrow=nrow)
        print('Done!')


def reconstruction(model, test_loader, device, epoch, args, prefix, nrow=14):
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            if args.model_type == 'probabilistic':
                recon_batch, z, mu, logvar = model(data)
            else:
                recon_batch, mu = model(data)
            if i < int(nrow / 2):
                n = min(data.size(0), nrow)
                if i == 0:
                    comparison = torch.cat([data[:n], recon_batch[:n]])
                else:
                    comparison = torch.cat([comparison, data[:n], recon_batch[:n]])
            else:
                break

        save_image(comparison.cpu(),
                   '{}/{}_recon_{}_{}_{}.png'.format(args.resultpath, prefix, args.model_type, args.source_data, epoch), nrow=nrow)



