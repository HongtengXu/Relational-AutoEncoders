import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.utils import save_image


def visualization_pca(model, test_loader, device, args):
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
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
    x_embedded = PCA(n_components=2).fit_transform(zs)
    for i in range(10):
        plt.scatter(x_embedded[labels == i, 0], x_embedded[labels == i, 1], label='{}'.format(i))
    plt.legend()
    plt.savefig('{}/pca_{}.pdf'.format(args.resultpath, args.datapath))
    plt.close('all')


def visualization_tsne(model, test_loader, device, args):
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
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
    x_embedded = TSNE(n_components=2).fit_transform(zs)
    for i in range(10):
        plt.scatter(x_embedded[labels == i, 0], x_embedded[labels == i, 1], label='{}'.format(i))
    plt.legend()
    plt.savefig('{}/tsne_{}.pdf'.format(args.resultpath, args.datapath))
    plt.close('all')


def visualization_tsne_trajectory(model, test_loader, device, args):
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            if i == 0:
                zs = mu.cpu().numpy()
                labels = label.cpu().numpy()
            else:
                zs = np.concatenate((zs, mu.cpu().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().numpy()), axis=0)

        num_sample = min([zs.shape[0], 5000])
        zs = zs[:num_sample, :]
        labels = labels[:num_sample]

        for i, (data, label) in enumerate(test_loader):
            if i == 0:
                data = data.to(device)
                _, mu, logvar = model(data)
                mu = mu[:4, :]
                ls = label[:4].cpu().numpy()
            else:
                break

        latents = torch.randn(60, mu.size(1)).to(device)
        pairs = [[mu[0, :], mu[1, :]],
                 [mu[0, :], mu[2, :]],
                 [mu[0, :], mu[3, :]],
                 [mu[1, :], mu[2, :]],
                 [mu[1, :], mu[3, :]],
                 [mu[2, :], mu[3, :]]]
        trajs = ['{}->{}'.format(ls[0], ls[1]),
                 '{}->{}'.format(ls[0], ls[2]),
                 '{}->{}'.format(ls[0], ls[3]),
                 '{}->{}'.format(ls[1], ls[2]),
                 '{}->{}'.format(ls[1], ls[3]),
                 '{}->{}'.format(ls[2], ls[3])]
        colors = ['red', 'blue', 'yellow', 'green', 'black', 'cyan']
        for i in range(len(pairs)):
            src = pairs[i][0]
            dst = pairs[i][1]
            for j in range(10):
                y1 = (9 - j)/9
                y2 = j/9
                n = 10 * i + j
                latents[n, :] = y1 * src + y2 * dst

        samples = model.decode(latents)
        _, mu_traj, logvar = model(samples)
        mu_traj = mu_traj.cpu().numpy()

        latents = np.concatenate((zs, mu_traj), axis=0)
        if latents.shape[1] > 2:
            z_embedded = TSNE(n_components=2).fit_transform(latents)
        else:
            z_embedded = latents
        x_embedded = z_embedded[:-60, :]
        y_embedded = z_embedded[-60:, :]

        for i in range(10):
            plt.scatter(x_embedded[labels == i, 0], x_embedded[labels == i, 1], s=0.7, label='{}'.format(i))

        for i in range(len(pairs)):
            points = y_embedded[10*i:10*(i+1), :]
            plt.plot(points[:, 0], points[:, 1], 'o-', c=colors[i], label=trajs[i])
            # for j in range(points.shape[0] - 1):
            #     plt.plot([points[j, 0], points[j+1, 0]], [points[j, 1], points[j+1, 1]], 'x-', c=colors[i])

        plt.legend()
    plt.savefig('{}/traj_tsne_{}.pdf'.format(args.resultpath, args.datapath))
    plt.close('all')


def visualization_pca_trajectory(model, test_loader, device, args):
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            if i == 0:
                zs = mu.cpu().numpy()
                labels = label.cpu().numpy()
            else:
                zs = np.concatenate((zs, mu.cpu().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().numpy()), axis=0)

        num_sample = min([zs.shape[0], 5000])
        zs = zs[:num_sample, :]
        labels = labels[:num_sample]

        for i, (data, label) in enumerate(test_loader):
            if i == 0:
                data = data.to(device)
                _, mu, logvar = model(data)
                mu = mu[:4, :]
                ls = label[:4].cpu().numpy()
            else:
                break

        latents = torch.randn(60, mu.size(1)).to(device)
        pairs = [[mu[0, :], mu[1, :]],
                 [mu[0, :], mu[2, :]],
                 [mu[0, :], mu[3, :]],
                 [mu[1, :], mu[2, :]],
                 [mu[1, :], mu[3, :]],
                 [mu[2, :], mu[3, :]]]
        trajs = ['{}->{}'.format(ls[0], ls[1]),
                 '{}->{}'.format(ls[0], ls[2]),
                 '{}->{}'.format(ls[0], ls[3]),
                 '{}->{}'.format(ls[1], ls[2]),
                 '{}->{}'.format(ls[1], ls[3]),
                 '{}->{}'.format(ls[2], ls[3])]
        colors = ['red', 'blue', 'yellow', 'green', 'black', 'cyan']
        for i in range(len(pairs)):
            src = pairs[i][0]
            dst = pairs[i][1]
            for j in range(10):
                y1 = (9 - j)/9
                y2 = j/9
                n = 10 * i + j
                latents[n, :] = y1 * src + y2 * dst

        samples = model.decode(latents)
        _, mu_traj, logvar = model(samples)
        mu_traj = mu_traj.cpu().numpy()

        latents = np.concatenate((zs, mu_traj), axis=0)
        if latents.shape[1] > 2:
            z_embedded = PCA(n_components=2).fit_transform(latents)
        else:
            z_embedded = latents
        x_embedded = z_embedded[:-60, :]
        y_embedded = z_embedded[-60:, :]

        for i in range(10):
            plt.scatter(x_embedded[labels == i, 0], x_embedded[labels == i, 1], s=0.7, label='{}'.format(i))

        for i in range(len(pairs)):
            points = y_embedded[10*i:10*(i+1), :]
            plt.plot(points[:, 0], points[:, 1], 'o-', c=colors[i], label=trajs[i])
        plt.legend()
    plt.savefig('{}/traj_pca_{}.pdf'.format(args.resultpath, args.datapath))
    plt.close('all')


def interpolation_2d(model, data_loader, device, epoch, args):
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            if i == 0:
                data = data.to(device)
                _, mu, logvar = model(data)
                mu = mu[:4, :]
            else:
                break

        latents = torch.randn(100, mu.size(1)).to(device)
        for i in range(10):
            for j in range(10):
                x1 = (9 - i)/9
                x2 = i/9
                y1 = (9 - j)/9
                y2 = j/9
                n = 10 * i + j
                latents[n, :] = y1 * (x1 * mu[0, :] + x2 * mu[1, :]) + y2 * (x1 * mu[2, :] + x2 * mu[3, :])

        samples = model.decode(latents).cpu()
        s = int(args.x_dim ** 0.5)
        save_image(samples.view(100, args.nc, s, s),
                   '{}/interp2d_{}_{}.png'.format(args.resultpath, args.datapath, epoch), nrow=10)


def sampling(model, device, epoch, args, prior=None):
    model.eval()
    with torch.no_grad():
        if prior is None:  # normal prior
            sample = torch.randn(100, args.z_dim)
        else:
            mu = prior[0]
            logvar = prior[1]
            n_components = prior[0].size(0)
            sample = torch.randn(100, args.z_dim)
            for i in range(100):
                idx = int(n_components * np.random.rand())
                std = torch.exp(0.5 * logvar[idx, :])
                eps = torch.randn_like(std)
                sample[i, :] = mu[idx, :] + eps * std

        sample = model.decode(sample.to(device)).cpu()
        s = int(args.x_dim ** 0.5)
        save_image(sample.view(100, args.nc, s, s),
                   '{}/samples_{}_{}.png'.format(args.resultpath, args.datapath, epoch), nrow=10)


def reconstruction(model, test_loader, device, epoch, args):
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, z, mu, logvar = model(data)
            if i < 5:
                n = min(data.size(0), 10)
                comparison = torch.cat([data[:n], recon_batch[:n]])
                save_image(comparison.cpu(),
                           '{}/recon_{}_{}_{}.png'.format(args.resultpath, args.datapath, epoch, i), nrow=n)
