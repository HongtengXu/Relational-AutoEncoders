import matplotlib.pyplot as plt
import numpy as np
import pickle

dataset = 'MNIST'
models = ['vae', 'gmvae2', 'vampprior2', 'wae', 'swae', 'rvae', 'rae2']
modelnames = ['VAE', 'GMVAE', 'VampPrior', 'WAE', 'SWAE', 'RAE-P', 'RAE-D']
plt.figure(figsize=(5, 5))
for nn in range(len(models)):
    model = models[nn]
    modelname = modelnames[nn]
    with open('Results/{}/loss_{}.pkl'.format(model, dataset), 'rb') as f:
        loss = pickle.load(f)
    loss = np.asarray(loss)
    # if model == 'vampprior2':
    #     for i in range(loss.shape[0]):
    #         n = int(np.log10(loss[i, 0])) + 1
    #         if n > 2:
    #             loss[i, 0] /= 10 ** (n-2)
    print(model, loss.shape)
    if model == 'rae2':
        plt.semilogy(range(1, loss.shape[0] + 1), loss[:, 0], c='black', label=modelname, linewidth=2)
    elif model == 'rvae':
        plt.semilogy(range(1, loss.shape[0] + 1), loss[:, 0], 'k--', label=modelname, linewidth=2)
    else:
        plt.semilogy(range(1, loss.shape[0] + 1), loss[:, 0], label=modelname)
plt.legend()
plt.xlabel('The number of epochs')
plt.ylabel('Testing reconstruction loss')
plt.savefig('Results/cmp_reconstruction_loss_MNIST.pdf')
plt.close('all')


plt.figure(figsize=(5, 5))
for nn in range(len(models)):
    model = models[nn]
    modelname = modelnames[nn]
    with open('Results/{}/loss_{}.pkl'.format(model, dataset), 'rb') as f:
        loss = pickle.load(f)
    loss = np.asarray(loss)
    print(model, loss.shape)
    if model == 'rae2':
        plt.plot(range(26, loss.shape[0] + 1), loss[25:, 0], c='black', label=modelname, linewidth=2)
    elif model == 'rvae':
        plt.plot(range(26, loss.shape[0] + 1), loss[25:, 0], 'k--', label=modelname, linewidth=2)
    else:
        plt.plot(range(26, loss.shape[0] + 1), loss[25:, 0], label=modelname)
plt.legend()
plt.xlabel('The number of epochs')
plt.ylabel('Testing reconstruction loss')
plt.savefig('Results/cmp_reconstruction_loss_MNIST_enlarged.pdf')
plt.close('all')
