import matplotlib.pyplot as plt
import numpy as np
import pickle

dataset = 'MNIST'
models = ['vae', 'gmvae', 'vampprior', 'wae', 'swae']

plt.figure(figsize=(5, 5))
for model in models:
    with open('Results/{}/loss_{}.pkl'.format(model, dataset), 'rb') as f:
        loss = pickle.load(f)
    loss = np.asarray(loss)
    if model == 'vampprior':
        for i in range(loss.shape[0]):
            n = int(np.log10(loss[i, 0])) + 1
            if n > 2:
                loss[i, 0] /= 10 ** (n-2)
    print(model, loss.shape)
    plt.plot(range(1, loss.shape[0]+1), loss[:, 0], label=model)
plt.legend()
plt.xlabel('The number of epochs')
plt.ylabel('Testing reconstruction loss')
plt.savefig('Results/cmp_reconstruction_loss.pdf')
plt.close('all')
