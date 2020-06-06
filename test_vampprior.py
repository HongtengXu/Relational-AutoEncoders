import argparse
import pickle
import torch.utils.data
import Methods.vampprior as vampprior
import Methods.evaluation as evaluation
from Methods.models import load_datasets, AE_CelebA, AE_MNIST


parser = argparse.ArgumentParser(description='VampPrior Autoencoder Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--source-data', type=str, default='MNIST',
                    help='data name')
parser.add_argument('--datapath', type=str, default='Data',
                    help='data path')
parser.add_argument('--resultpath', type=str, default='Results',
                    help='result path')
parser.add_argument('--landmark-interval', type=int, default=10,
                    help='interval for recording')
parser.add_argument('--x-dim', type=int, default=784,
                    help='input dimension')
parser.add_argument('--z-dim', type=int, default=8,
                    help='latent dimension')
parser.add_argument('--K', type=int, default=10,
                    help='the number of clusters')
parser.add_argument('--nc', type=int, default=1,
                    help='the number of channels')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='the weight of regularizer')
parser.add_argument('--model-type', type=str, default='probabilistic',
                    help='the type of model')
parser.add_argument('--loss-type', type=str, default='MSE',
                    help='the type of loss')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print(device)

if __name__ == '__main__':
    if args.source_data == 'MNIST':
        args.x_dim = int(28 * 28)
        args.z_dim = 8
        args.nc = 1
        model = AE_MNIST(z_dim=args.z_dim, nc=args.nc, model_type=args.model_type)
        prior = vampprior.Prior(data_size=[args.K, args.nc, 28, 28], device=device)
    else:
        args.x_dim = int(64 * 64)
        args.z_dim = 64
        args.nc = 3
        model = AE_CelebA(z_dim=args.z_dim, nc=args.nc, model_type=args.model_type)
        prior = vampprior.Prior(data_size=[args.K, args.nc, 64, 64], device=device)

    src_loaders = load_datasets(args=args)
    loss = vampprior.train_model(model, prior, src_loaders['train'], src_loaders['val'], device, args)

    # conditional generation
    prior.eval()
    model.eval()
    x = prior()
    _, _, z_p_mean, z_p_logvar = model(x)
    prior_list = [z_p_mean, z_p_logvar]
    for i in range(args.K):
        evaluation.sampling(model, device, i + 1, args, prefix='vampprior',
                            prior=[z_p_mean[i, :].unsqueeze(0), z_p_logvar[i, :].unsqueeze(0)],
                            nrow=4)

    # t-sne visualization
    if args.source_data == 'MNIST':
        evaluation.visualization_tsne(model, src_loaders['val'], device, args, prefix='vampprior', prior=prior_list)
    else:
        evaluation.visualization_tsne2(model, src_loaders['val'], device, args, prefix='vampprior', prior=prior_list)

    # save models and learning results
    model = model.to('cpu')
    prior = prior.to('cpu')
    torch.save(model.state_dict(),
               '{}/vampprior_model_{}_{}.pt'.format(args.resultpath, args.model_type, args.source_data))
    torch.save(prior.state_dict(),
               '{}/vampprior_prior_{}_{}.pt'.format(args.resultpath, args.model_type, args.source_data))
    with open('{}/vampprior_loss_{}_{}.pkl'.format(args.resultpath, args.model_type, args.source_data), 'wb') as f:
        pickle.dump(loss, f)
    print('\n')
