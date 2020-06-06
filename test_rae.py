import argparse
import pickle
import torch.utils.data
import Methods.prae as prae
import Methods.drae as drae
import Methods.evaluation as evaluation
from Methods.models import load_datasets, AE_CelebA, AE_MNIST


parser = argparse.ArgumentParser(description='Examples for Relational Regularized Autoencoders')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='the learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--source-data', type=str, default='MNIST',
                    help='data set, MNIST or CelebA')
parser.add_argument('--datapath', type=str, default='Data',
                    help='data path')
parser.add_argument('--resultpath', type=str, default='Results',
                    help='result path')
parser.add_argument('--landmark-interval', type=int, default=50,
                    help='interval for recording')
parser.add_argument('--x-dim', type=int, default=784,
                    help='input dimension')
parser.add_argument('--z-dim', type=int, default=8,
                    help='latent dimension')
parser.add_argument('--K', type=int, default=10,
                    help='the number of Gaussian components')
parser.add_argument('--nc', type=int, default=1,
                    help='the number of input channels')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='the weight of regularizer')
parser.add_argument('--beta', type=float, default=0.1,
                    help='the weight of relational regularizer')
parser.add_argument('--model-type', type=str, default='deterministic',
                    help='the type of autoencoder')
parser.add_argument('--loss-type', type=str, default='MSE',
                    help='the type of loss')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print(device)

if __name__ == '__main__':
    src_loaders = load_datasets(args=args)
    print(args.source_data, args.model_type)
    if args.source_data == 'MNIST':
        args.x_dim = int(28 * 28)
        args.z_dim = 8
        args.nc = 1
        model = AE_MNIST(z_dim=args.z_dim, nc=args.nc, model_type=args.model_type)
        if args.model_type == 'probabilistic':
            prior = prae.Prior(data_size=[args.K, args.z_dim])
        else:
            prior = drae.Prior(data_size=[args.K, args.z_dim])
    else:
        args.x_dim = int(64 * 64)
        args.z_dim = 64
        args.nc = 3
        model = AE_CelebA(z_dim=args.z_dim, nc=args.nc, model_type=args.model_type)
        if args.model_type == 'probabilistic':
            prior = prae.Prior(data_size=[args.K, args.z_dim])
        else:
            prior = drae.Prior(data_size=[args.K, args.z_dim])

    if args.model_type == 'probabilistic':
        loss = prae.train_model(model, prior, src_loaders['train'], src_loaders['val'], device, args)
    else:
        loss = drae.train_model(model, prior, src_loaders['train'], src_loaders['val'], device, args)

    # conditional generation
    prior.eval()
    model.eval()
    z_p_mean, z_p_logvar = prior()
    prior_list = [z_p_mean, z_p_logvar]
    for i in range(args.K):
        evaluation.sampling(model, device, i + 1, args, prefix='rae',
                            prior=[z_p_mean[i, :].unsqueeze(0), z_p_logvar[i, :].unsqueeze(0)],
                            nrow=4)

    # t-sne visualization
    if args.source_data == 'MNIST':
        evaluation.visualization_tsne(model, src_loaders['val'], device, args, prefix='rae', prior=prior_list)
    else:
        evaluation.visualization_tsne2(model, src_loaders['val'], device, args, prefix='rae', prior=prior_list)

    # save models and learning results
    model = model.to('cpu')
    prior = prior.to('cpu')
    torch.save(model.state_dict(), '{}/rae_model_{}_{}.pt'.format(args.resultpath, args.model_type, args.source_data))
    torch.save(prior.state_dict(), '{}/rae_prior_{}_{}.pt'.format(args.resultpath, args.model_type, args.source_data))
    with open('{}/rae_loss_{}_{}.pkl'.format(args.resultpath, args.model_type, args.source_data), 'wb') as f:
        pickle.dump(loss, f)
    print('\n')
