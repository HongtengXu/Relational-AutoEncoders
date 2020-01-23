import argparse
import Methods.gmvae as gmvae
import Methods.rae as rae
import Methods.rvae as rvae
import Methods.vampprior as vampprior
from Methods.models import AE_MNIST
import torch
import Methods.evaluation as evaluation

parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
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
parser.add_argument('--nc', type=int, default=1,
                    help='the number of channels')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='the weight of regularizer')
parser.add_argument('--beta', type=float, default=0.1,
                    help='the weight of regularizer')
parser.add_argument('--model-type', type=str, default='deterministic',
                    help='the type of model')
parser.add_argument('--loss-type', type=str, default='MSE',
                    help='the type of loss')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print(device)

if __name__ == '__main__':
    args.x_dim = int(28 * 28)
    args.z_dim = 8
    args.nc = 1
    args.loss_type = 'MSE'
    args.landmark_interval = 5
    for modelname in ['vampprior', 'gmvae', 'rae', 'rvae']:
        if modelname == 'gmvae':
            args.resultpath = 'Results/gmvae2_clusters'
            args.model_type = 'probabilistic'
            prior = gmvae.Prior(data_size=[10, args.z_dim])
            model = AE_MNIST(z_dim=args.z_dim, nc=args.nc, model_type=args.model_type)
            model.load_state_dict(torch.load('Results/gmvae2/model_MNIST.pt'), strict=False)
            prior.load_state_dict(torch.load('Results/gmvae2/prior_MNIST.pt'), strict=False)
        elif modelname == 'rae':
            args.resultpath = 'Results/rae2_clusters'
            args.model_type = 'deterministic'
            prior = rae.Prior(data_size=[10, args.z_dim])
            model = AE_MNIST(z_dim=args.z_dim, nc=args.nc, model_type=args.model_type)
            model.load_state_dict(torch.load('Results/rae2/model_MNIST.pt'), strict=False)
            prior.load_state_dict(torch.load('Results/rae2/prior_MNIST.pt'), strict=False)
        elif modelname == 'rvae':
            args.resultpath = 'Results/rvae_clusters'
            args.model_type = 'probabilistic'
            prior = rvae.Prior(data_size=[10, args.z_dim])
            model = AE_MNIST(z_dim=args.z_dim, nc=args.nc, model_type=args.model_type)
            model.load_state_dict(torch.load('Results/rvae/model_MNIST.pt'), strict=False)
            prior.load_state_dict(torch.load('Results/rvae/prior_MNIST.pt'), strict=False)
        else:
            args.resultpath = 'Results/vampprior2_clusters'
            args.model_type = 'probabilistic'
            prior = vampprior.Prior(data_size=[10, args.nc, 28, 28], device=device)
            model = AE_MNIST(z_dim=args.z_dim, nc=args.nc, model_type=args.model_type)
            model.load_state_dict(torch.load('Results/vampprior2/model_MNIST.pt'), strict=False)
            prior.load_state_dict(torch.load('Results/vampprior2/prior_MNIST.pt'), strict=False)

        model = model.to('cpu')
        prior = prior.to('cpu')
        prior.eval()
        model.eval()
        if modelname == 'rae' or modelname == 'gmvae' or modelname == 'rvae':
            z_p_mean, z_p_logvar = prior()
        else:
            x = prior()
            _, _, z_p_mean, z_p_logvar = model(x)

        for i in range(10):
            evaluation.sampling(model, device, i + 1, args,
                                prior=[z_p_mean[i, :].unsqueeze(0), z_p_logvar[i, :].unsqueeze(0)],
                                nrow=7)


