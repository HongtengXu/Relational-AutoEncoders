import argparse
import torch.utils.data
from Methods.models import load_datasets, AE_CelebA, AE_MNIST


parser = argparse.ArgumentParser(description='HAE Example')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


pathname = 'Data'
for src in ['MNIST', 'CelebA']:
    # print(src)
    if src == 'MNIST':
        model = AE_MNIST(z_dim=2, nc=1)
    else:
        model = AE_CelebA(z_dim=10, nc=3)

    args.source_data = src
    src_loaders = load_datasets(args=args)
    for i, (data, _) in enumerate(src_loaders['train']):
        data = data.to(device)
        print(i, data.size())
        if i == 0:
            x_recon, z, _, _ = model(data)
            print(x_recon.size())
        else:
            break




