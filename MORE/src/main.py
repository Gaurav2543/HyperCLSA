import argparse
from utils import set_seed
import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        help='more or hypertmo')
    parser.add_argument('--file_dir', '-fd', type=str, required=True,
                        help='The dataset file folder.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='The dataset file folder.')
    parser.add_argument('--seed', '-s', type=int, default=20,
                        help='Random seed, default: 20.')
    parser.add_argument('--num_epoch', '-ne', type=int, default=20000,
                        help='Training epochs, default: 20000.')
    parser.add_argument('--lr_e_pretrain', '-lrep', type=float, default=0.001,
                        help='Classifier learning rate, default: 0.001.')
    parser.add_argument('--lr_e', '-lr', type=float, default=5e-4,
                        help='Learning rate, default: 0.0005.')
    parser.add_argument('--lr_c', '-lrc', type=float, default=0.001,
                        help='Classifier learning rate, default: 0.001.')
    parser.add_argument('--dim_he_list', '-dh', nargs='+', type=int, default=[400, 200, 200],
                        help='Hidden layer dimensions of HGCN.')
    parser.add_argument('--num_class', '-nc', type=int, required=True,
                        help='Number of classes.')
    parser.add_argument('--k_neigs', '-kn', type=int, default=4,
                        help='Number of vertices in hyperedge.')
    parser.add_argument('--view_list', '-vl', nargs='+', type=int, default=[1,2,3],
                        help='List of views.')
    parser.add_argument('--num_epoch_pretrain', '-nep', type=int, default=500,
                        help='Number of pretrain epochs, default: 500.')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Make command-line arguments available to train.py
    train.args = args

    # Start training/testing using unified parameters
    train.train_test(args.file_dir, args.view_list, args.num_class,
                     args.lr_e_pretrain, args.lr_e, args.lr_c,
                     args.num_epoch_pretrain, args.num_epoch, args.k_neigs)