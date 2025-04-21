import argparse
from train_clclsa import train_test_CLCLSA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir',    type=str, default='BRCA')
    parser.add_argument('--view_list',   nargs='+', type=int, default=[1,2,3])
    parser.add_argument('--num_class',   type=int, default=5)
    parser.add_argument('--lr',          type=float, default=5e-4)
    parser.add_argument('--epochs',      type=int,   default=5000)
    parser.add_argument('--hidden_dims', nargs=2,    type=int, default=[400,200])
    parser.add_argument('--latent_dim',  type=int,   default=128)
    parser.add_argument('--attn_heads',  type=int,   default=4)
    parser.add_argument('--feature_selection_method', '-fsm', type=str, default='boruta',
                    help='one of [rfecv, boruta, lasso, stability, rfe, mrmr, ga]')
    parser.add_argument('--lambda_contrast', type=float, default=0.5)
    parser.add_argument('--seed',        type=int,   default=42)
    args = parser.parse_args()
    
    fs = args.feature_selection_method

    train_test_CLCLSA(
        args.file_dir, args.view_list, args.num_class,
        lr=args.lr, epochs=args.epochs,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        attn_heads=args.attn_heads,
        lambda_contrast=args.lambda_contrast,
        seed=args.seed,
        fs_method=fs
    )
