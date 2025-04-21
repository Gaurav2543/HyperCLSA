import argparse
from train_clclsa import train_test_CLCLSA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir',    type=str, default='BRCA')
    parser.add_argument('--view_list',   nargs='+', type=int, default=[1,2,3])
    parser.add_argument('--num_class',   type=int, default=5)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--epochs',      type=int,   default=7500)
    parser.add_argument('--hidden_dims', nargs=2,    type=int, default=[400,200])
    parser.add_argument('--latent_dim',  type=int,   default=128)
    parser.add_argument('--attn_heads',  type=int,   default=4)
    parser.add_argument('--feature_selection_method', '-fsm', type=str, default='boruta',
                    help='one of [rfecv, boruta, lasso, stability, rfe, mrmr, ga]')
    parser.add_argument('--lambda_contrast', type=float, default=0.5)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--use-cross-attention', dest='use_cross_attention', action='store_true', default=True)
    parser.add_argument('--no-cross-attention', dest='use_cross_attention', action='store_false')
    parser.add_argument('--use-hybrid-attention', action='store_true', default=False,
                        help="Use Pathway Attention for view 1")
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
        fs_method=fs,
        use_cross_attention=args.use_cross_attention
    )
