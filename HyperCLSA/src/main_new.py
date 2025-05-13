import wandb
from train import train_test_CLCLSA
from utils import logger, set_seed

if __name__ == "__main__":
    # ─── Manually set your best parameters ───
    best_params = {
        "graph_method": "radius",                
        "k_neigs": 6,                         
        "radius_eps": 0.696573230951862,                    
        "latent_dim": 64,                    
        "attn_heads": 1,                      # ← must divide latent_dim
        "hidden1": 400,                       
        "hidden2": 256,                       
        "lr": 0.0018717267107673911,                          
        "weight_decay": 0.000877872150373703,                 
        "lambda_contrast": 0.29547396996824615,               
        "fs_method": "boruta",                # "rfe" or "boruta"
        "boruta_max_iter": 60,                # if using boruta
        "rfe_k": 500,                       # if using rfe
        "rfe_step": 0.1,
    }

    # ─── Resolve fs_kwargs based on fs_method ───
    fs_method = best_params["fs_method"]
    if fs_method == "boruta":
        fs_kwargs = {"max_iter": best_params["boruta_max_iter"]}
    elif fs_method == "rfe":
        fs_kwargs = {"k": best_params["rfe_k"], "step": best_params["rfe_step"]}
    else:
        raise ValueError("Invalid fs_method")

    # ─── Log with wandb ───
    wandb.init(project="HyperCLSA", config=best_params)

    # ─── Set seed and run ───
    set_seed(42)
    metrics = train_test_CLCLSA(
        data_folder="BRCA",
        # data_folder="ROSMAP",
        view_list=[1, 2, 3],
        # view_list=[2,3],
        # num_class=2,
        num_class=5,
        lr=best_params["lr"],
        epochs=5000,
        hidden_dims=[best_params["hidden1"], best_params["hidden2"]],
        latent_dim=best_params["latent_dim"],
        attn_heads=best_params["attn_heads"],
        lambda_contrast=best_params["lambda_contrast"],
        graph_method=best_params["graph_method"],
        k_neigs=best_params["k_neigs"],
        radius_eps=best_params["radius_eps"],
        fs_method=fs_method,
        fs_kwargs=fs_kwargs,
        n_splits_cv=5
    )

    wandb.log(metrics)
    wandb.finish()
    logger.info(f"Final evaluation: macro-F1 = {metrics['mean_f1_macro']:.4f}")
