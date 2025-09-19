import optuna, os, json, torch
from functools import partial
import wandb
from train import train_test_CLCLSA
from utils import logger, set_seed

CACHE_DIR = "selected_features"  # central FS cache


def objective(trial):
    # -------- suggest hyper‑params ----------
    graph_method = trial.suggest_categorical("graph_method",
                                            ["knn", "radius", "mutual_knn"])
    k_neigs   = trial.suggest_int("k_neigs", 3, 6)
    radius_eps = trial.suggest_float("radius_eps", 0.5, 2.0, log=True)

    # choose latent dimension first
    latent_dim = trial.suggest_categorical("latent_dim", [64, 128, 256])

    # derive a head list that divides latent_dim
    possible_heads = [1, 2, 3, 4, 6]
    valid_heads    = [h for h in possible_heads if latent_dim % h == 0]
    attn_heads     = trial.suggest_categorical("attn_heads", valid_heads)

    hidden_1  = trial.suggest_categorical("hidden1", [256, 400, 512])
    hidden_2  = trial.suggest_categorical("hidden2", [128, 200, 256])
    
    lr          = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    wd          = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    lambda_con  = trial.suggest_float("lambda_contrast", 0.25, 1.0)

    fs_method   = trial.suggest_categorical("fs_method", ["shap"])
    if fs_method == "boruta":
        max_iter = trial.suggest_int("boruta_max_iter", 30, 75, step=15)
        fs_kwargs = {"max_iter": max_iter}
    elif fs_method == "rfe":  # rfe
        k_feats  = trial.suggest_categorical("rfe_k", [300, 500, 1000])
        step     = trial.suggest_float("rfe_step", 0.05, 0.2, step=0.05)
        fs_kwargs = {"k": k_feats, "step": step}
    elif fs_method == "shap": # shap
        k_feats  = trial.suggest_categorical("rfe_k", [300, 500, 1000])
        fs_kwargs = {"k": k_feats}

    run_name = f"trial-{trial.number}_{graph_method}_k{k_neigs}_ld{latent_dim}_h{attn_heads}_fs{fs_method}"
    wandb.init(project="HyperCLSA", config=trial.params, reinit=True, name=run_name)

    # -------- run 5‑fold CV training ----------
    set_seed(42)  
    metrics = train_test_CLCLSA(
        data_folder="BRCA",
        view_list=[1,2,3],
        num_class=5,
        lr=lr, epochs=5000,
        hidden_dims=[hidden_1, hidden_2], latent_dim=latent_dim,
        attn_heads=attn_heads, lambda_contrast=lambda_con,
        graph_method=graph_method, k_neigs=k_neigs, radius_eps=radius_eps,
        fs_method=fs_method, fs_kwargs=fs_kwargs,
        n_splits_cv=5
    )

    mean_f1 = metrics["mean_f1_macro"]   
    trial.report(mean_f1, step=0)
    wandb.log({"mean_f1": mean_f1})
    wandb.finish()
    return mean_f1

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="HyperCLSA‑sweep",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=20, timeout=None)

    # -------- save best hyper‑params --------
    best_params = study.best_trial.params
    with open("best_params.json", "w") as fp:
        json.dump(best_params, fp, indent=2)

    logger.info(f"Best F1‑macro: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")