"""
Training Script for Meta-Learning Models

Supports

- ProtoNet
- MAML
- Reptile

Can be run directly (uses config.py values) or called
programmatically with parameter overrides — used by
run_experiments.py to sweep model x k_shot x seed grids.

Dataset selection (CICIOT / TONIOT) is controlled by
cfg.DATASET in config.py.
"""

import argparse

import torch
import torch.optim as optim

from src.config import config as cfg

from src.data.dataset import CICIoTDataset

from src.data.meta_dataloader import (
    get_episode_dataloaders,
    get_zero_day_dataloader,
)

from src.models.meta_factory import get_meta_model

from src.training.meta_trainer import MetaTrainer

from src.experiments.experiment_manager import ExperimentManager

from src.evaluation.meta_evaluator import MetaEvaluator

from src.meta.algorithms.protonet_algorithm import ProtoNetAlgorithm

from src.meta.algorithms.maml_algorithm import MAMLAlgorithm

from src.meta.algorithms.reptile_algorithm import ReptileAlgorithm

from src.visualization.visualizer import Visualizer

from src.utils.seed import set_seed


def main(

    model_name=None,

    k_shot=None,

    seed=None,

    epochs=None,

    episodes=None,

    experiment_suffix=None,

):

    # ------------------------------------------------------
    # Resolve parameters (explicit args override config.py)
    # ------------------------------------------------------

    model_name = model_name or cfg.MODEL_NAME

    k_shot = k_shot if k_shot is not None else cfg.K_SHOT

    seed = seed if seed is not None else cfg.SEED

    epochs = epochs if epochs is not None else cfg.EPOCHS

    episodes = episodes if episodes is not None else cfg.EPISODES

    # ------------------------------------------------------
    # Resolve dataset-dependent settings          # <-- CHANGED
    # ------------------------------------------------------

    if cfg.DATASET == "TONIOT":                   # <-- CHANGED

        data_dir = "toniot"                       # <-- CHANGED

        zero_day_classes = cfg.TONIOT_ZERO_DAY_CLASSES   # <-- CHANGED

    else:                                         # <-- CHANGED

        data_dir = None                           # <-- CHANGED

        zero_day_classes = cfg.ZERO_DAY_CLASSES   # <-- CHANGED

    set_seed(seed)

    print("=" * 80)
    print("META LEARNING TRAINING")
    print(f"Dataset: {cfg.DATASET} | Model: {model_name} "     # <-- CHANGED
          f"| K-Shot: {k_shot} | Seed: {seed}")                # <-- CHANGED
    print("=" * 80)

    train_dataset = CICIoTDataset(

        feature_file="X_train.csv",

        label_file="y_train.csv",

        data_dir=data_dir,                        # <-- CHANGED

    )

    val_dataset = CICIoTDataset(

        feature_file="X_validation.csv",

        label_file="y_validation.csv",

        data_dir=data_dir,                        # <-- CHANGED

    )

    test_dataset = CICIoTDataset(

        feature_file="X_test.csv",

        label_file="y_test.csv",

        data_dir=data_dir,                        # <-- CHANGED

    )

    # ------------------------------------------------------
    # Episode DataLoaders
    # ------------------------------------------------------

    train_loader, val_loader, meta_train_classes = get_episode_dataloaders(

        train_dataset=train_dataset,

        val_dataset=val_dataset,

        n_way=cfg.N_WAY,

        k_shot=k_shot,

        query_size=cfg.QUERY_SIZE,

        episodes=episodes,

        zero_day_classes=zero_day_classes,        # <-- CHANGED

    )

    zero_day_loader = None

    if zero_day_classes:                          # <-- CHANGED

        if len(zero_day_classes) < cfg.N_WAY:     # <-- CHANGED

            raise ValueError(

                f"zero_day_classes has {len(zero_day_classes)} classes "   # <-- CHANGED
                f"but N_WAY={cfg.N_WAY}. Hold out at least N_WAY classes "
                f"so zero-day episodes match the training task size."

            )

        zero_day_loader = get_zero_day_dataloader(

            test_dataset=test_dataset,

            n_way=cfg.N_WAY,

            k_shot=k_shot,

            query_size=cfg.QUERY_SIZE,

            episodes=cfg.ZERO_DAY_EPISODES,

            zero_day_classes=zero_day_classes,    # <-- CHANGED

        )

    # ------------------------------------------------------
    # Experiment Manager
    # ------------------------------------------------------

    experiment_name = f"{cfg.DATASET}_{model_name}"   # <-- CHANGED

    if experiment_suffix:

        experiment_name = f"{cfg.DATASET}_{model_name}_{experiment_suffix}"   # <-- CHANGED

    experiment = ExperimentManager(

        experiment_name

    )

    experiment.create_readme()

    experiment.save_config({

        "dataset": cfg.DATASET,                   # <-- CHANGED

        "model": model_name,

        "epochs": epochs,

        "learning_rate": cfg.LEARNING_RATE,

        "optimizer": "Adam",

        "loss": "PrototypicalLoss" if model_name == "PROTONET" else "CrossEntropyLoss",

        "n_way": cfg.N_WAY,

        "k_shot": k_shot,

        "query_size": cfg.QUERY_SIZE,

        "episodes": episodes,

        "seed": seed,

        "embedding_dim": 128,

        "input_features": train_dataset.X.shape[1],

        "device": cfg.DEVICE,

        "zero_day_classes": list(zero_day_classes),          # <-- CHANGED

        "meta_train_classes": meta_train_classes.tolist(),

    })

    input_dim = train_dataset.X.shape[1]

    model = get_meta_model(

        model_name,

        input_dim,

    )

    # ------------------------------------------------------
    # Optimizer (must exist BEFORE the algorithm)
    # ------------------------------------------------------

    optimizer = optim.Adam(

        model.parameters(),

        lr=cfg.REPTILE_META_LR if model_name == "REPTILE" else cfg.LEARNING_RATE,

    )

    # ------------------------------------------------------
    # Algorithm
    # ------------------------------------------------------

    if model_name == "PROTONET":

        algorithm = ProtoNetAlgorithm()

    elif model_name == "MAML":

        algorithm = MAMLAlgorithm(

            meta_optimizer=optimizer,

            inner_lr=cfg.INNER_LR,

            inner_steps=cfg.INNER_STEPS,

            first_order=True,

            device=cfg.DEVICE,

        )

    elif model_name == "REPTILE":

        algorithm = ReptileAlgorithm(

            meta_optimizer=optimizer,

            inner_lr=cfg.REPTILE_INNER_LR,

            inner_steps=cfg.REPTILE_INNER_STEPS,

            device=cfg.DEVICE,

        )

    else:

        raise ValueError(

            f"Unsupported Meta Model: {model_name}"

        )

    print(model)

    # ------------------------------------------------------
    # Trainer
    # ------------------------------------------------------

    trainer = MetaTrainer(

        model=model,

        algorithm=algorithm,

        train_episodes=train_loader,

        val_episodes=val_loader,

        optimizer=optimizer,

        device=cfg.DEVICE,

        experiment=experiment,

        epochs=epochs,

        checkpoint_name=f"{model_name.lower()}.pt",

    )

    history = trainer.train()

    # ------------------------------------------------------
    # Training Curve Plots
    # ------------------------------------------------------

    visualizer = Visualizer(experiment.plots_dir)

    visualizer.plot_loss(history)

    visualizer.plot_accuracy(history)

    visualizer.plot_metrics(history)

    # ------------------------------------------------------
    # Evaluate Best Model — SEEN CLASSES
    # ------------------------------------------------------

    evaluator = MetaEvaluator(

        model=model,

        algorithm=algorithm,

        episode_loader=val_loader,

        device=cfg.DEVICE,

        checkpoint_path=experiment.checkpoint_dir /
        f"{model_name.lower()}.pt",

        experiment=experiment,

    )

    metrics = evaluator.evaluate()

    # ------------------------------------------------------
    # Evaluate Best Model — ZERO-DAY CLASSES
    # ------------------------------------------------------

    zero_day_metrics = None

    if zero_day_loader is not None:

        zero_day_evaluator = MetaEvaluator(

            model=model,

            algorithm=algorithm,

            episode_loader=zero_day_loader,

            device=cfg.DEVICE,

            checkpoint_path=experiment.checkpoint_dir /
            f"{model_name.lower()}.pt",

            experiment=experiment,

            tag="zero_day",

        )

        zero_day_metrics = zero_day_evaluator.evaluate()

    print()

    print("Meta Training Completed Successfully.")

    experiment.summary()

    print("Meta Training Completed.")

    return {

        "experiment_name": experiment.experiment_name,

        "history": history,

        "seen_metrics": metrics,

        "zero_day_metrics": zero_day_metrics,

    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=None)

    parser.add_argument("--k-shot", type=int, default=None)

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=None)

    parser.add_argument("--episodes", type=int, default=None)

    args = parser.parse_args()

    main(

        model_name=args.model.upper() if args.model else None,

        k_shot=args.k_shot,

        seed=args.seed,

        epochs=args.epochs,

        episodes=args.episodes,

    )