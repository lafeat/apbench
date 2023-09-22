import argparse


def options():
    parser = argparse.ArgumentParser(
        description="Construct poisoned training data for the given network and dataset"
    )
    parser.add_argument(
        "--type", default="tap", type=str
    )
    parser.add_argument(
        "--net", default="ResNet18", type=lambda s: [str(item) for item in s.split(",")]
    )
    parser.add_argument(
        "--dataset",
        default="c10",
        type=str,
        choices=[
            "c10",
            "c100",
        ],
    )
    parser.add_argument(
        "--recipe",
        default="untargeted",
        type=str,
        choices=["grad_explosion", "tensorclog", "untargeted", "targeted"],
    )
    parser.add_argument(
        "--threatmodel",
        default="single-class",
        type=str,
        choices=["single-class", "third-party", "random-subset"],
    )

    # Reproducibility management:
    parser.add_argument(
        "--poisonkey",
        default=3,
        type=str,
        help="Initialize poison setup with this key.",
    )  # Also takes a triplet 0-3-1
    parser.add_argument(
        "--modelkey", default=1, type=int, help="Initialize the model with this key."
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Disable CUDNN non-determinism."
    )

    # Poison properties / controlling the strength of the attack:
    parser.add_argument("--eps", default=8.0, type=float)
    parser.add_argument(
        "--budget",
        default=1.0,
        type=float,
        help="Fraction of training data that is poisoned",
    )

    # Files and folders
    parser.add_argument(
        "--name",
        default="",
        type=str,
        help="Name tag for the result table and possibly for export folders.",
    )
    parser.add_argument("--poison_path", default="./dataset/tap_poisons", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--resume_idx", default=None, type=int)

    # Poison brewing:
    parser.add_argument("--attackoptim", default="PGD", type=str)
    parser.add_argument("--attackiter", default=250, type=int)
    parser.add_argument("--init", default="randn", type=str)  # randn / rand
    parser.add_argument("--tau", default=0.05, type=float)
    parser.add_argument(
        "--scheduling", action="store_false", help="Disable step size decay."
    )
    parser.add_argument(
        "--restarts", default=8, type=int, help="How often to restart the attack."
    )
    parser.add_argument(
        "--poison_partition",
        default=None,
        type=int,
        help="How many poisons to craft at a time",
    )

    parser.add_argument(
        "--pbatch", default=512, type=int, help="Poison batch size during optimization"
    )
    parser.add_argument(
        "--pshuffle",
        action="store_true",
        help="Shuffle poison batch during optimization",
    )
    parser.add_argument(
        "--paugment",
        action="store_false",
        help="Do not augment poison batch during optimization",
    )
    parser.add_argument(
        "--data_aug",
        type=str,
        default="default",
        help="Mode of diff. data augmentation.",
    )

    # Poisoning algorithm changes
    parser.add_argument(
        "--full_data",
        action="store_true",
        help="Use full train data (instead of just the poison images)",
    )
    parser.add_argument(
        "--adversarial", default=0, type=float, help="Adversarial PGD for poisoning."
    )
    parser.add_argument(
        "--ensemble",
        default=1,
        type=int,
        help="Ensemble of networks to brew the poison on",
    )
    parser.add_argument(
        "--stagger",
        action="store_true",
        help="Stagger the network ensemble if it exists",
    )
    parser.add_argument(
        "--step", action="store_true", help="Optimize the model for one epoch."
    )
    parser.add_argument(
        "--max_epoch",
        default=None,
        type=int,
        help="Train only up to this epoch before poisoning.",
    )

    parser.add_argument("--loss", default="similarity", type=str)

    parser.add_argument("--centreg", default=0, type=float)
    parser.add_argument("--normreg", default=0, type=float)
    parser.add_argument("--repel", default=0, type=float)

    parser.add_argument("--nadapt", default=2, type=int, help="Meta unrolling steps")
    parser.add_argument(
        "--clean_grad",
        action="store_true",
        help="Compute the first-order poison gradient.",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained models from torchvision, if possible [only valid for ImageNet].",
    )
    parser.add_argument(
        "--optimization", default="conservative", type=str, help="Optimization Strategy"
    )

    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument(
        "--noaugment",
        action="store_true",
        help="Do not use data augmentation during training.",
    )
    parser.add_argument(
        "--gradient_noise",
        default=None,
        type=float,
        help="Add custom gradient noise during training.",
    )
    parser.add_argument(
        "--gradient_clip",
        default=None,
        type=float,
        help="Add custom gradient clip during training.",
    )
    parser.add_argument(
        "--independent_brewing",
        action="store_true",
        help="Brew each poison independently.",
    )

    parser.add_argument("--lmdb_path", default=None, type=str)
    parser.add_argument(
        "--cache_dataset", action="store_true", help="Cache the entire thing :>"
    )

    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument(
        "--save",
        default="poison_dataset",
        help="Export poisons into a given format. Options are full/limited/automl/numpy.",
    )

    parser.add_argument(
        "--local_rank",
        default=None,
        type=int,
        help="Distributed rank. This is an INTERNAL ARGUMENT! "
        "Only the launch utility should set this argument!",
    )

    return parser
