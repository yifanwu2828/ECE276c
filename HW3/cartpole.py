import os
import pathlib
import sys
import argparse

import torch as th

from common.trainer import Trainer
from common.utils import make_unique_timestamp


try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa
    

def CLI():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env_id",
        type=str,
        choices=["InvertedPendulum-v2", "HalfCheetah-v2", "Hopper-v3", "NavEnv-v0"],
        help="Envriment to train on",
    )
    p.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=[
            "ppo",
            "sac",
        ],
        help="RL algo to use",
    )
    p.add_argument("--num_steps", "-n", type=int, default=0.5 * 1e6)
    p.add_argument("--rollout_length", type=int, default=None)
    p.add_argument("--batch_size", "-bs", type=int, default=256)
    # p.add_argument("--buffer_size", type=int, default=1 * 1e6)
    p.add_argument("--log_every_n_updates", "-lg", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=5 * 1e3)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--save_freq", type=int, default=50_000)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", type=int, default=2)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--use_wandb", "-wb", action="store_true")

    args = p.parse_args()

    args.device = "cuda" if args.cuda else "cpu"

    # Enforce type int
    args.num_steps = int(args.num_steps)
    args.batch_size = int(args.batch_size)
    args.log_every_n_updates = int(args.log_every_n_updates)

    # How often (in terms of steps) to output training info.
    args.log_interval = args.batch_size * args.log_every_n_updates

    return args

if __name__ == "__main__":
    args = CLI()

    path = pathlib.Path(__file__).parent.resolve()
    print(path)