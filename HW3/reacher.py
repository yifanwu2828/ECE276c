import argparse
import pathlib
import time

import gym
from tqdm import tqdm

import numpy as np
import torch as th
import matplotlib.pyplot as plt

from algo import vpg_with_baseline
import pytorch_utils as ptu



try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa
    

def plot_loss_rew(loss, rew):
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(loss, '-r')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training loss at each iteration')
    ax1.grid(True)
    ax2.plot(rew)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mean Reward')
    ax2.set_title('Average rewards at each Iteration')
    ax2.grid(True)
    plt.show()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env_id",
        type=str,
        choices=["ReacherPyBulletEnv−v1"],
        help="Envriment to train on",
        default="ReacherPyBulletEnv−v1",
    )
    p.add_argument("--train", "-t", action="store_true")
    p.add_argument("--num_itrs", "-itr", type=int, default=200)
    p.add_argument("--batch_size", "-bs", type=int, default=500)
    p.add_argument("--eval_interval", type=int, default=5 * 1e3)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument(
        "--cuda",
        action="store_true",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", type=int, default=2)

    args = p.parse_args()

    args.device = "cuda" if args.cuda else "cpu"

    # Enforce type int
    args.num_steps = int(args.num_itrs)
    args.batch_size = int(args.batch_size)
    ptu.init_gpu(use_gpu=False)  # use cpu

    path = pathlib.Path(__file__).parent.resolve()
    print(path)
    
    env = gym.make(f"modified_gym_env:ReacherPyBulletEnv-v1", rand_init=True)
    env.seed(args.seed)

    ptu.set_random_seed(args.seed)

    obs_dim = env.observation_space.shape[0] - 1  #? Why?
    act_dim = env.action_space.shape[0]
    
    if args.train:
        policy = ptu.MLPDiagGaussianpolicy(
            obs_dim,
            act_dim,
            hidden_sizes=[128, 128],
            activation="relu",
            
        ).to(ptu.device)
        ic(policy)
        
        mean_reward_rtg, train_loss_rtg = vpg_with_baseline(
            env,
            policy,
            num_itrs=200,
            batch_size=2000,
            gamma=0.9,
            lr=1e-2,
            baseline=True,
            baseline_type="average",
            verbose=True
        )
        plot_loss_rew(train_loss_rtg, mean_reward_rtg)
        
        # Save
        print("Saving ...")
        th.save(policy, path /"actor.pth")
    
    
    # load
    print("Loading ...")
    policy = th.load(path / "actor.pth")
    
    
    setup = 0
    env.render(mode="human")
    
    for i in range(5):
        obs = env.reset()
        done = False
        
        while not done:
            if setup == 0:
                time.sleep(8)
                setup +=1
            act, _ = policy.get_action(ptu.to_torch(obs), deterministic=True)
            act = ptu.to_numpy(act)
            
            obs, rew, done, _ = env.step(act)
            time.sleep(0.1)
        