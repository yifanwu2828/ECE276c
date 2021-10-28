import argparse
import pathlib


import gym


import matplotlib.pyplot as plt

import pytorch_utils as ptu

from algo import reinforce, vpg_with_baseline

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

def CLI():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env_id",
        type=str,
        choices=["CartPole-v1"],
        help="Envriment to train on",
        default="CartPole-v1",
    )

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

    return args


if __name__ == "__main__":
    args = CLI()

    path = pathlib.Path(__file__).parent.resolve()
    print(path)

    ptu.init_gpu(use_gpu=args.cuda)  # use cpu

    env = gym.make(args.env_id)
    env.seed(args.seed)

    ptu.set_random_seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = ptu.MLPCategoricalpolicy(
        obs_dim, act_dim, hidden_sizes=[64, 64], activation="relu_inplace"
    ).to(ptu.device)

    mean_reward_reinforce, train_loss_reinforce = reinforce(
        env,
        policy,
        num_itrs=200,
        batch_size=500,
        gamma=0.99,
        lr=5e-3)
    plot_loss_rew(train_loss_reinforce, mean_reward_reinforce)

    mean_reward_rtg, train_loss_rtg = vpg_with_baseline(
        env,
        policy,
        num_itrs=200,
        batch_size=500,
        gamma=0.99,
        lr=5e-3,
        baseline=False,
    )
    plot_loss_rew(train_loss_rtg, mean_reward_rtg)
    
    
    mean_reward_baseline, train_loss_baseline = vpg_with_baseline(
        env,
        policy,
        num_itrs=200,
        batch_size=500,
        gamma=0.99,
        lr=5e-3,
        baseline=False,
    )
    plot_loss_rew(train_loss_baseline, mean_reward_baseline)
    
    # ===============================================================
    # 600
    mean_reward_baseline_600, train_loss_baseline_600 = vpg_with_baseline(
        env,
        policy,
        num_itrs=200,
        batch_size=600,
        gamma=0.99,
        lr=5e-3,
        baseline=False,
    )
    plot_loss_rew(train_loss_baseline_600, mean_reward_baseline_600)
    
    # 800
    mean_reward_baseline_800, train_loss_baseline_800 = vpg_with_baseline(
        env,
        policy,
        num_itrs=200,
        batch_size=800,
        gamma=0.99,
        lr=5e-3,
        baseline=False,
    )
    plot_loss_rew(train_loss_baseline_800, mean_reward_baseline_800)
    
    # 1000
    mean_reward_baseline_1000, train_loss_baseline_1000 = vpg_with_baseline(
        env,
        policy,
        num_itrs=200,
        batch_size=1000,
        gamma=0.99,
        lr=5e-3,
        baseline=False,
    )
    plot_loss_rew(train_loss_baseline_1000, mean_reward_baseline_1000)