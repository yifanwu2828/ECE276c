import argparse
import pathlib
import time

import gym

import numpy as np
import torch as th

import matplotlib.pyplot as plt

import pytorch_utils as ptu
from ddpg import DDPG

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    path = pathlib.Path(__file__).parent.resolve()

    # TODO: change to rand_init = True
    env = gym.make(f"modified_gym_env:ReacherPyBulletEnv-v1", rand_init=True)

    ptu.init_gpu(use_gpu=args.cuda)  # using cpu

    if args.train:
        ddpg = DDPG(
            env,
            buffer_size=int(1e5),
            gamma=0.99,
            tau=0.005,
            pi_lr=1e-4,
            qf_lr=1e-3,
            seed=0,
            device=ptu.device,
            save_dir=str(path),
            hidden_sizes=(400, 300),
            activation="relu_inplace",
        )
        
        ic(ddpg.actor)
        ic(ddpg.critic)
        
        loss_actor, loss_critic, train_rew = ddpg.train(
            num_steps=int(2e5),
            batch_size=64,
            start_steps=1_000,
            steps_per_epoch=1_500,
            update_every=100,
            update_after=1000,
            act_noise_scale=0.1,
        )
        
        plt.plot(loss_actor)
        plt.title("policy loss")
        plt.show()
        plt.plot(loss_critic)
        plt.title("Q function loss")
        plt.show()
        plt.plot(train_rew)
        plt.title("Episode Return")
        plt.xlabel("Epoch")
        plt.show()
        

    else:
        # load
        print("Loading ...")
        policy = th.load(path / "actor.pth")

        # setup = 0
        env.render(mode="human")

        mean_ep_ret = []
        mean_ep_len = []
        for i in range(10):
            obs = env.reset()
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done:
                # if setup == 0:
                #     time.sleep(8)
                #     setup +=1
                act = policy.get_action(ptu.to_torch(obs), noise_scale=0.0)

                obs, rew, done, _ = env.step(act)
                time.sleep(0.1)
                ep_ret += rew
                ep_len += 1

            mean_ep_ret.append(ep_ret)
            mean_ep_len.append(ep_len)

        mean_ep_ret = np.mean(mean_ep_ret)
        mean_ep_len = np.mean(mean_ep_len)

        ic(mean_ep_ret)
        ic(mean_ep_len)
