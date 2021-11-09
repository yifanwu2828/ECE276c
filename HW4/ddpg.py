import os
from copy import deepcopy
from itertools import zip_longest
from math import tau
from typing import Tuple, Dict, Any, Iterable, Union, Optional

import numpy as np
from numpy.core.fromnumeric import mean
import torch as th
from torch import nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

import pytorch_utils as ptu
from buffer import ReplayBuffer


def zip_strict(*iterables: Iterable) -> Iterable:
    """
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.
    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    # ! Slow
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def soft_update(
    target: nn.Module,
    source: nn.Module,
    tau: float,
    one: th.Tensor,
    safe_zip=False,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``
    target parameters are slowly updated towards the main parameters.
    :param target: Target network
    :param source: Source network
    :param tau: the soft update coefficient controls the interpolation:
        ``tau=1`` corresponds to copying the parameters to the target ones
        whereas nothing happens when ``tau=0``.
    :param one: dummy variable to equals to th.ones(1, device=device)
        Since it's a constant should pre-define it on proper device.
    :param safe_zip: if true, will raise error
        if source and target have different length of parameters.
    See https://github.com/DLR-RM/stable-baselines3/issues/93
    """
    with th.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        if safe_zip:
            # ! This is slow.
            for t, s in zip_strict(target.parameters(), source.parameters()):
                t.data.mul_(1.0 - tau)
                t.data.addcmul_(s.data, one, value=tau)
        else:
            # * Fast but safty not gurantee.
            # * should check if source and target have the same parameters outside.
            for t, s in zip(target.parameters(), source.parameters()):
                t.data.mul_(1.0 - tau)
                t.data.addcmul_(s.data, one, value=tau)


def one_gradient_step(
    loss: th.Tensor,
    opt: th.optim.Optimizer,
    net: nn.Module,
    max_grad_norm: Optional[float] = None,
) -> None:
    """Take one gradient step with grad clipping.(AMP support)"""
    # Unscale_update.
    loss.backward()
    if max_grad_norm is not None:
        clip_grad_norm_(net.parameters(), max_norm=max_grad_norm)
    opt.step()


class DDPG:
    def __init__(
        self,
        env,
        buffer_size,
        gamma,
        tau,
        pi_lr,
        qf_lr,
        seed,
        device,
        save_dir,
        hidden_sizes=(64, 64),
        activation="relu",
    ):  
        self.save_dir = save_dir
        self.device = device
        self.seed = seed
        ptu.set_random_seed(seed)

        self.env = env
        self.env.seed = seed

        self.test_env = deepcopy(env)
        self.test_env.seed = seed

        # ! only for this modified reacher env. we need to subtract obs_dim by 1
        self.obs_dim = env.observation_space.shape[0] - 1
        self.act_dim = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0]

        self.gamma = gamma
        self.tau = tau

        self.buffer = ReplayBuffer(
            self.obs_dim,
            self.act_dim,
            self.seed,
            self.device,
            max_size=buffer_size,
        )

        # Deterministic actor
        self.actor = ptu.MLPDeterministicPolicy(
            self.obs_dim,
            self.act_dim,
            hidden_sizes,
            activation,
            self.act_limit,
        ).to(self.device)

        # Q function Critic
        self.critic = ptu.MLPQFunction(
            self.obs_dim,
            self.act_dim,
            hidden_sizes,
            activation,
        ).to(self.device)

        # Set target param equal to main param should be same as deep copy.
        # TODO: should include actor_target as well.
        self.critic_target = deepcopy(self.critic).to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=pi_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=qf_lr)

        # a constant for quick soft update
        self.one = th.ones(1, device=self.device)

    def train(
        self,
        num_steps: int,
        batch_size: int,
        start_steps: int,
        steps_per_epoch: int,
        update_after=1000,
        update_every=50,
        act_noise_scale: float = 0.1,
        max_grad_norm: Optional[float] = None,
    ):

        obs = self.env.reset()
        ep_ret, ep_len = 0.0, 0

        for t in tqdm(range(num_steps), dynamic_ncols=True):
            # Random exploration step
            if t < start_steps:
                act = self.env.action_space.sample()
            else:
                obs_th = th.as_tensor(obs, dtype=th.float32).to(self.device)
                act = self.actor.get_action(obs_th, act_noise_scale)

            next_obs, rew, done, _ = self.env.step(act)
            ep_ret += rew
            ep_len += 1

            # Done mask removes the time limit constrain of some env to keep makorvian.
            # Agent keeps alive should not be assigned to done by env's time limit.
            done = float(done) if ep_len < self.env._max_episode_steps else 0
            self.buffer.store(obs, act, rew, done, next_obs,)

            # update obs
            obs = next_obs
            
            # update
            if t >= update_after and t % update_every == 0:
                for _ in range(update_every):
                    batch = self.buffer.sample(batch_size)
                    self.update(data=batch)

            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch
                self.evaluation(epoch, num_eval_episodes=10)

    @th.no_grad()
    def evaluation(self, epoch, num_eval_episodes: int):
        """
        Evaluate the performance of the deterministic version of the agent.
        :param num_episodes: Number of episodes to evaluate for.
        :return: Mean reward and std of rewards for the last 100 episodes.
        """
        valid_returns, valid_ep_lens = [], []
        for t in range(num_eval_episodes):
            obs = self.test_env.reset()
            ep_ret, ep_len = 0.0, 0
            done = False

            while not done:
                obs_th = th.as_tensor(obs, dtype=th.float32).to(self.device)
                # Deterministic action
                act = self.actor.get_action(obs_th, noise_scale=0.0)
                obs, rew, done, _ = self.test_env.step(act)
                ep_ret += rew
                ep_len += 1
            valid_returns.append(ep_ret)
            valid_ep_lens.append(ep_len)

        mean_rew, std_rew = np.mean(valid_returns), np.std(valid_returns)
        mean_ep_len = np.mean(valid_ep_lens)
        print(
            ptu.colorize(
            f"Epoch: {epoch}\t"
            f"Mean Reward: {mean_rew: .2f} +/- {std_rew: .2f}\t"
            f"with ep_len {mean_ep_len: .2f}", color="white"
            )
        )
        if mean_rew > -5 and mean_ep_len < 30:
            self.save(path=os.path.join(self.save_dir, "actor.pth")) 
    
    def save(self, path):
        print("Saving ...")
        th.save(self.actor, path)

    def loss_q(self, data):
        obs, act, rew, next_obs, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["next_obs"],
            data["done"],
        )

        q = self.critic(obs, act)

        with th.no_grad():
            q_pi_target = self.critic_target(next_obs, self.actor(next_obs))
            backup = rew + self.gamma * (1 - done) * q_pi_target

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()
        return loss_q

    def loss_pi(self, data):
        obs = data["obs"]
        # Q(s,a) = Q(sm \mu(s))
        q_pi = self.critic(obs, self.actor(obs))
        return -q_pi.mean()

    def update(self, data):
        # First run one gradient descent step for Q.
        
        self.critic_optimizer.zero_grad()
        loss_q = self.loss_q(data)
        loss_q.backward()
        self.critic_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for param in self.critic.parameters():
            param.requires_grad = False

        # Next run one gradient descent step for pi.
        self.actor_optimizer.zero_grad()
        loss_pi = self.loss_pi(data)
        loss_pi.backward()
        self.actor_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for param in self.critic.parameters():
            param.requires_grad = True

        # Finally, update target networks by polyak averaging.
        # * here tau = (1 - polyak)
        soft_update(self.critic_target, self.critic, self.tau, self.one)

    



    

