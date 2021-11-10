import numpy as np

import torch as th
from torch import optim
from tqdm import tqdm
import pytorch_utils as ptu



def reinforce(env, policy, num_itrs, batch_size, gamma, lr, verbose=False):

    average_reward_lst = []  # store step over episode
    train_loss_lst = []
    # define optimizer
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # train
    for itr in tqdm(range(num_itrs)):
        n_traj = 0
        batch_loss_sum = 0

        batch_rewards = []

        # Rollout buffer
        traj_rewards = []
        traj_log_prob_sum = 0

        # reset environment
        obs = env.reset()
        done = False

        for batch_i in range(batch_size):
            action, log_pi = policy.get_action(ptu.from_numpy(obs))
            next_obs, reward, done, _ = env.step(ptu.to_numpy(action))
            # env.render()

            # store data
            traj_rewards.append(reward)
            traj_log_prob_sum += log_pi

            # update obs
            obs = next_obs

            # trajectory done or batch finished
            if done or (batch_i == batch_size - 1):

                # gamma **[0, 1, ... T]
                discounts = np.power(gamma, np.arange(len(traj_rewards)))

                # sum of G(t), no grad
                traj_discounted_return = (
                    ptu.to_torch(traj_rewards) * ptu.to_torch(discounts)
                ).sum()

                batch_loss_sum += traj_discounted_return * traj_log_prob_sum

                # reset obs
                obs = env.reset()

                batch_rewards.append(np.sum(traj_rewards))
                n_traj += 1
                traj_rewards = []
                traj_log_prob_sum = 0

        # batch finish
        mean_batch_reward = np.mean(batch_rewards)
        average_reward_lst.append(mean_batch_reward)
        loss = -batch_loss_sum / n_traj
        train_loss_lst.append(ptu.to_numpy(loss))
        
        if itr % 10 == 0 and verbose:
        
            print(
            ptu.colorize(
                f"Episode [{itr + 1}/{num_itrs}] loss: {loss.item():.2f}, mean reward: {mean_batch_reward:.2f}, n_traj: {n_traj}",
                color="blue",
            )
        )

        # update policy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return average_reward_lst, train_loss_lst


def vpg_with_baseline(
    env, policy, num_itrs, batch_size, gamma, lr, baseline=False, baseline_type='average', verbose=False
):
    """
    Policy gradient with baseline
    """
    average_reward_list = []  
    train_loss_lst = []

    
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # train
    for itr in tqdm(range(num_itrs)):
        n_traj = 0
        batch_loss_sum = 0
        batch_rewards = []
        batch_log_prob = []
        batch_reward_to_go = []

        traj_rewards = []
        traj_log_prob_list = []

        # each bach contain multiply data of trajs        
        
        # reset environment
        obs = env.reset()
        done = False

        for batch_i in range(batch_size):
            action, log_pi = policy.get_action(ptu.from_numpy(obs))
            next_obs, reward, done, _ = env.step(ptu.to_numpy(action))

            # store data
            traj_rewards.append(reward)
            traj_log_prob_list.append(log_pi)

            # update obs
            obs = next_obs

            # trajectory done or batch finished
            if done or (batch_i == batch_size - 1):
                reward_to_go = [
                    np.sum(
                        [
                            gamma ** (t_prime - t) * traj_rewards[t_prime]
                            for t_prime in range(t, len(traj_rewards))
                        ]
                    )
                    for t in range(1, len(traj_rewards) + 1)
                ]

                # store whole traj data
                batch_log_prob.extend(traj_log_prob_list)
                batch_reward_to_go.extend(reward_to_go)

                # reset state
                batch_rewards.append(np.sum(traj_rewards))
                n_traj += 1

                traj_rewards = []
                traj_log_prob_list = []

                obs = env.reset()

        batch_rtg = ptu.to_torch(batch_reward_to_go) # no grad
        batch_log_prob_th = th.stack(batch_log_prob)  # stack from list keep the correct grad
        
        if baseline:
            # rtg = rtg - b
            if baseline_type == "average":
                baseline = batch_rtg.mean()
            batch_rtg -= baseline
        
        # \sum {grad_log_pi * (rtg-b)}
        batch_loss_sum = th.sum(batch_log_prob_th * batch_rtg)
        
        mean_batch_reward = np.mean(batch_rewards)
        average_reward_list.append(mean_batch_reward)
        # taking average
        loss = -batch_loss_sum / n_traj
        train_loss_lst.append(ptu.to_numpy(loss))

        if itr % 10 == 0 and verbose:
            print(
                ptu.colorize(
                    f"Episode [{itr + 1}/{num_itrs}] loss: {loss.item():.2f}, mean reward: {mean_batch_reward:.2f}, n_traj: {n_traj}",
                    color="blue",
                )
            )
        # update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return average_reward_list, train_loss_lst