import argparse
import sys
import os
import time
import pathlib
from typing import Tuple, Callable, Union, Optional

import gym
import numpy as np
import matplotlib.pyplot as plt

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class MDP:
    def __init__(self, env: gym.Env):
        self.env = env
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.P = np.zeros(
            (self.nS, self.nA, self.nS)
        )  # Transition probability: (nS x nA x nS) -> [0,1]
        self.R = np.zeros_like(self.P)  # Reward: SxA -> R
        self.terminal_states = np.zeros(
            env.nS, dtype=int
        )  # Terminal states: S -> [0,1]
        for s in range(env.nS):
            for a in range(env.nA):
                for (prob, nxt_s, rew, done) in env.P[s][a]:
                    self.P[s, a, nxt_s] += prob
                    self.R[s, a, nxt_s] += rew * prob
                    if done:
                        self.terminal_states[nxt_s] = 1

        # Placeholders for estimated transition probabilities and reward function
        self.P_hat = None
        self.R_hat = None

        # max episode length for a trial
        self.max_ep_len = None

        self.Q = None
        self.n_episodes = None
        self.alpha = None
        self.gamma = None
        self.eps_lst = None

    def update_params(
        self,
        n_episodes: int,
        learning_rate: float,
        gamma: float,
        max_ep_len: Optional[int] = None,
    ) -> None:
        self.Q = np.zeros((self.nS, self.nA))
        self.alpha = learning_rate
        self.gamma = gamma
        self.n_episodes = n_episodes

        self.eps_lst = []

        # Set max_ep_len or use default
        self.max_ep_len = (
            max_ep_len
            if max_ep_len is not None and isinstance(max_ep_len, int)
            else self.env._max_episode_steps  # noqa
        )

        # Initial exploration probability
        self.explore_prob = 1.0
        # exponential decay
        self.exploration_decay = 0.001

        # minimum of exploration probability
        self.exploration_prob_min = 0.01

    def TestPolicy(
        self,
        policy: Callable,
        trials: int = 100,
        render: bool = False,
        verbose: bool = False,
    ) -> float:
        """
        Test a policy by running it in the given environment.

        :param policy: A policy to run.
        :param env: The environment to run the policy in.
        :param render: Whether to render the environment.

        :returns: success rate over # of trials.
        """
        assert trials > 0 and isinstance(trials, int)

        success = 0
        reward = 0
        for _ in range(trials):
            obs = self.env.reset()
            done = False
            while not done:
                act = policy(obs)
                obs, rew, done, info = self.env.step(act)
                reward += rew
                if render:
                    self.env.render()
                    time.sleep(0.1)
                if done and obs == 15:
                    success += 1
        success_rate = success / trials
        mean_reward = reward / trials
        if verbose:
            print(f"Success rate: {success_rate}")
        return success_rate, mean_reward

    def learnModel(self, n_samples: int = 10 ** 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate transition probabilities p(s'|a, s) and
        reward function r(s, a, s') over n_samples random samples.

        :param n_samples: Number of random samples to use.
        :returns: transition probabilities and reward function.
        """
        assert n_samples > 0 and isinstance(n_samples, int)

        # Dimension of observation space and action space (both discrete)
        P = np.zeros(
            (self.nS, self.nA, self.nS)
        )  # transition probability: S x A x S' -> [0, 1]
        R = np.zeros_like(P)  # reward r(s, a, s')

        obs = self.env.reset()
        done = False
        for _ in range(n_samples):
            # Random action
            act = self.env.action_space.sample()
            nxt_obs, rew, done, _ = self.env.step(act)

            P[obs, act, nxt_obs] += 1
            R[obs, act, nxt_obs] += rew
            obs = nxt_obs

            if done:
                obs = self.env.reset()
        # Normalize transition probabilities -> [0,1]
        p1 = P.copy()  # Don't modify P dircetly
        for s in range(self.nS):
            for a in range(self.nA):
                total_counts = np.sum(P[s, a, :])
                if total_counts != 0:
                    p1[s, a, :] /= total_counts
        # Avoid division by zero error
        R /= np.where(P != 0, P, 1)

        # Store the estimated transition probabilities and reward function
        self.P_hat = p1
        self.R_hat = R
        return p1, R

    def PolicyEval(
        self, V: np.ndarray, policy: np.ndarray, gamma: float, theta: float
    ) -> np.ndarray:
        """
        Policy evaluation
        :param V: value function
        :param policy: a policy
        :param gamma: discount factor
        :param theta: tolerance or termination threshold
        """
        assert 0 < gamma < 1
        assert 0 < theta <= 1e-2, "Threshold should be a small positive number"

        # Using the esitimations of the transition probabilities and reward function
        if self.P_hat is None or self.R_hat is None:
            self.learnModel()

        while True:
            delta = 0.0
            for s in range(self.nS):
                act = policy[s]
                Vs = 0
                for nxt_s in range(self.nS):
                    Vs += self.P_hat[s, act, nxt_s] * (
                        self.R_hat[s, act, nxt_s] + gamma * V[nxt_s]
                    )
                # Calculate delta
                delta = max(delta, abs(Vs - V[s]))
                # Update V
                V[s] = Vs
            if delta < theta:
                break
        return V

    def PolicyImprovement(self, V: np.ndarray, gamma: float) -> np.ndarray:
        """
        Policy improvement
        :param V: value function
        :param gamma: discount factor
        """
        assert 0 < gamma < 1
        # Using the esitimations of the transition probabilities and reward function
        if self.P_hat is None or self.R_hat is None:
            self.learnModel()

        policy = np.zeros(self.nS, dtype=int)
        for s in range(self.nS):
            Q = np.zeros(self.nA)  # Q(s_t, a) we don't need to store all of them
            for a in range(self.nA):
                for nxt_s in range(self.nS):
                    Q[a] += self.P_hat[s, a, nxt_s] * (
                        self.R_hat[s, a, nxt_s] + gamma * V[nxt_s]
                    )
            policy[s] = np.argmax(Q)
        return policy

    def PolicyIteration(
        self, max_iter: int = 50, gamma: float = 0.99, theta: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Policy iteration
        :param policy: a policy
        :param max_iter: maximum number of iterations
        :param gamma: discount factor
        :param theta: tolerance or termination threshold
        """
        assert max_iter > 0 and isinstance(max_iter, int)
        assert 0 < gamma < 1
        assert 0 < theta <= 1e-2, "Theta should be a small positive number"

        # Initialize V(s), \pi(s)
        V = np.zeros(self.nS)
        PI = np.zeros(self.nS, dtype=int)  # since actions are integers
        success_rates = []
        mean_rewards = []

        print(f"\n-------- Policy Iteration --------:")
        for i in range(max_iter):
            PI_old = PI.copy()
            print(f"Iteration {i+1}: ", end="")
            # Policy Evaluation
            V = self.PolicyEval(V, PI, gamma, theta)

            # Policy Improvement
            PI = self.PolicyImprovement(V, gamma)

            PI_fn = lambda s: PI[s]
            success_rate, mean_rew = self.TestPolicy(
                PI_fn, trials=100, render=False, verbose=True
            )
            success_rates.append(success_rate)
            mean_rewards.append(mean_rew)

            if np.all(PI_old == PI):
                print(f"\nPolicy is stable in {i} iterations")
                break

        return PI, V, success_rates, mean_rewards

    def qValue(self, V: np.ndarray, s: int, gamma: float) -> np.ndarray:
        """
        Calculate the q-value of a state
        :param V: value function
        :param s: state
        :param gamma: discount factor
        """
        Q = np.zeros(self.nA)
        for a in range(self.nA):
            for nxt_s in range(self.nS):
                Q[a] += self.P_hat[s, a, nxt_s] * (
                    self.R_hat[s, a, nxt_s] + gamma * V[nxt_s]
                )
        return Q

    def ValueIter(
        self, max_iter: int = 50, gamma: float = 0.99, theta: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        assert max_iter > 0 and isinstance(max_iter, int)
        assert 0 < gamma < 1

        # Initialize V(s), \pi(s)
        V = np.zeros(self.nS)
        PI = np.zeros(self.nS, dtype=int)  # since actions are integers
        success_rates = []
        mean_rewards = []

        print(f"\n-------- Value Iteration --------:")
        for i in range(max_iter):
            print(f"Iteration {i+1}: ", end="")
            delta = 0.0
            for s in range(self.nS):
                v_old = V[s]
                # V(s) = max_a Q(s, a)
                Q = self.qValue(V, s, gamma)  # Q(s_t, a) -> Vector of Q-values
                V[s] = max(Q)

                delta = max(delta, abs(V[s] - v_old))
            if delta < theta:
                break

            PI = self.PolicyImprovement(V, gamma)

            PI_fn = lambda s: PI[s]
            success_rate, mean_rew = self.TestPolicy(
                PI_fn, trials=100, render=False, verbose=True
            )
            success_rates.append(success_rate)
            mean_rewards.append(mean_rew)
        return PI, V, success_rates, mean_rewards

    def _update_epsilon_greedy(self, e: int) -> None:
        """
        Calculate the epsilon-greedy policy
        :param e: current episode
        """
        assert e >= 0 and isinstance(e, int)
        eps = e / self.n_episodes
        self.explore_prob = 1 - eps

    def _update_exponential_decay(self, e: int) -> None:
        """
        Exponential decay
        :param e: current episode
        """
        assert e >= 0 and isinstance(e, int)
        self.explore_prob = max(
            self.exploration_prob_min, np.exp(-self.exploration_decay * e)
        )

    def get_exploration_prob(self, episode: int, strategy: str) -> None:
        """
        Get the exploration probability
        :param e: current episode
        :param strategy: exploration strategy
        """
        if strategy == "epsilon":
            self._update_epsilon_greedy(episode)
        elif strategy == "exponential":
            self._update_exponential_decay(episode)
        else:
            raise NotImplementedError(f"Strategy {strategy} is not implemented")

    def _explore(self) -> int:
        return self.env.action_space.sample()

    def _exploit(self, s: int) -> int:
        return np.argmax(self.Q[s, :])  # Q: S X A

    def get_actions(self, s: int) -> int:
        """
        Epsilon-greedy policy

        :param s: current state
        :param episode: current episode
        """
        if np.random.uniform(0, 1) < self.explore_prob:
            # random action
            act = self._explore()
        else:
            act = self._exploit(s)
        return act

    def update_Q(self, s: int, a: int, r: float, nxt_s: int) -> None:
        """
        In SARSA update
            Q(s, a) <- Q(s, a) + alpha * (R(s, a) + gamma * Q(s', a') - Q(s, a))
        Let Q(s',a') search for the highest possible return:
            max_a Q(s', a')
        Then,
            Q(s, a) <- Q(s, a) + alpha * (R(s, a) + gamma * max_a Q(s', a) - Q(s, a))

        """
        self.Q[s, a] += self.alpha * (
            r + self.gamma * np.max(self.Q[nxt_s, :]) - self.Q[s, a]
        )

    def QLearning(
        self,
        n_episodes: int,
        gamma: float,
        alpha: float,
        strategy: str,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Off-policy TD Learning (Q-Learning)
        """
        assert n_episodes > 0 and isinstance(n_episodes, int)
        assert 0 < gamma < 1
        assert 0 < alpha < 1
        assert isinstance(strategy, str)

        # update the hyperparameters
        self.update_params(n_episodes, alpha, gamma)

        success_rates = []
        print(
            f"\n-------- Q Learning (alpha={alpha}, gamma={gamma}, strategy: {strategy}) --------:"
        )
        for e in range(n_episodes + 1):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            # Training
            self.get_exploration_prob(e, strategy)
            for t in range(self.max_ep_len):
                act = self.get_actions(obs)
                nxt_obs, rew, done, _ = self.env.step(act)
                # update Q-value:
                self.update_Q(obs, act, rew, nxt_obs)
                episode_reward += rew
                if done:
                    break
                obs = nxt_obs
            # Evaluation
            policy_fn = lambda s: np.argmax(self.Q[s, :])
            if e % 100 == 0:
                if verbose:
                    print(f"Episode {e}: ", end="")
                success_rate, _ = self.TestPolicy(
                    policy_fn, trials=100, render=False, verbose=verbose
                )
                success_rates.append(success_rate)

        # Final policy recovery
        policy_fn = lambda s: np.argmax(self.Q[s, :])
        return policy_fn, self.Q, success_rates


def simple_plot(
    arr,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str = "",
    show: bool = True,
    timeout: Optional[int] = None,
) -> None:
    if len(arr) == 2:
        plt.plot(arr[0], arr[1])
    else:
        plt.plot(arr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save_path:
        plt.savefig(save_path)

    if timeout is not None:
        plt.pause(timeout)
        plt.draw()
        plt.waitforbuttonpress(timeout=5)
        plt.close()
    elif show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_id",
        type=str,
        help="Envriment to train on",
        default="FrozenLake-v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", "-r", action="store_true")
    parser.add_argument("--question_numer", "-q", type=int, choices=[3, 4, 5, 6, 7, 8])
    args = parser.parse_args()

    # Current path
    path = pathlib.Path(__file__).parent.resolve()
    plot_dir = path / "plots"

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    # =============================================================================================
    # Create the environment
    """
    The surface is described using a grid like the following:
    SFFF       (S: starting point, safe)
    FHFH       (F: frozen surface, safe)
    FFFH       (H: hole, fall to your doom)
    HFFG       (G: goal, where the frisbee is located)
    """
    env = gym.make(args.env_id)
    env.render()
    ic(env.observation_space)
    ic(env.action_space)

    # Create an MDP from the env as a reference
    mdp = MDP(env)

    # =============================================================================================
    # 1.3. Naive policy
    if args.question_numer == 3 or args.question_numer is None:
        policy = lambda s: (s + 1) % 4
        naive_success_rates = []
        for _ in range(10):
            naive_success_rate, _ = mdp.TestPolicy(policy, render=args.render)
            naive_success_rates.append(naive_success_rate)
        print(f"Average naive_success_rates: {np.mean(naive_success_rates)}")

    # =============================================================================================
    # 1.4. LearnModel()
    if args.question_numer == 4 or args.question_numer is None:
        p1, r1 = mdp.learnModel(n_samples=10 ** 5)
        mse = lambda x, y: np.mean((x - y) ** 2)

        MSE_P, MSE_R = mse(mdp.P, p1), mse(mdp.R, r1)
        ic(MSE_P, MSE_R)

    # =============================================================================================
    # 1.5. Policy iteration
    if args.question_numer == 5 or args.question_numer is None:
        pi_PI, V_pi, success_rates_PI, _ = mdp.PolicyIteration(
            50, theta=sys.float_info.epsilon
        )
        ic(pi_PI)
        ic(V_pi)

        simple_plot(
            success_rates_PI,
            xlabel="Iteration",
            ylabel="Success rate",
            title="Average rate of success of the learned policy (Policy Iteration)",
            timeout=5,
            save_path=plot_dir / "PI.png",
        )

    # =============================================================================================
    # 1.6. Value iteration
    if args.question_numer == 6 or args.question_numer is None:
        pi_VI, V_pi, success_rates_VI, _ = mdp.ValueIter(
            50, theta=sys.float_info.epsilon
        )
        ic(pi_VI)
        ic(V_pi)

        simple_plot(
            success_rates_VI,
            xlabel="Iteration",
            ylabel="Success rate",
            title="Average rate of success of the learned policy (Value Iteration)",
            timeout=5,
            save_path=plot_dir / "VI.png",
        )

    # =============================================================================================
    # 2.1. Q-Learning
    learning_rate = [0.05, 0.1, 0.25, 0.5]
    discount_factor = [0.9, 0.95, 0.99]
    if args.question_numer == 7 or args.question_numer is None:
        # (a)
        for lr in learning_rate:
            try:
                pi_QL, Q_pi, success_rates_QL = mdp.QLearning(
                    n_episodes=5_000,
                    gamma=0.99,
                    alpha=lr,
                    strategy="epsilon",
                    verbose=True,
                )
                simple_plot(
                    success_rates_QL,
                    xlabel="Iteration",
                    ylabel="Success rate",
                    title=(
                        f"Average rate of success of the learned policy (Q Learning), "
                        + r"$\alpha=$"
                        + f"{lr}, "
                        + r"$\gamma=$"
                        + f"{0.99}"
                    ),
                    # save_path=plot_dir / f'QL_a_{lr}_r_{g}.png',
                    show=True,
                )
                print("\n")
            except KeyboardInterrupt:
                plt.close()
                break
        # (b)
        for g in discount_factor:
            try:
                pi_QL, Q_pi, success_rates_QL = mdp.QLearning(
                    n_episodes=5_000,
                    gamma=g,
                    alpha=0.05,
                    strategy="epsilon",
                    verbose=True,
                )
                simple_plot(
                    success_rates_QL,
                    xlabel="Iteration",
                    ylabel="Success rate",
                    title=(
                        f"Average rate of success of the learned policy (Q Learning), "
                        + r"$\alpha=$"
                        + f"{0.05}, "
                        + r"$\gamma=$"
                        + f"{g}"
                    ),
                    # save_path=plot_dir / f'QL_a_{lr}_r_{g}.png',
                    show=True,
                )
                print("\n")
            except KeyboardInterrupt:
                plt.close()
                break

    # =============================================================================================
    # 2.1. Q-Learning (Solve using Q-learning by proposing a different strategy to explore)
    if args.question_numer == 8 or args.question_numer is None:
        for lr in learning_rate:
            try:
                for g in discount_factor:
                    pi_QL, Q_pi, success_rates_QL = mdp.QLearning(
                        n_episodes=5_000,
                        gamma=g,
                        alpha=lr,
                        strategy="exponential",  # "epsilon"
                        verbose=True,
                    )
                    simple_plot(
                        success_rates_QL,
                        xlabel="Iteration",
                        ylabel="Success rate",
                        title=(
                            f"Average rate of success of the learned policy (Q Learning), "
                            + r"$\alpha=$"
                            + f"{lr}, "
                            + r"$\gamma=$"
                            + f"{g}"
                        ),
                        save_path=plot_dir / f"QL_exp_a_{lr}_r_{g}.png",
                        show=True,
                    )
                    print("\n")
            except KeyboardInterrupt:
                plt.close()
                break
