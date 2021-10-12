import argparse
import sys
import time
import pathlib
from typing import Tuple, Callable, Union, Optional

import numpy as np
import matplotlib.pyplot as plt

import gym

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class MDP:
    def __init__(self, env: gym.Env):
        self.env = env
        self.nS = env.nS
        self.nA = env.nA
        self.P = np.zeros((env.nS, env.nA, env.nS))  # Transition probability: (nS x nA x nS) -> [0,1]
        self.R = np.zeros((env.nS, env.nA, env.nS))  # Reward: SxA -> R
        for s in range(env.nS):
            for a in range(env.nA):
                for (prob, nxt_s, rew, done) in env.P[s][a]:
                    self.P[s,a,nxt_s] += prob
                    self.R[s,a, nxt_s] += rew * prob
        self.P_hat = None
        self.R_hat = None
        
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
        P = np.zeros((self.nS, self.nA, self.nS))  # transition probability: S x A x S' -> [0, 1]
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
        R /= np.where(P!=0, P, 1)
        
        # Store the estimated transition probabilities and reward function 
        self.P_hat = p1
        self.R_hat = R
        return p1, R
    
    def PolicyEval(
        self,
        V: np.ndarray,
        policy: np.ndarray,
        gamma: float,
        theta: float
    ):
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
                   Vs += self.P_hat[s, act, nxt_s] * (self.R_hat[s, act, nxt_s] + gamma * V[nxt_s])
                # Calculate delta
                delta = max(delta, abs(Vs - V[s]))
                # Update V
                V[s] = Vs       
            if delta < theta:
                break
        return V
    
    def PolicyImprovement(self, V: np.ndarray, gamma: float):
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
            Q = np.zeros(self.nA) # Q(s_t, a) we don't need to store all of them
            for a in range(self.nA):
                for nxt_s in range(self.nS):
                    Q[a] += self.P_hat[s, a, nxt_s] * (self.R_hat[s, a, nxt_s] + gamma * V[nxt_s])
            policy[s] = np.argmax(Q)
        return policy
    
    def PolicyIteration(
        self,
        max_iter: int = 50,
        gamma: float = 0.99,
        theta: float = 1e-8
    ):
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
        
        print(f'\n-------- Policy Iteration --------:')
        for i in range(max_iter):
            PI_old = PI.copy()
            print(f'Iteration {i+1}: ', end='')
            # Policy Evaluation
            V = self.PolicyEval(V, PI, gamma, theta)

            # Policy Improvement
            PI = self.PolicyImprovement(V, gamma)
            
            PI_fn = lambda s: PI[s]
            success_rate, mean_rew = self.TestPolicy(PI_fn, trials=100, render=False, verbose=True)
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
                    Q[a] += self.P_hat[s, a, nxt_s] * (self.R_hat[s, a, nxt_s] + gamma * V[nxt_s])
        return Q

    def ValueIter(
        self,
        max_iter: int = 50,
        gamma: float = 0.99,
        theta: float = 1e-8
    ):
        assert max_iter > 0 and isinstance(max_iter, int)
        assert 0 < gamma < 1
        
        # Initialize V(s), \pi(s)
        V = np.zeros(self.nS)
        PI = np.zeros(self.nS, dtype=int)  # since actions are integers
        success_rates = []
        mean_rewards = []

        print(f'\n-------- Value Iteration --------:')
        for i in range(max_iter):
            print(f'Iteration {i+1}: ', end='')
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
            success_rate, mean_rew = self.TestPolicy(PI_fn, trials=100, render=False, verbose=True)
            success_rates.append(success_rate)
            mean_rewards.append(mean_rew)
        return PI, V, success_rates, mean_rewards
                

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
    args = parser.parse_args()

    # Current path
    path = pathlib.Path(__file__).parent.resolve()
    
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
    # 3. Naive policy
    policy = lambda s: (s + 1) % 4
    naive_success_rates = []
    for _ in range(10):
        naive_success_rate, _ = mdp.TestPolicy(policy, render=args.render)
        naive_success_rates.append(naive_success_rate)
    print(f"Average naive_success_rates: {np.mean(naive_success_rates)}")

    # =============================================================================================
    # 4. LearnModel()
    p1, r1 = mdp.learnModel(n_samples=10 ** 5)
    mse = lambda x, y: np.mean((x - y) ** 2)
    
    MSE_P, MSE_R =  mse(mdp.P, p1), mse(mdp.R, r1)
    ic(MSE_P, MSE_R)
    
    # =============================================================================================
    # 5. Policy iteration
    PI, V_pi, success_rates, mean_rewards = mdp.PolicyIteration(50, theta=sys.float_info.epsilon)
    ic(PI)
    ic(V_pi)

    plt.plot(success_rates)
    plt.xlabel("Iteration")
    plt.ylabel("Success rate")
    plt.title("Average rate of success of the learned policy (Policy Iteration)")
    # plt.show()
    
    # =============================================================================================
    # 5. Value iteration
    PI, V_pi, success_rates, mean_rewards = mdp.ValueIter(50, theta=sys.float_info.epsilon)
    ic(PI)
    ic(V_pi)

    plt.plot(success_rates)
    plt.xlabel("Iteration")
    plt.ylabel("Success rate")
    plt.title("Average rate of success of the learned policy (Value Iteration)")
    plt.show()