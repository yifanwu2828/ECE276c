import time

import numpy as np
import matplotlib.pyplot as plt

import gym
import pybullet_envs

from icecream import ic


l0 = 0.1
l1 = 0.11


def getForwardModel(q0, q1):
    """
    Returns the end-effector position given the joint angles.
    :param q0: angle of joint 0
    :param q1: angle of joint 1
    :return: x, y postion of the end-effector
    """
    x = l0 * np.cos(q0) + l1 * np.cos(q0 + q1)
    y = l0 * np.sin(q0) + l1 * np.sin(q0 + q1)
    return x, y


def getJacobian(q0, q1):
    """
    Returns the Jacobian matrix of the forward model.
    :param q0: angle of joint 0
    :param q1: angle of joint 1
    :return: Jacobian matrix
    """
    J = np.array(
        [
            [-l0 * np.sin(q0) - l1 * np.sin(q0 + q1), -l1 * np.sin(q0 + q1)],
            [l0 * np.cos(q0) + l1 * np.cos(q0 + q1), l1 * np.cos(q0 + q1)],
        ]
    )
    return J


def get_reference_trajectory(theta):
    """
    Returns the reference trajectory.
    :param theta: angle \in [-pi, pi]
    :return: reference trajectory
    """
    x_ref = (0.19 + 0.02 * np.cos(4 * theta)) * np.cos(theta)
    y_ref = (0.19 + 0.02 * np.cos(4 * theta)) * np.sin(theta)
    return x_ref, y_ref


if __name__ == "__main__":
    env = gym.make("ReacherBulletEnv-v0")
    # env.render(mode="human")
    env.reset()

    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.action_space}")

    q0, q0_dot = env.unwrapped.robot.central_joint.current_position()
    ic(q0, q0_dot)

    q1, q1_dot = env.unwrapped.robot.elbow_joint.current_position()
    ic(q1, q1_dot)

    x, y = getForwardModel(q0, q1)
    ic(x, y)

    X_ref, Y_ref = [], []
    for theta in np.linspace(-np.pi, np.pi, 360):
        x_ref, y_ref = get_reference_trajectory(theta)
        X_ref.append(x_ref)
        Y_ref.append(y_ref)
    plt.plot(X_ref, Y_ref)
    plt.xlim(-0.3, 0.3)
    plt.show()
