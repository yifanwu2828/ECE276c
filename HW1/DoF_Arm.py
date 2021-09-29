import time

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("TkAgg")

import gym
import pybulletgym

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
    env = gym.make("ReacherPyBulletEnv-v0")
    # env.render(mode="human")
    env.reset()

    # print(f"observation space: {env.observation_space}")
    # print(f"action space: {env.action_space}")

    q0, q0_dot = env.unwrapped.robot.central_joint.current_position()
    ic(q0, q0_dot)

    q1, q1_dot = env.unwrapped.robot.elbow_joint.current_position()
    ic(q1, q1_dot)

    x, y = getForwardModel(q0, q1)
    ic(x, y)
    
    
    T = 360
    X_ref, Y_ref = [], []
    theta = np.linspace(-np.pi, np.pi, T)
    e = np.zeros((T, 2))
    Kp = 1
    Ki = 0
    Kd = 0
    for t in range(0, T):
        x_ref, y_ref = get_reference_trajectory(theta[t])
        X_ref.append(x_ref)
        Y_ref.append(y_ref)
        
        x_error = x_ref - x
        y_error = y_ref - y
        e[t, :] = [x_error, y_error]
        
        if t == 0:
            continue
        
        ux_t = Kp * e[t, 0] + Ki + Kd * (e[t, 0] - e[t-1, 0])
        uy_t = Kp * e[t, 1] + Ki + Kd * (e[t, 1] - e[t-1, 1])
        
        
                
        J = getJacobian(q0, q1)
        J_inv, rank = la.pinv(J, return_rank=True)
        assert rank == J.shape[1], "Jacobian matrix is singular"
        delta_q = J_inv @ e[t, :]
        # ic(delta_q)
        # import ipdb; ipdb.set_trace()
                
    plt.plot(X_ref, Y_ref)
    plt.xlim(-0.3, 0.3)
    # plt.show()
