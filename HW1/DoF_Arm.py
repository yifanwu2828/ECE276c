import time
import pathlib
from  typing import Tuple


import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import gym
import pybulletgym

from icecream import ic


class EzReacher(gym.ObservationWrapper):
    """
    A custom env wrapper 
    """

    def __init__(self, env):
        super().__init__(env)
        self.l0 = 0.1
        self.l1 = 0.11
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation
    
    def setJointPosition(self, q0, q1) -> None:
        """
        Takes in a joint position and returns the next state and reward
        """
        self.unwrapped.robot.central_joint.reset_position(q0, 0)
        self.unwrapped.robot.elbow_joint.reset_position(q1, 0)
    
    def getCurrentJointPosition(self) -> np.ndarray:
        """
        Returns the current joint position and velocity
        """
        q0, q0_dot = self.unwrapped.robot.central_joint.current_position()
        q1, q1_dot = self.unwrapped.robot.elbow_joint.current_position()
        return q0, q0_dot, q1, q1_dot
    
    def getForwardModel(self, q0, q1) -> Tuple[float, float]:
        """
        Returns the end-effector position given the joint angles.
        :param q0: angle of joint 0 in radians.
        :param q1: angle of joint 1 in radians.
        :return: x, y postion of the end-effector
        """
        x = self.l0 * np.cos(q0) + self.l1 * np.cos(q0 + q1)
        y = self.l0 * np.sin(q0) + self.l1 * np.sin(q0 + q1)
        return x, y

    def getInverseKinematics(self, x, y) -> Tuple[float, float]:
        """
        Returns the inverse kinematics of the robot.
        :param x: x position of the end-effector
        :param y: y position of the end-effector
        :return: joint angles q0 and q1 in radians
        """
        q1 = np.arccos((x**2 + y**2 - self.l0**2 - self.l1**2) / (2 * self.l0 * self.l1))
        q0 = np.arctan2(y, x) - np.arctan2(self.l1 * np.sin(q1), self.l0 + self.l1 * np.cos(q1))
        return q0, q1

    def getJacobian(self, q0, q1) -> np.array:
        """
        Returns the Jacobian matrix of the forward model.
        :param q0: angle of joint 0 in radians.
        :param q1: angle of joint 1 in radians.
        :return: Jacobian matrix
        """
        J = np.array(
            [
                [-self.l0 * np.sin(q0) - self.l1 * np.sin(q0 + q1), -self.l1 * np.sin(q0 + q1)],
                [self.l0 * np.cos(q0) + self.l1 * np.cos(q0 + q1), self.l1 * np.cos(q0 + q1)],
            ]
        )
        return J


class PDController:
    def __init__(self, p: np.ndarray, d: np.ndarray, env: gym.Env):
        self.Kp = np.diag([p, p])
        self.Kd = np.diag([d, d])
        self.env = env
    
    def end_effector_control(self, state_err, velocity_err, q0, q1):
        # P control
        delta_state = self.Kp @ state_err + self.Kd @ velocity_err
                
        J = self.env.getJacobian(q0, q1)
        J_inv, rank = la.pinv(J, return_rank=True)
        assert rank == J.shape[1], "Jacobian matrix is singular"
        q = J_inv @ delta_state
        return q
        

def get_reference_trajectory(num_steps, start= -np.pi, end= np.pi):
    """
    Returns the reference trajectory.
    :param theta: angle \in [-pi, pi]
    :return: (x,y) of reference trajectory
    """
    angles = np.linspace(start, end, num_steps)
    ref = np.empty((num_steps, 2))
    for t, theta in enumerate(angles):
        x_ref = (0.19 + 0.02 * np.cos(4 * theta)) * np.cos(theta)
        y_ref = (0.19 + 0.02 * np.cos(4 * theta)) * np.sin(theta)
        ref[t, :] = x_ref, y_ref
    return ref

def getReferenceVelocity(theta):
    """
    Get reference velocity
    :param theta: float, theta
    :return: x_dot, y_dot velocity of reference trajectory
    """
    x_dot = - np.sin(theta)*(np.cos(4*theta)/50 + 19/100) - \
        (2*np.sin(4*theta)*np.cos(theta))/25
    y_dot = np.cos(theta)*(np.cos(4*theta)/50 + 19/100) - \
        (2*np.sin(4*theta)*np.sin(theta))/25
    return x_dot, y_dot


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.resolve()

    env = gym.make("ReacherPyBulletEnv-v0")
    env = EzReacher(env)
    # env.seed(42)
    # env.render(mode="human")
    env.reset()
    # PID parameters
    p = 200
    d = 10
    contorller = PDController(p, d, env)
    
    # Time step
    T = 400
    V_ref = 0
    
    # -------------------------------------------------------------------------
    # Save trajectory
    # with open(path/"ref.npy", "wb") as f:
        # np.save(f, get_reference_trajectory(T))
    # -------------------------------------------------------------------------
    
    # Load reference trajectory
    with open(path/"ref.npy", "rb") as f:
        REF = np.load(f)
        
    # errors
    e = np.empty_like((REF))
    # real trajectory
    traj = np.empty_like(REF)

    # q0, q1 = env.getInverseKinematics(REF[0, 0], REF[0, 1])
    # env.setJointPosition(q0, q1)
    
    for t in range(0, T):
        q0, q0_dot, q1, q1_dot = env.getCurrentJointPosition()
        cur_X = env.getForwardModel(q0, q1)
        cur_J = env.getJacobian(q0, q1)        
        cur_V = cur_J @ np.array([q0_dot, q1_dot])

        state_err = REF[t, :] - cur_X
        velocity_err = V_ref - cur_V
            
        q = contorller.end_effector_control(state_err, velocity_err, q0, q1)
        
        env.step(q)
        # time.sleep(0.1)
        traj[t, :] = cur_X                               

    MSE = np.mean((REF - traj)**2)
    ic(MSE)

    plt.plot(REF[:, 0], REF[:, 1], label="reference")
    plt.scatter(REF[0, 0], REF[0, 1], marker='^', c='b', label="reference_start&end")
    plt.scatter(traj[0, 0], traj[0, 1], c="r", label="start")
    plt.scatter(traj[-1, 0], traj[-1, 1], c='g', label="end")
    plt.plot(traj[:, 0], traj[:, 1], label="trajectory")
    plt.xlim(-0.3, 0.3)
    plt.legend()
    plt.show()
