import argparse
import time
import pathlib
from typing import Tuple


import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import gym
import pybulletgym

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class PDReacher(gym.ObservationWrapper):
    """
    A custom env wrapper
    """

    def __init__(self, env, Kp, Kd):
        super().__init__(env)
        self.l0 = 0.1
        self.l1 = 0.11
        self.Kp = Kp
        self.Kd = Kd

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

    def getInverseKinematics(self, x: float, y: float) -> Tuple[float, float]:
        """
        Returns the inverse kinematics of the robot.
        :param x: x position of the end-effector
        :param y: y position of the end-effector
        :return: joint angles q0 and q1 in radians
        """
        q1 = np.arccos(
            (x ** 2 + y ** 2 - self.l0 ** 2 - self.l1 ** 2) / (2 * self.l0 * self.l1)
        )
        q0 = np.arctan2(y, x) - np.arctan2(
            self.l1 * np.sin(q1), self.l0 + self.l1 * np.cos(q1)
        )
        return q0, q1

    def getJacobian(self, q0: float, q1: float) -> np.ndarray:
        """
        Returns the Jacobian matrix of the forward model.
        :param q0: angle of joint 0 in radians.
        :param q1: angle of joint 1 in radians.
        :return: Jacobian matrix
        """
        J = np.array(
            [
                [
                    -self.l0 * np.sin(q0) - self.l1 * np.sin(q0 + q1),
                    -self.l1 * np.sin(q0 + q1),
                ],
                [
                    self.l0 * np.cos(q0) + self.l1 * np.cos(q0 + q1),
                    self.l1 * np.cos(q0 + q1),
                ],
            ]
        )
        return J

    def getInverseJacobian(self, q0, q1) -> np.array:
        """
        Returns the inverse Jacobian matrix of the forward model.
        :return: inverse Jacobian matrix
        """
        J = self.getJacobian(q0, q1)
        J_inv, rank = la.pinv(J, return_rank=True)
        assert rank == J.shape[1], "Jacobian matrix is singular"
        return J_inv

    def end_effector_control(self, state_err, velocity_err, q0, q1):

        delta_state = self.Kp @ state_err + self.Kd @ velocity_err
        J_inv = self.getInverseJacobian(q0, q1)
        q = J_inv @ delta_state
        return q

    def joint_control(self, q_err, q_dot_err):
        q = self.Kp @ q_err + self.Kd @ q_dot_err
        return q


def get_reference_trajectory(num_steps, start=-np.pi, end=np.pi):
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


def plot_trajectory(true_traj, ref_traj, title):
    """
    Plots the reference and simulation trajectories.
    :param ref: reference trajectory
    :param x_sim: x position of the simulation
    :param y_sim: y position of the simulation
    :param x_ref: x position of the reference
    :param y_ref: y position of the reference
    """
    fig = plt.figure(figsize=(15, 10))
    plt.plot(ref_traj[:, 0], ref_traj[:, 1], label="reference")
    plt.scatter(
        ref_traj[0, 0], ref_traj[0, 1], marker="^", c="b", label="reference_start&end"
    )
    plt.scatter(true_traj[0, 0], true_traj[0, 1], c="r", label="start")
    plt.scatter(true_traj[-1, 0], true_traj[-1, 1], c="g", label="end")
    plt.plot(true_traj[:, 0], true_traj[:, 1], label="trajectory")
    plt.title(title)
    plt.xlim(-0.3, 0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error(error: np.ndarray, title: str):
    plt.figure()
    plt.gcf().canvas.mpl_connect(
        "key_release_event",
        lambda event: [plt.close() if event.key in ["escape", "Q"] else None],
    )
    plt.plot(np.zeros_like(error), "--")
    plt.plot(error[:, 0] + error[:, 1])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.show()


def tracking_by_end_effector(args):
    """
    Runs the simulation and plots the results.
    """
    # Initialize the environment
    env = gym.make(args.env_id)
    # Warp the environment with pd controller
    p = 0.579
    d = 0.0295
    Kp = np.diag([p, p])
    Kd = np.diag([d, d])
    env = PDReacher(env, Kp, Kd)
    env.seed(args.seed)
    # Render the environment
    if args.render:
        env.render(mode="human")
    env.reset()
    # Not Random start position
    if not args.random_start:
        print("Start on the same position")
        q0, q1 = env.getInverseKinematics(REF[0, 0], REF[0, 1])
        env.setJointPosition(q0, q1)

    else:
        print("Start on random reachable position")

    # errors
    e = np.empty_like(REF)

    # real trajectory
    traj = np.empty_like(REF)

    for t in range(0, args.num_steps):
        q0, q0_dot, q1, q1_dot = env.getCurrentJointPosition()
        cur_X = env.getForwardModel(q0, q1)
        cur_J = env.getJacobian(q0, q1)
        cur_X_dot = cur_J @ np.array([q0_dot, q1_dot])

        state_err = REF[t, :] - cur_X
        velocity_err = V_ref - cur_X_dot
        e[t, :] = state_err

        q = env.end_effector_control(state_err, velocity_err, q0, q1)

        env.step(q)
        if args.render:
            time.sleep(0.1)
        traj[t, :] = cur_X

    MSE = np.mean((REF - traj) ** 2)
    ic(MSE)

    plot_error(e, title="End Effector Position PD Control Error")
    plot_trajectory(
        true_traj=traj, ref_traj=REF, title="End Effector Position PD Control"
    )


def tracking_by_joint(args):
    """
    Runs the simulation and plots the results.
    """
    # Initialize the environment
    env = gym.make(args.env_id)
    # Warp the environment with pd controller
    p = 2.0
    d = 0.1
    Kp = np.diag([p, p])
    Kd = np.diag([d, d])
    env = PDReacher(env, Kp, Kd)
    env.seed(args.seed)
    # Render the environment
    if args.render:
        env.render(mode="human")
    env.reset()
    # Not Random start position
    if not args.random_start:
        q0, q1 = env.getInverseKinematics(REF[0, 0], REF[0, 1])
        env.setJointPosition(q0, q1)

    # errors
    e = np.empty_like(REF)
    # real trajectory
    traj = np.empty_like(REF)

    for t in range(0, args.num_steps):
        q0, q0_dot, q1, q1_dot = env.getCurrentJointPosition()

        q_ref = env.getInverseKinematics(REF[t, 0], REF[t, 1])

        # Since V_ref is 0
        q_dot_ref = 0

        # error
        q_err = q_ref - np.array([q0, q1])
        q_dot_err = q_dot_ref - np.array([q0_dot, q1_dot])

        q = env.joint_control(q_err, q_dot_err)

        env.step(q)
        # if args.render:
        #     time.sleep(0.1)
        e[t, :] = q_err
        traj[t, :] = env.getForwardModel(q0, q1)

    MSE = np.mean((REF - traj) ** 2)
    ic(MSE)

    plot_error(e, title="Joint Position PD Control Error")
    plot_trajectory(true_traj=traj, ref_traj=REF, title="Joint PD Control")


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument(
        "--env_id",
        type=str,
        help="Envriment to train on",
        choices=["ReacherPyBulletEnv-v0"],
        default="ReacherPyBulletEnv-v0",
    )
    p.add_argument(
        "--num_steps",
        "-ns",
        type=int,
        help="Number of steps to sample from trajectory",
        default=1000,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--random_start",
        "-rs",
        action="store_true",
        help="Random start postion or start on the track",
    )
    p.add_argument("--render", "-r", action="store_true")
    args = p.parse_args()

    # Current path
    path = pathlib.Path(__file__).parent.resolve()

    # -------------------------------------------------------------------------
    # Save trajectory
    # REF = get_reference_trajectory(args.num_steps)
    # with open(path/"ref.npy", "wb") as f:
    #     np.save(f, REF)
    # -------------------------------------------------------------------------

    # Load reference trajectory
    with open(path / "ref.npy", "rb") as f:
        REF = np.load(f)

    # Reference velocity
    V_ref = 0

    tracking_by_end_effector(args)
    tracking_by_joint(args)
