import argparse

import numpy as np
import matplotlib.pyplot as plt

from racecar.SDRaceCar import SDRaceCar


try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


def wrap_angle(angle):
    """
    converts angles outside of +/-PI to +/-PI
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi



def plot_trajectory(true_traj, ref_traj, title):
    """
    Plots the reference and simulation trajectories.
    :param true_traj: true trajectory
    :param ref: reference trajectory
    :param title: title of the plot
    """
    plt.scatter(
        ref_traj[0, 0], ref_traj[0, 1], marker="^", c="b", label="reference_start&end"
    )
    plt.plot(
        ref_traj[0, :],
        ref_traj[1, :],
        'k',
        linewidth=3,
        alpha=0.7)
    plt.scatter(true_traj[0, 0], true_traj[0, 1], c="r", label="start")
    plt.scatter(true_traj[-1, 0], true_traj[-1, 1], c="g", label="end")
    plt.plot(true_traj[:, 0], true_traj[:, 1], label="trajectory")
    plt.title(title)
    plt.legend()
    plt.show(block=True)

def plot_error(error):
    plt.gcf().canvas.mpl_connect(
        "key_release_event",
        lambda event: [plt.close() if event.key in ["escape", "Q"] else None],
    )
    plt.plot(np.zeros_like(error), "--")
    plt.plot(error[:, 0] + error[:, 1])
    plt.show(block=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--track",
        "-t",
        type=str,
        choices=["FigureEight", "Linear", "Circle"],
        default="FigureEight",
    )
    args = p.parse_args()

    env = SDRaceCar(render_env=True, track=args.track)

    env.reset()
    # observation space: [x, y, theta, v_x, v_y, theta_dot, h]
    # action space: [wheel angle, thrust]
    
    
    #track model
    track_len = 5000
    if args.track.lower() == "figureeight":
        t = np.linspace(-1 / 2 * np.pi, 3 / 2 * np.pi, track_len)
        track = 20 * np.vstack(
                [np.cos(t), np.multiply(np.sin(t), np.cos(t))])
        track_boundaries = [-40, 40, -20, 20]
        maxStep = 500
        Kp = 0.1
        Kd = 0.35
        
    elif args.track.lower() == "linear":
        track = np.vstack([
                10 * np.linspace(0, 100, track_len),
                0 * np.ones(track_len)
            ])
        track_boundaries = [-5, 35, -20, 20]
        maxStep = 50
        Kp = -100000
        Kd = 10
    else:  # default to circle track
        t = np.linspace(-1 / 2 * np.pi, 3 / 2 * np.pi, track_len)
        track = 10 * np.vstack([np.cos(t), np.sin(t) + 1])
        track_boundaries = [-30, 30, -20, 40]
        maxStep = 350
        Kp = 0.1
        Kd = 0.285


    
    
    done = False
    traj = []
    ref_traj = []
    
    while not done:
        x, y, theta, v_x, v_y, theta_dot, h = env.get_observation()

        dx = h[0] - x
        dy = h[1] - y

        v = np.sqrt(v_x ** 2 + v_y ** 2)

        steering_angle = wrap_angle(np.arctan2(dy, dx) - theta)

        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
        a = R @ np.array([dx, dy])
        
        # if v >5:
            # thrust = -1
        # else:
        thrust = Kp * (a[1]**2 - a[0]**2) + Kd * (0 -v)
        ic(thrust)
        
        thrust = np.clip(thrust, -1, 1)
        act = [steering_angle, thrust]
        nxt_obs, rew, done, _ = env.step(act)
        
        traj.append([x, y])
        ref_traj.append(h)
        # env.render()
    
    traj = np.asarray(traj)
    ref_traj = np.asarray(ref_traj)
    ic(traj.shape, ref_traj.shape)
    
    MSE = np.mean((ref_traj - traj) ** 2)
    ic(MSE)

    
    e = ref_traj - traj
    
    plot_error(e)
    plot_trajectory(traj, track, title=f"{args.track}")
    