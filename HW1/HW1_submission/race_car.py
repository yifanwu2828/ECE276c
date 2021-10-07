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
        ref_traj[0, 0], ref_traj[1, 0], marker="^", c="b", label="reference_start&end"
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

def plot_error(error, title):
    plt.gcf().canvas.mpl_connect(
        "key_release_event",
        lambda event: [plt.close() if event.key in ["escape", "Q"] else None],
    )
    plt.plot(np.zeros_like(error), "--")
    plt.plot(error[:, 0] + error[:, 1])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Error")
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
    p.add_argument("--render", '-r', action="store_true")
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
        Kp = 0.1
        Kd = 0.35
        
    elif args.track.lower() == "linear":
        track = np.vstack([
                10 * np.linspace(0, 100, track_len),
                0 * np.ones(track_len)
            ])
        Kp = 0.0001
        Kd = 0.001
    else:  # default to circle track
        t = np.linspace(-1 / 2 * np.pi, 3 / 2 * np.pi, track_len)
        track = 10 * np.vstack([np.cos(t), np.sin(t) + 1])
        maxStep = 350
        Kp = 0.1
        Kd = 0.285
    
    
    done = False
    traj = []
    ref_traj = []
    
    step = 0
    while not done:
        x, y, theta, v_x, v_y, theta_dot, h = env.get_observation()

        dx = h[0] - x
        dy = h[1] - y

        v = np.sqrt(v_x ** 2 + v_y ** 2)

        steering_angle = wrap_angle(np.arctan2(dy, dx) - theta)

        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
        err = R @ np.array([dx, dy])
        
        
        thrust = Kp * (err[1]**2 + err[0]**2) + Kd * (0 - v)
        thrust = np.clip(thrust, -1, 1)
        
        act = [steering_angle, thrust]
        nxt_obs, rew, done, _ = env.step(act)
        
        traj.append([x, y])
        ref_traj.append(h)
        if args.render:
            env.render()
        step += 1
            
    if args.render:
        plt.close()        

    traj = np.asarray(traj)
    ref_traj = np.asarray(ref_traj)
    
    MSE = np.mean((ref_traj - traj) ** 2)
    ic(MSE)

    ic(step)
    
    e = ref_traj - traj
    plot_error(e, title=f"{args.track} Error")
    plot_trajectory(traj, track, title=f"{args.track}")
    