import random
from enum import Enum

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F


class StrToActivation(Enum):
    """Torch activation function."""

    relu = nn.ReLU()
    relu_inplace = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    leaky_relu = nn.LeakyReLU()
    sigmoid = nn.Sigmoid()
    selu = nn.SELU()
    softplus = nn.Softplus()
    identity = nn.Identity()


class MLPDeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        # The final output layer of the actor was a tanh layer, to bound the actions
        self.net = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim],
            activation,
            output_activation="tanh",
        )
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit

        self.hidden_sizes = hidden_sizes

        # init weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights of the neural network.
        The other layers were initialized from uniform distributions[−1 / sqrt(f), 1 / sqrt(f)]
            where f is the fan-in of the layer.
        The final layer weights and biases of both the actor and critic were initialized
            from a uniform distribution [−3 × 10−3, 3 × 10−3] and [3 × 10−4, 3 × 10−4]] for the
        """
        w1 = 1 / np.sqrt(self.obs_dim)
        w2 = 1 / np.sqrt(self.hidden_sizes[0])
        w3 = 3e-3

        self.net[0].weight.data.uniform_(-w1, w1)
        self.net[2].weight.data.uniform_(-w2, w2)
        self.net[4].weight.data.uniform_(-w3, w3)

    def forward(self, state: th.Tensor) -> th.Tensor:
        return self.net(state) * self.act_limit

    def get_action(self, obs: th.tensor, noise_scale: float) -> np.ndarray:
        """

        :param obs:
        :param noise_scale: Std for Gaussian random noise
        """
        with th.no_grad():
            act_th = self(obs)
            act_np = act_th.cpu().detach().numpy()
        # adding an uncorrelated, mean-zero Gaussian noise
        act_np += noise_scale * np.random.randn(self.act_dim)
        return np.clip(act_np, -self.act_limit, self.act_limit)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        
        # self.q = mlp([obs_dim] + self.hidden_sizes + [1], activation)
        self.fc1 = nn.Linear(obs_dim, self.hidden_sizes[0])
        self.fc2 = nn.Linear(self.hidden_sizes[0] + self.act_dim, self.hidden_sizes[1])
        self.fc3 = nn.Linear(self.hidden_sizes[1], 1)
        
        self._init_weights()
        
        
    
    def forward(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        x = self.fc1(state)
        x = F.relu(x)
        
        x = self.fc2(th.cat([x, action], dim=-1))
        x = F.relu(x)
        
        q_val = self.fc3(x)
        return th.squeeze(q_val, -1)

    def _init_weights(self):
        """
        Initialize the weights of the neural network.
        The other layers were initialized from uniform distributions[−1 / sqrt(f), 1 / sqrt(f)]
            where f is the fan-in of the layer.
        The final layer weights and biases of both the actor and critic were initialized
            from a uniform distribution [−3 × 10−3, 3 × 10−3]
        """
        w1 = 1 / np.sqrt(self.obs_dim)
        w2 = 1 / np.sqrt(self.hidden_sizes[0] + self.act_dim)
        w3 = 3e-3

        self.fc1.weight.data.uniform_(-w1, w1)
        self.fc2.weight.data.uniform_(-w2, w2)
        self.fc3.weight.data.uniform_(-w3, w3)
    
    
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)

device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if th.cuda.is_available() and use_gpu:
        device = th.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = th.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_random_seed(seed: int) -> None:
    """Set random seed to both numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


def mlp(sizes, activation, output_activation=nn.Identity()):
    # String name to Activation function conversion
    if isinstance(activation, str):
        activation = StrToActivation[activation.lower()].value
    if isinstance(output_activation, str):
        output_activation = StrToActivation[output_activation.lower()].value
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


def from_numpy(*args, **kwargs):
    return th.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor: th.Tensor) -> np.ndarray:
    return tensor.to("cpu").detach().numpy()


def to_torch(array) -> th.Tensor:
    """
    Convert a numpy array to a PyTorch tensor.
    Note: it copies the data by default.
    :param array:
    :param device: PyTorch device to which the values will be converted.
    :return: torch tensor.
    """
    return th.as_tensor(array, dtype=th.float32, device=device)
