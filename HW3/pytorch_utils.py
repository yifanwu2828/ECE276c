
import random
from enum import Enum

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical, Normal, MultivariateNormal


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


class MLPCategoricalpolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, state: th.Tensor)-> th.Tensor:
        return self.net(state)

    def get_action(self, state)-> th.Tensor:
        logits = self.forward(state)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob

class MLPDiagGaussianpolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, std_init: float = 0.2):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.sigma = th.nn.Parameter(th.diag(std_init * th.ones(act_dim, dtype=th.float32)), requires_grad=True)
        orthogonal_init(self.net, gain = np.sqrt(2))
        
    def forward(self, state: th.Tensor) -> th.Tensor:
        return self.net(state)

    def get_action(self, state: th.Tensor, deterministic=False) ->  th.Tensor:
        mu = self.forward(state)
        cov = th.abs(self.sigma) + 1e-3
        action_dist = MultivariateNormal(mu, cov)
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
        return action, log_prob

def orthogonal_init(module: nn.Module, gain: float = 1) -> None:
    """Orthogonal initialization."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

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