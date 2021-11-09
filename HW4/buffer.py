import numpy as np
import torch as th

from typing import Union, Dict

from HW4.pytorch_utils import set_random_seed


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, seed, device, max_size):

        self.device = device

        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size = 0, 0
        self.max_size = max_size
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def store(self, obs, act, rew, done, next_obs):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.next_obs_buf[self.ptr] = next_obs

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        """
        Uniformly sample `batch_size` samples from the buffer
        """
        assert isinstance(batch_size, int), "batch_size must be int"
        assert self.size != 0, "Buffer is empty"
        # Uniform sampling
        ind = self.rng.integers(self.size, size=batch_size)
        return self._get_batch_from_index(ind)

    def _get_batch_from_index(
        self,
        batch_idxes: Union[np.ndarray, slice],
    ) -> Dict[str, th.Tensor]:
        """
        Get a batch data based on index.
        :param batch_idxes: Index of batch.
        """
        assert isinstance(batch_idxes, (slice, np.ndarray))
        sample_batch = {
            "obs": self.to_torch(self.obs_buf[batch_idxes]),
            "act": self.to_torch(self.act_buf[batch_idxes]),
            "rew": self.to_torch(self.rew_buf[batch_idxes]),
            "done": self.to_torch(self.done_buf[batch_idxes]),
            "next_obs": self.to_torch(self.next_obs_buf[batch_idxes]),
        }
        return sample_batch

    def to_torch(self, array):
        return th.as_tensor(array, dtype=th.float32, device=self.device)
