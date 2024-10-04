from typing import List
import bittensor as bt
import numpy as np

from reward import Reward
from reward_registry import REWARD_REGISTRY


@REWARD_REGISTRY.register_module(module_name='BinaryReward')
class BinaryReward(Reward):

    def __init__(self):
        super().__init__("BinaryReward")

    def penalty(self, y_pred: float) -> float:
        # Penalize if prediction is not within [0, 1]
        bad = (y_pred < 0.0) or (y_pred > 1.0)
        return 0.0 if bad else 1.

    def __call__(
            self,
            label: float,
            responses: List,
    ) -> np.array:
        """
        Returns a tensor of rewards for the given query and responses.

        Args:
        - label (float): 1 if image was fake, 0 if real.
        - responses (List[float]): A list of responses from the miners.

        Returns:
        - np.array: A tensor of rewards for the given query and responses.
        """
        miner_rewards = []
        for uid in range(len(responses)):
            try:
                pred = responses[uid]
                reward = 1. if np.round(pred) == label else 0.
                reward *= self.penalty(pred, 1.)
                miner_rewards.append(reward)

            except Exception as e:
                bt.logging.error("Couldn't count miner reward for {}, his predictions = {} and his labels = {}".format(
                    uid, responses[uid], label))
                bt.logging.exception(e)
                miner_rewards.append(0)

        return np.array(miner_rewards)