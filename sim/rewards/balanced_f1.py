from typing import List
import bittensor as bt
import numpy as np

from .base_reward import Reward
from .utils import MinerPerformanceTracker
from .reward_registry import REWARD_REGISTRY


@REWARD_REGISTRY.register_module(module_name='BalancedF1Reward')
class BalancedF1Reward(Reward):

    def __init__(self, verbose=False):
        super().__init__("BalancedF1Reward")
        self.performance_tracker = MinerPerformanceTracker()
        self.verbose = verbose

    def penalty(self, y_pred: float, historical_performance: float) -> float:
        # Penalize if prediction is not within [0, 1]
        bad = (y_pred < 0.0) or (y_pred > 1.0)
        low_accuracy = historical_performance < 0.50
        return 0.0 if bad or low_accuracy else 1.0

    def __call__(
        self,
        uids: List[int],
        responses: List[float],
        label: float,
        hotkeys: List[str] = None
    ) -> np.array:
        """
        Returns an array of rewards for the given label and miner responses.

        Args:
        - label (float): The true label (1.0 for fake, 0.0 for real).
        - responses (List[float]): A list of responses from the miners.
        - uids (List[int]): List of miner UIDs.
        - axons (List[bt.axon]): List of miner axons.
        - performance_tracker (MinerPerformanceTracker): Tracks historical performance metrics per miner.

        Returns:
        - np.array: An array of rewards for the given label and responses.
        """
        hotkeys = ['mock'] * len(uids) if hotkeys is None else hotkeys

        miner_rewards = []
        for uid, miner_hotkey, pred_prob in zip(uids, hotkeys, responses):
            try:            
                # Check if the miner hotkey has changed
                if uid in self.performance_tracker.miner_hotkeys and self.performance_tracker.miner_hotkeys[uid] != miner_hotkey:
                    self.performance_tracker.reset_miner_history(uid, miner_hotkey)
                    if self.verbose:
                        bt.logging.info(f"Miner hotkey changed for UID {uid}. Reset performance metrics.")

                # Apply penalty if prediction is invalid
                pred = int(np.round(pred_prob))
                true_label = int(label)

                # Update miner's performance history
                self.performance_tracker.update(uid, pred, true_label, miner_hotkey)
                
                #is_new_miner = self.performance_tracker.get_prediction_count(uid) < 10
                metrics = self.performance_tracker.get_metrics(uid, window=10)
                metrics_flipped = self.performance_tracker.get_metrics(uid, window=10, flip=True)

                f1 = metrics['f1_score']
                f1_flipped = metrics_flipped['f1_score']
                reward = (f1 + f1_flipped) / 2.
                
                penalty = self.penalty(pred_prob, reward)
                reward *= penalty

                miner_rewards.append(reward)
                
            except Exception as e:
                #bt.logging.error(f"Couldn't calculate reward for miner {uid}, prediction: {responses[uid] if uid < len(responses) else 'N/A'}, label: {label}")
                bt.logging.exception(e)
                miner_rewards.append(0.0)

        return np.array(miner_rewards)
