from typing import List, Dict
import bittensor as bt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def count_penalty(y_pred: float, historical_performance: float) -> float:
    # Penalize if prediction is not within [0, 1]
    bad = (y_pred < 0.0) or (y_pred > 1.0)
    low_accuracy = historical_performance < 0.60
    return 0.0 if bad or low_accuracy else 1.0

def get_rewards(
        label: float,
        responses: List[float],
        uids: List[int],
        #axons: List[bt.axon],
        hotkeys,
        performance_tracker,
        verbose=False
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
    miner_rewards = []
    for uid, miner_hotkey, pred_prob in zip(uids, hotkeys, responses):
        try:            
            # Check if the miner hotkey has changed
            if uid in performance_tracker.miner_hotkeys and performance_tracker.miner_hotkeys[uid] != miner_hotkey:
                bt.logging.info(f"Miner hotkey changed for UID {uid}. Resetting performance metrics.")
                performance_tracker.reset_miner_history(uid, miner_hotkey)

            # Apply penalty if prediction is invalid
            pred = int(np.round(pred_prob))
            true_label = int(label)
            # Update miner's performance history
            performance_tracker.update(uid, pred, true_label, miner_hotkey)

            # Get historical performance metrics for two time windows
            metrics_100 = performance_tracker.get_metrics(uid, window=100)
            metrics_10 = performance_tracker.get_metrics(uid, window=10)
            
            # Check if the miner is new (less than 50 predictions)
            is_new_miner = performance_tracker.get_prediction_count(uid) < 50

            # Calculate rewards for different time windows
            reward_100 = 1.0 if is_new_miner else sum(0.2 * metrics_100[m] for m in ['accuracy', 'precision', 'recall', 'f1_score', 'mcc'])
            if reward_100 > 0.9:
                reward_100 = 1.

            reward_10 = (
                0.2 * metrics_10['accuracy'] +
                0.2 * metrics_10['precision'] +
                0.2 * metrics_10['recall'] +
                0.2 * metrics_10['f1_score'] +
                0.2 * metrics_10['mcc']
            )
            if reward_10 > 0.9:
                reward_10 = 1.
                
            correct = 1. if pred == true_label else 0.

            # Calculate final reward: 20% from 10-prediction window, 80% from correctness
            reward = reward_100 * .50 + reward_10 * .25 + correct * .25
            
            # Calculate and apply penalty
            penalty = count_penalty(pred_prob, reward_100)
            reward *= penalty

            miner_rewards.append(reward)

            # Optionally, log metrics for debugging
            if verbose:
                bt.logging.info(f"""
                Miner {uid} Performance:
                100-prediction window:
                    Accuracy:  {metrics_100['accuracy']:.4f}
                    Precision: {metrics_100['precision']:.4f}
                    Recall:    {metrics_100['recall']:.4f}
                    F1 Score:  {metrics_100['f1_score']:.4f}
                    MCC:       {metrics_100['mcc']:.4f}
                10-prediction window:
                    Accuracy:  {metrics_10['accuracy']:.4f}
                    Precision: {metrics_10['precision']:.4f}
                    Recall:    {metrics_10['recall']:.4f}
                    F1 Score:  {metrics_10['f1_score']:.4f}
                    MCC:       {metrics_10['mcc']:.4f}
                Historical:    {reward_100:.4f}
                Penalty:   {penalty:.4f}
                Final Reward: {reward:.4f}
                """)
            
        except Exception as e:
            #bt.logging.error(f"Couldn't calculate reward for miner {uid}, prediction: {responses[uid] if uid < len(responses) else 'N/A'}, label: {label}")
            bt.logging.exception(e)
            miner_rewards.append(0.0)

    return np.array(miner_rewards)


def old_get_rewards(
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
            reward *= count_penalty(pred, 1.)
            miner_rewards.append(reward)

        except Exception as e:
            bt.logging.error("Couldn't count miner reward for {}, his predictions = {} and his labels = {}".format(
                uid, responses[uid], label))
            bt.logging.exception(e)
            miner_rewards.append(0)

    return np.array(miner_rewards)