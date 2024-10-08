from typing import List
import numpy as np
import bittensor as bt


def update_scores(scores, rewards: np.ndarray, uids: List[int]):
    """Performs exponential moving average on the scores based on the rewards received from the miners."""

    # Check if rewards contains NaN values.
    if np.isnan(rewards).any():
        bt.logging.warning(f"NaN values detected in rewards: {rewards}")
        # Replace any NaN values in rewards with 0.
        rewards = np.nan_to_num(rewards, nan=0)

    # Ensure rewards is a numpy array.
    rewards = np.asarray(rewards)

    # Check if `uids` is already a numpy array and copy it to avoid the warning.
    if isinstance(uids, np.ndarray):
        uids_array = uids.copy()
    else:
        uids_array = np.array(uids)

    # Handle edge case: If either rewards or uids_array is empty.
    if rewards.size == 0 or uids_array.size == 0:
        print(f"rewards: {rewards}, uids_array: {uids_array}")
        print(
            "Either rewards or uids_array is empty. No updates will be performed."
        )
        return

    # Check if sizes of rewards and uids_array match.
    if rewards.size != uids_array.size:
        raise ValueError(
            f"Shape mismatch: rewards array of shape {rewards.shape} "
            f"cannot be broadcast to uids array of shape {uids_array.shape}"
        )

    # Compute forward pass rewards, assumes uids are mutually exclusive.
    # shape: [ metagraph.n ]
    scattered_rewards: np.ndarray = np.zeros_like(scores)
    scattered_rewards[uids_array] = rewards
    #print(f"Scattered rewards: {rewards}")

    # Update scores with rewards produced by this step.
    # shape: [ metagraph.n ]
    alpha: float = .02
    return alpha * scattered_rewards + (1 - alpha) * scores
    #return scattered_rewards / len(scattered_rewards)
