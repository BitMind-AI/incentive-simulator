from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
import bittensor as bt
import pandas as pd
import numpy as np

from glicko import GlickoRater
from miner_performance_tracker import MinerPerformanceTracker
from reward import get_rewards, old_get_rewards
from scoring import update_scores   
from weights import set_weights


def run_simulation(history_df: pd.DataFrame, limit: int):
    """
    Iteratively computes rewards, scores and weights from labels and miner predictions

    Args:
        history_df: DataFrame where each row is a challenge, and contains columns 
            'label', 'pred', and 'miner_uid'
        limit: Number of rows to iterate over (for debugging or running over smaller windows). 
            Set to None for all rows

    Returns:
        history_df with additional columns 'rewards_new', 'rewards_old', 'scores_new', 'scores_old'
        Note: if limit is < len(history_df), the first `limit` rows will contain rewards and scores, and 
        the rest will be nan.
    """
    perf_tracker = MinerPerformanceTracker()
    glicko = GlickoRater()
    v_history = defaultdict(list)
    rd_history = defaultdict(list)
    
    metagraph = bt.metagraph(netuid=34)
    subtensor = bt.subtensor()

    miner_sample_size = history_df['miner_uid'].apply(len).unique()[0]

    keys = ['old', 'new']
    scores = {k: np.zeros(256, dtype=np.float32) for k in keys}

    weight_history = {k: [] for k in keys}
    score_history = {k: [] for k in keys}
    reward_history = {k: [] for k in keys}

    limit = len(history_df) if limit is None else limit

    progress_bar = tqdm(
        history_df.iterrows(), 
        total=limit,
        #position=mp.current_process()._identity[0],
        desc="Computing Rewards, Scores and Weights")

    for i, challenge_row in progress_bar:
        if i >= limit:
            break

        label = challenge_row['label']
        preds = challenge_row['pred']
        uids = challenge_row['miner_uid']

        #new_rewards = glicko.update_ratings(uids, preds, label)
        #for uid in range(257):
        #    v_history[uid].append(glicko.volatility[uid])
        #   rd_history[uid].append(glicko.rd[uid])

        new_rewards = get_rewards(
            label,
            preds,
            uids,
            [1] * miner_sample_size, # mocking hotkeys for now
            perf_tracker)

        old_rewards = old_get_rewards(label, preds)
    
        scores['new'] = update_scores(scores['new'], new_rewards, uids)
        scores['old'] = update_scores(scores['old'], old_rewards, uids)
    
        reward_history['new'].append(new_rewards)
        reward_history['old'].append(old_rewards)
    
        score_history['new'].append(scores['new'])
        score_history['old'].append(scores['old'])

        new_weight_uids, new_weights = set_weights(scores['new'], metagraph, subtensor)
        weight_history['new'].append(dict(zip(new_weight_uids, new_weights)))

        old_weight_uids, old_weights = set_weights(scores['old'], metagraph, subtensor)
        weight_history['old'].append(dict(zip(old_weight_uids, old_weights)))
    
    for k in reward_history:
        diff = len(history_df) - len(reward_history[k])
        if diff != 0:
            reward_history[k] += [np.nan] * diff
        history_df['rewards_' + k] = reward_history[k]

    for k in score_history:
        diff = len(history_df) - len(score_history[k])
        if diff != 0:
            score_history[k] += [np.nan] * diff
        history_df['scores_' + k] = score_history[k]

    for k in weight_history:
        diff = len(history_df) - len(weight_history[k])
        if diff != 0:
            weight_history[k] += [np.nan] * diff
        history_df['weights_' + k] = weight_history[k]
    
    return history_df#, rd_history, v_history