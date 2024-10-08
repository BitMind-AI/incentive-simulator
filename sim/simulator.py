from collections import defaultdict
from typing import Dict, Tuple, List
from functools import partial
from tqdm.notebook import tqdm
import multiprocessing as mp
import bittensor as bt
import pandas as pd
import numpy as np
import itertools
import psutil
import os

from .rewards import REWARD_REGISTRY, Reward
from .scoring import update_scores   
from .weights import set_weights


def _simulation_fn(reward_data_tup: Tuple[Tuple[str, pd.DataFrame], Reward]):
    """
    Helper function for parallelizing simulations
    """
    print()  # to force progress bars to show up in notebook
    reward_cls = reward_data_tup[1]
    history_df = reward_data_tup[0][1]
    vali_name = reward_data_tup[0][0]
    history_df.name = vali_name
    sim = Simulator(reward_cls)
    return vali_name, reward_cls, sim.run(history_df)


def worker(reward_data_tup):
    # Set CPU affinity
    process = psutil.Process()
    cpu_id = os.getpid() % psutil.cpu_count(logical=True)
    process.cpu_affinity([cpu_id])
    
    # Set nice value (lower value = higher priority, be careful with negative values)
    os.nice(10)  # slightly lower priority than default
    return _simulation_fn(reward_data_tup)

    
def run_simulations(
    history_dfs: Dict[str, pd.DataFrame], 
    reward_cls_list: List[str]=['BinaryReward', 'WeightedHistoryReward']
) -> Dict[str, pd.DataFrame]:
    """
    Parallelize simulations with optimized resource allocation
    """
    vali_df_tuples = [(vali, df.copy()) for vali, df in history_dfs.items()]
    vali_reward_combos = list(itertools.product(vali_df_tuples, reward_cls_list))
    n_cpu = psutil.cpu_count(logical=True)

    print(f"Parallelizing {len(vali_reward_combos)} simulations over {n_cpu} cpus")
    with mp.get_context("spawn").Pool(n_cpu) as pool:
        result_tuples = list(pool.imap_unordered(worker, vali_reward_combos))

    for vali, reward_cls, result_df in result_tuples:
        cols = ['_'.join([col, reward_cls]) for col in ['rewards', 'scores', 'weights']]
        history_dfs[vali][cols] = result_df[cols]
    
    return history_dfs


class Simulator:

    def __init__(self, reward, metagraph=None, subtensor=None):
        self.reward = REWARD_REGISTRY[reward]()
        self.metagraph = bt.metagraph(netuid=34) if metagraph is None else metagraph
        self.subtensor = bt.subtensor() if subtensor is None else subtensor

    def run(self, history_df: pd.DataFrame):
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
        scores = np.zeros(256, dtype=np.float32) 
        history = {
            'weights': [],
            'scores': [],
            'rewards': []
        }

        pbar_desc = self.reward.name
        if hasattr(history_df, 'name'):
            pbar_desc += f" | {history_df.name}"

        pid = mp.current_process()._identity[0]
        progress_bar = tqdm(
            history_df.iterrows(), 
            total=len(history_df),
            position=pid,
            desc=pbar_desc)

        for i, challenge_row in progress_bar:
            label = challenge_row['label']
            preds = challenge_row['pred']
            uids = challenge_row['miner_uid']

            rewards = self.reward(uids, preds, label)
            history['rewards'].append(rewards)
        
            scores = update_scores(scores, rewards, uids)
            history['scores'].append(scores)

            new_weight_uids, new_weights = set_weights(scores, self.metagraph, self.subtensor)
            weight_dict = dict(zip(new_weight_uids, new_weights))
            history['weights'].append(weight_dict)

        history_df.loc[:, f'rewards_{self.reward.name}'] = history['rewards']
        history_df.loc[:, f'scores_{self.reward.name}'] = history['scores']
        history_df.loc[:, f'weights_{self.reward.name}'] = history['weights']

        return history_df
       