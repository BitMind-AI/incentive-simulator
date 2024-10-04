from collections import defaultdict
from typing import Dict, Tuple, List
from functools import partial
from tqdm.notebook import tqdm
import multiprocessing as mp
import bittensor as bt
import pandas as pd
import numpy as np
import itertools

from .rewards import REWARD_REGISTRY, Reward
from .scoring import update_scores   
from .weights import set_weights


def _simulation_fn(reward_data_tup: Tuple[Tuple[str, pd.DataFrame], Reward]):
    """
    Helper function for parallelizing simulations
    """
    print()  # to force progress bars to show up in notebook
    reward_fn = reward_data_tup[1]
    history_df = reward_data_tup[0][1]
    vali_name = reward_data_tup[0][0]
    history_df.name = vali_name
    sim = Simulator(reward_fn)
    return sim.run(history_df)


def run_simulations(
    history_dfs: Dict[str, pd.DataFrame], reward_cls_list: List[str]=['BinaryReward', 'WeightedHistoryReward']):
    """
    Parallelize simulations
    """

    vali_df_tuples = [(vali, df.copy()) for vali, df in history_dfs.items()] 
    vali_reward_combos = list(itertools.product(vali_df_tuples, reward_cls_list))

    n_cpu = mp.cpu_count()
    print(f"Parallelizing {len(vali_reward_combos)} simulations over {n_cpu} cpus")

    with mp.Pool(n_cpu) as pool:
        result_dfs = pool.map(
            _simulation_fn,
            vali_reward_combos)

    for sim_args, result_df in zip(vali_reward_combos, result_dfs):
        vali = sim_args[0][0]
        reward_cls = sim_args[1]
        cols = [
            '_'.join([col, reward_cls])
            for col  in ['rewards', 'scores', 'weights']
        ]
        history_dfs[vali][cols] = result_df[cols]
    return history_dfs


class Simulator:

    def __init__(self, reward):
        self.reward = REWARD_REGISTRY[reward]()

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
        metagraph = bt.metagraph(netuid=34)
        subtensor = bt.subtensor()
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

            new_weight_uids, new_weights = set_weights(scores, metagraph, subtensor)
            weight_dict = dict(zip(new_weight_uids, new_weights))
            history['weights'].append(weight_dict)

        history_df.loc[:, f'rewards_{self.reward.name}'] = history['rewards']
        history_df.loc[:, f'scores_{self.reward.name}'] = history['scores']
        history_df.loc[:, f'weights_{self.reward.name}'] = history['weights']

        return history_df
       