from collections import defaultdict
from functools import partial
from tqdm import tqdm
import multiprocessing as mp
import bittensor as bt
import pandas as pd
import numpy as np

from rewards import REWARD_REGISTRY, Reward
from scoring import update_scores   
from weights import set_weights


class Simulator:

    def __init__(
            self,
            reward_fns=['BinaryReward', 'WeightedHistoryReward'],
        ):
        """
        """
        self.reward_fns = [REWARD_REGISTRY[fn] for fn in reward_fns]

    def run(self, history_df: pd.DataFrame):
        with mp.Pool(mp.cpu_count()) as pool:
            result_dfs = pool.map(
                partial(self._run_sim, history_df=history_df),
                self.reward_fns)

        for reward, result_df in zip(self.reward_fns, result_dfs):
            cols = [
                '_'.join([col, reward.name])
                for col  in ['rewards', 'scores', 'weights']
            ]
            history_df[cols] = result_df[cols]
        return history_df


    def _run_sim(self, history_df: pd.DataFrame, reward: Reward):
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

        progress_bar = tqdm(
            history_df.iterrows(), 
            total=len(history_df),
            #position=mp.current_process()._identity[0],
            desc="Computing Rewards, Scores and Weights")

        for i, challenge_row in progress_bar:
            label = challenge_row['label']
            preds = challenge_row['pred']
            uids = challenge_row['miner_uid']

            rewards = reward(uids, preds, label)
            history['rewards'].append(rewards)
        
            scores = update_scores(scores, rewards, uids)
            history['scores'] = scores

            new_weight_uids, new_weights = set_weights(scores['new'], metagraph, subtensor)
            weight_dict = dict(zip(new_weight_uids, new_weights))
            history['weights'].append(weight_dict)

        history_df[f'rewards_{reward.name}'] = history['rewards']
        history_df[f'scores_{reward.name}'] = history['scores']
        history_df[f'weights_{reward.name}'] = history['weights']

        return history_df
       