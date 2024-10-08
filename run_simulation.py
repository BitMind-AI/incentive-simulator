from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm.auto import tqdm
import pandas as pd
import argparse
import joblib
import time
import os

from sim.plotting import plot_metric, plot_multi_validator_metric, plot_incentive, plot_incentives, plot_incentive_over_time
from sim.subnet_data import get_wandb_history, align_dataframes_by_timestamp
from sim.incentive import assemble_W, assemble_S, compute_incentive
from sim.blocks import average_challenges_per_tempo
from sim import Simulator, run_simulations

WANDB_VALIDATOR_UIDS = [
    17, 20, 250, 6, 39, 19, 107, 193, 131, 122, 236, 107
]


def load_history(
    cache_dir: str, 
    load_cached: bool, 
    validators: List[str], 
    start_ts: int,
    end_ts: int) -> Dict[str, pd.DataFrame]:
    """
    Load validator history from cache or W&B.

    Args:
        fname (str): Filename for the cache.
        load_cached (bool): Whether to load from cache.
        write_cache (bool): Whether to write to cache.
        validators (List[str]): List of validator names.
        start_ts (int): Start timestamp for W&B query.
        end_ts (int): End timestamp for W&B query.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of validator histories.
    """
    history_dfs = {}
    for vali in validators:
        vali_history_path = os.path.join(cache_dir, f"{vali}-history.pkl")
        if load_cached and os.path.exists(vali_history_path):
            history_dfs[vali] = joblib.load(vali_history_path)
            print(f"Loaded {vali_history_path}")
        else:
            history_dfs[vali] = get_wandb_history(
                project='bitmind-subnet',
                entity='bitmindai',
                validator_name=vali,
                start_ts=start_ts,
                end_ts=end_ts, 
                verbosity=2)
            joblib.dump(history_dfs[vali], vali_history_path)
            print(f"Saved {vali_history_path}")
    return history_dfs


def process_history(
    history_dfs: Dict[str, pd.DataFrame], 
    min_avg_challenges: Optional[int],
    limit: int) -> Dict[str, pd.DataFrame]:
    """
    Process and filter validator histories.

    Args:
        history_dfs (Dict[str, pd.DataFrame]): Dictionary of validator histories.
        min_avg_challenges (Optional[int]): Minimum average challenges per tempo.

    Returns:
        Dict[str, pd.DataFrame]: Filtered dictionary of validator histories.
    """
    drop = []
    fmt_time = lambda ts: datetime.fromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S')
    for vali, history_df in history_dfs.items():
        print(vali)
        if history_df.shape[0] == 0:
            avg, counts = 0, 0
            min_ts, max_ts = 'n/a', 'n/a'
        else:
            avg, counts = average_challenges_per_tempo(history_df)
            min_ts = fmt_time(history_df._timestamp.min())
            max_ts = fmt_time(history_df._timestamp.max())
        print(f'\tDate Range: {min_ts} : {max_ts}')
        print(f'\tDataFrame shape: {history_df.shape}')
        print(f'\tAverage Challenges Per Tempo: {avg}')
        print(f'\t\tChallenges Per Tempo: {counts}')
        if min_avg_challenges and min_avg_challenges > avg:
            drop.append(vali)
    
    if min_avg_challenges and len(drop) > 0:
        print(f"Dropping {drop}")
        history_dfs = {vali: df for vali, df in history_dfs.items() if vali not in drop}

    if limit is not None:
        print(f"Truncating validator data to first {limit} rows")
        for v in history_dfs:
            history_dfs[v] = history_dfs[v].iloc[:limit]
    
    print("Loaded data from validators:")
    for vali in history_dfs:
        print(f"\t{vali}")
    
    return history_dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations on validator data.")
    parser.add_argument("--name", help="Experiment name, used to name output directory")
    parser.add_argument("--reward_fns", nargs="+", default=['BalancedF1Reward'], help="List of reward functions to use")
    parser.add_argument("--load_cached_history", action="store_true", help="Load history from cached file")
    parser.add_argument("--start_ts", type=int, default=1727940981, help="Start timestamp for W&B query")
    parser.add_argument("--end_ts", type=int, default=1728176143, help="End timestamp for W&B query")
    parser.add_argument("--min_avg_challenges", type=int, default=40, help="Minimum average challenges per tempo")
    parser.add_argument("--limit", type=int, default=None, help="Limit history rows for quicker testing")
    parser.add_argument("--validator_uids", nargs="+", type=int, default=WANDB_VALIDATOR_UIDS, help="List of validator UIDs")

    args = parser.parse_args()

    os.makedirs(args.name, exist_ok=True)

    validators = [f'validator-{uid}-1.1.0' for uid in args.validator_uids]

    start = time.time()
    history_dfs = load_history(
        cache_dir=args.name, 
        load_cached=args.load_cached_history,
        validators=validators,
        start_ts=args.start_ts,
        end_ts=args.end_ts)
    
    history_dfs = process_history(
        history_dfs, 
        args.min_avg_challenges, 
        args.limit)
    print(f"Done. Loaded and processed subnet data in {time.time() - start}s")

    start = time.time()
    scored_dfs = run_simulations(history_dfs, args.reward_fns)
    joblib.dump(scored_dfs, os.path.join(args.name, f'simulation-{time.time()}.pkl'))
    
    scored_dfs = align_dataframes_by_timestamp(scored_dfs)
    joblib.dump(scored_dfs, os.path.join(args.name, f'simulation-aligned-{time.time()}.pkl'))
    
    print(f"Done. Ran simulations in {time.time() - start}s")