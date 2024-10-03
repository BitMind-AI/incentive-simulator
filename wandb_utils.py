from collections import defaultdict
from datetime import datetime
import pandas as pd
import wandb


def get_wandb_history(
    project='bitmind-subnet',
    entity='bitmindai',
    validator_name='validator-193-1.1.0',
    running_only=False,
    start_ts=None,
    end_ts=None, 
    verbosity=0):

    api = wandb.Api()
    validator_runs = api.runs(f"{entity}/{project}")

    filters = None
    if start_ts:
        formatted_time = datetime.fromtimestamp(start_ts).strftime('%Y-%m-%dT%H:%M:%S')
        filters = {"created_at": {"$gte": formatted_time}}

    wandb_proj = f"{entity}/{project}"
    print(f"Querying wandb probject {wandb_proj} with filters {filters}")
    runs = api.runs(wandb_proj, filters=filters)

    all_history_df = pd.DataFrame()
    for run in validator_runs:
        if run.name != validator_name:
            continue

        if running_only and run.state != 'running':
            continue

        history_df = run.history()
        if history_df.shape[0] == 0:
            continue

        if start_ts is not None:
            #history_df = history_df[history_df['_timestamp'] >= start_ts]
            if run.summary['_timestamp'] < start_ts:
                continue
        if end_ts is not None:
            #history_df = history_df[history_df['_timestamp'] <= end_ts]
            if run.summary['_timestamp'] > end_ts:
                continue

        if history_df.shape[0] == 0:
            continue

        readable_time = datetime.fromtimestamp(run.summary['_timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Loaded {history_df.shape[0]} challenges from {run.name} ({readable_time})")

        if 'miner_uids' in history_df.columns:
            history_df = history_df.rename({'miner_uids': 'miner_uid'}, axis=1)
        if 'predictions' in history_df.columns:
            history_df = history_df.rename({'predictions': 'pred'}, axis=1)

        history_df = history_df[['_timestamp', 'miner_uid', 'pred', 'label']]

        all_history_df = pd.concat([all_history_df, history_df], axis=0)
    return all_history_df