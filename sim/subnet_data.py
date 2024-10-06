from collections import defaultdict
from datetime import datetime
import pandas as pd
import wandb


def formatted_ts_from_epoch(ts):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S')


def get_wandb_history(
    project='bitmind-subnet',
    entity='bitmindai',
    validator_name='validator-193-1.1.0',
    running_only=False,
    start_ts=None,
    end_ts=None, 
    verbosity=1):

    filters = {"display_name": validator_name}
    if start_ts or end_ts:
        filters["created_at"] = {
            "$gte": formatted_ts_from_epoch(start_ts) if start_ts else None,
            "$lte": formatted_ts_from_epoch(end_ts) if end_ts else None
        }

    wandb_proj = f"{entity}/{project}"
    if verbosity > 0:
        print(f"--- Querying {wandb_proj} for {validator_name} ---")
        print(f"\tFilters:")
        for f, v in filters.items():
            print(f"\t\t{f} = {v}")

    api = wandb.Api()
    validator_runs = api.runs(wandb_proj, filters=filters)
    if len(validator_runs) == 0:
        if start_ts is None and end_ts is None:
            return pd.DataFrame()  # base case

        if verbosity > 0:
            print("\tNo runs found with specified filters. Falling back to most recent run with the given name.")
        all_history_df = get_wandb_history(project, entity, validator_name, running_only, None, None, verbosity)
        return filter_df_by_time(all_history_df, start_ts, end_ts, verbosity > 1)

    if verbosity > 0:
        print(f"\tFound {len(validator_runs)} runs")

    all_history_df = pd.DataFrame()
    for run in validator_runs:
        if run.name != validator_name:
            continue

        if running_only and run.state != 'running':
            continue

        history_df = run.history()
        if history_df.shape[0] == 0:
            if verbosity > 1:
                print(f"\tRun {run.id}: Empty history dataframe")
            continue

        run_timestamp = formatted_ts_from_epoch(run.summary['_timestamp'])
        if verbosity > 0:
            print(f"\tRun {run.id}: Loaded {history_df.shape[0]} challenges from {run.name} ({run_timestamp})")

        if 'miner_uids' in history_df.columns:
            history_df = history_df.rename({'miner_uids': 'miner_uid'}, axis=1)
        if 'predictions' in history_df.columns:
            history_df = history_df.rename({'predictions': 'pred'}, axis=1)

        cols = ['_timestamp', 'miner_uid', 'pred', 'label']
        if 'miner_hotkeys' in history_df.columns:
            cols.append('miner_hotkeys')

        all_history_df = pd.concat([all_history_df, history_df[cols]], axis=0)
    return all_history_df


def filter_df_by_time(df, start_ts=None, end_ts=None, verbose=False):
    """ """

    if df.shape[0] == 0:
        return df

    if start_ts is not None:
        rows_before = df.shape[0]
        df = df[df['_timestamp'] >= start_ts]
        rows_after = df.shape[0]
        diff = rows_before - rows_after
        if verbose and diff > 0:
            print(f"\t\tDropped {diff} rows from before {formatted_ts_from_epoch(start_ts)} ({rows_after} remaining)")

    if end_ts is not None:
        rows_before = df.shape[0]
        df = df[df['_timestamp'] <= end_ts]
        rows_after = df.shape[0]
        diff = rows_before - df.shape[0]
        if verbose and diff > 0:
            print(f"\t\tDropped {diff} rows from after {formatted_ts_from_epoch(end_ts)} ({rows_after} remaining)")

    return df