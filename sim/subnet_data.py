from datetime import datetime, timedelta
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


def epoch_to_datetime(epoch):
    return datetime.utcfromtimestamp(epoch)


def datetime_to_epoch(dt):
    return int(dt.timestamp())


def round_timestamp(timestamp, round_to_minutes=1):
    rounded = timestamp.replace(second=0, microsecond=0)
    if round_to_minutes > 1:
        minutes = (rounded.minute // round_to_minutes) * round_to_minutes
        rounded = rounded.replace(minute=minutes)
    return rounded


def align_dataframes_by_timestamp(data_dict, timestamp_column='_timestamp', round_to_minutes=1):
    """
    Aligns dataframes in the data_dict by their epoch timestamp column, with configurable rounding.
    Ensures all dataframes have the same start and end times, and all rows align perfectly.
    Fills missing data with the most recently available data.
    
    :param data_dict: Dictionary of dataframes
    :param timestamp_column: Name of the timestamp column (containing epoch timestamps)
    :param round_to_minutes: Number of minutes to round timestamps to (1, 5, 10, etc.)
    :return: Dictionary of aligned dataframes
    """
    aligned_dict = {}
    all_timestamps = set()
    
    # First pass: round timestamps and collect all unique timestamps
    for key, df in data_dict.items():
        df = df.copy()
        df['_datetime'] = df[timestamp_column].apply(lambda x: round_timestamp(epoch_to_datetime(x), round_to_minutes))
        df['aligned_timestamp'] = df['_datetime'].apply(datetime_to_epoch)
        df = df.sort_values('aligned_timestamp')
        aligned_dict[key] = df
        all_timestamps.update(df['aligned_timestamp'])
    
    # Determine the common start and end times
    common_start = max(df['aligned_timestamp'].min() for df in aligned_dict.values())
    common_end = min(df['aligned_timestamp'].max() for df in aligned_dict.values())
    
    # Create a complete timestamp index within the common range
    all_timestamps = sorted([t for t in all_timestamps if common_start <= t <= common_end])
    timestamp_index = pd.Index(all_timestamps, name='aligned_timestamp')
    
    # Second pass: reindex all dataframes to the common timestamp index and fill missing data
    for key, df in aligned_dict.items():
        # Reindex the dataframe
        df_reindexed = df.set_index('aligned_timestamp').reindex(timestamp_index)
        
        # Fill missing data with the most recently available data
        df_filled = df_reindexed.ffill()
        
        # Add the 'data_available' column to indicate original vs filled data
        df_filled['data_available'] = df_filled.index.isin(df['aligned_timestamp']).astype(int)
        
        # Reset the index to make 'aligned_timestamp' a column again
        df_filled = df_filled.reset_index()
        
        # Add readable datetime column
        df_filled['_datetime'] = df_filled['aligned_timestamp'].apply(lambda x: epoch_to_datetime(x).strftime('%Y-%m-%d %H:%M:%S'))
        
        # Reorder columns
        cols = df_filled.columns.tolist()
        cols = ['aligned_timestamp', '_datetime', 'data_available'] + [col for col in cols if col not in ['aligned_timestamp', '_datetime', 'data_available', '_datetime']]
        df_filled = df_filled[cols]
        
        aligned_dict[key] = df_filled
    
    print(f"All dataframes aligned to time range: {epoch_to_datetime(common_start)} - {epoch_to_datetime(common_end)}")
    
    return aligned_dict