import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import colorsys


def adjust_lightness(color, factor):
    """Adjust the lightness of the given color by the provided factor."""
    rgb = mcolors.to_rgb(color)  # Convert the color to RGB
    hls = colorsys.rgb_to_hls(*rgb)  # Convert RGB to HLS
    adjusted_hls = (hls[0], max(0, min(1, hls[1] * factor)), hls[2])  # Adjust lightness (H, L, S)
    return colorsys.hls_to_rgb(*adjusted_hls)  # Convert back to RGB


def plot_metric(
    df, 
    metric='rewards', 
    suffixes=['old', 'new'], 
    uids=None,
    map_uids_to_colors=True,
    figure=None,
    label_suffix=None,
    legend_loc='best'):

    if not figure:
        plt.figure(figsize=(14, 6))
    else:
        plt.figure(figure.number)

    if map_uids_to_colors:
        # Create a color map to assign similar colors to the same miner_uids
        color_map = plt.get_cmap('tab10') 
        color_map = {uid: color_map(i / len(uids)) for i, uid in enumerate(sorted(uids))}
        uid_to_color = {
            'old': color_map,
            'new': {uid: adjust_lightness(color, .9) for uid, color in color_map.items()}
        }

    for suffix in suffixes:
        col_name = '_'.join([metric, suffix])

        miner_rewards = {}
        for i, row in df.iterrows():
            timestamp = row['_timestamp']
            miner_uids = row['miner_uid']
            rewards = row[col_name]

            for miner_uid, reward in zip(miner_uids, rewards):
                if uids is not None and miner_uid not in uids:
                    continue
                if miner_uid not in miner_rewards:
                    miner_rewards[miner_uid] = []
                miner_rewards[miner_uid].append((timestamp, reward))
    
            # For miner_uids that are missing in this timestamp, carry forward the previous reward
            for miner_uid in miner_rewards:
                if miner_uid not in miner_uids:
                    # Get the last known reward for this miner_uid
                    last_timestamp, last_reward = miner_rewards[miner_uid][-1]
                    miner_rewards[miner_uid].append((timestamp, last_reward))

        for miner_uid, rewards_data in miner_rewards.items():
            timestamps, rewards = zip(*rewards_data)
            label = f"Miner {miner_uid} [{suffix}]" + ("" if label_suffix is None else f" [{label_suffix}]")
            plt.plot(timestamps, rewards, label=label, color=uid_to_color[suffix][miner_uid] if map_uids_to_colors else None)

    if figure:
        return plt.gcf()

    plt.xlabel("Timestamp")
    plt.ylabel(metric.capitalize())
    plt.title(f"Miner {metric.capitalize()} Over Time")
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, loc=legend_loc)
    plt.show();


def plot_multi_validator_metric(    
    vali_dfs, 
    idx_range=None,
    metric='rewards', 
    suffixes=['old', 'new'], 
    uids=None,
    map_uids_to_colors=True,
    legend_loc='best'):
    
    plt.figure(figsize=(14, 6))
    fig = plt.gcf()
    for vali, df in vali_dfs.items():
        if idx_range:
            df = df.iloc[idx_range[0]:idx_range[1]]
        fig = plot_metric(
            df, 
            metric=metric,
            suffixes=suffixes,
            uids=uids,
            map_uids_to_colors=map_uids_to_colors,
            figure=fig,
            label_suffix=vali)

    plt.xlabel("Timestamp")
    plt.ylabel(metric.capitalize())
    plt.title(f"Miner {metric.capitalize()} Over Time")
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, loc=legend_loc)
    plt.show();    
