from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython.display import HTML, display
from matplotlib.colors import to_rgba
from tqdm.auto import tqdm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import bittensor as bt
import seaborn as sns
import numpy as np
import colorsys

from sim.incentive import assemble_W, assemble_S, compute_incentive


def animate_timeseries(
    X, 
    highlight_uids=[], 
    fixed_axes=True,
    figsize=(12, 8)):
    """
    """
    def get_timestep(frame):
        x = X[frame]
        sorted_pairs = sorted(zip(x, range(len(x))))
        y_sorted, sorted_indices = zip(*sorted_pairs)
        return y_sorted, sorted_indices

    def update(frame):
        color_map = np.array([[0.8, 0.8, 0.8, 1.0]] * 256) 
        y, sorted_indices = get_timestep(frame)
        for uid in highlight_uids:
            color_map[sorted_indices.index(uid)] = highlight_color_map[uid]
        scatter.set_offsets(np.column_stack((range(len(y)), y)))
        scatter.set_color(color_map)
        if not fixed_axes:
            ax.set_ylim(y_min, max(y) * 1.1)
        ax.set_title(f"Miner Incentives (Timestep: {frame})", fontsize=20, pad=20)
        return scatter,

    highlight_color_map = {
        uid: c for uid, c in zip(
            highlight_uids, 
            plt.cm.viridis(np.linspace(0, 1, len(highlight_uids)))
        )
    }

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    y, sorted_indices = get_timestep(0)
    color_map = np.array([[0.8, 0.8, 0.8, 1.0]] * 256) 
    for uid in highlight_uids:
        color_map[sorted_indices.index(uid)] = highlight_color_map[uid]
    scatter = ax.scatter(range(len(y)), y, c=color_map, s=50)

    y_min = 0 - np.min(np.nonzero(X)) * 0.9
    if fixed_axes:
        y_max = np.max(X) * 1.1

    ax.set_ylim(y_min, y_max if fixed_axes else max(y) * 1.1)
    ax.set_xlim(-5, len(y) + 5)
    ax.set_xticks(range(0, len(y) + 1, 50))
    ax.set_xticklabels(range(0, len(y), 50), rotation=45, ha='right')
    ax.set_title("Miner Incentives (Timestep: 0)", fontsize=20, pad=20)

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)

    anim = FuncAnimation(fig, update, frames=len(X), interval=200, blit=True)
    plt.close(fig)
    display(HTML(anim.to_jshtml()))


def plot_incentive_over_time(
    scored_validator_dfs,
    validator_uids,
    weight_column='weights_BinaryReward',
    netuid=34,
    highlight_uids=[]):
    """
    TODO option to have a fixed xlim and ylim
    TODO option to overlay multiple incentives over time
    """

    def get_new_I(
        timestep,
        validator_dfs,
        validator_uids,
        weight_column,
        metagraph=None):
        """
        TODO log stake to w&b for dynamic stake values instead
        of relying on current metagraph
        """
        S = assemble_S(validator_uids, metagraph=metagraph)
        W = assemble_W(validator_dfs, weight_column, idx=timestep)
        return compute_incentive(W, S)

    def update(frame):
        I = get_new_I(
            frame,
            scored_validator_dfs,
            validator_uids,
            weight_column,
            metagraph)
        
        sorted_pairs = sorted(zip(I, range(len(I))))
        y_sorted, sorted_indices = zip(*sorted_pairs)

        #color_map = np.array([[0.0, 1.0, 1.0, 1.0]] * 256) 
        color_map = np.array([[0.8, 0.8, 0.8, 1.0]] * 256) 
        for uid in highlight_uids:
            color_map[sorted_indices.index(uid)] = highlight_color_map[uid]

        scatter.set_offsets(np.column_stack((range(len(y_sorted)), y_sorted)))
        scatter.set_color(color_map)

        ax.set_ylim(0, max(I) * 1.1)
        ax.set_xlim(-5, len(y_sorted) + 5)
        ax.set_xticks(range(0, len(y_sorted) + 1, 50))
        ax.set_xticklabels(range(0, len(y_sorted), 50), rotation=45, ha='right')
        ax.set_title(f"Miner Incentives (Timestep: {frame})", fontsize=20, pad=20)
        progress_bar.update(1)
        return scatter,

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set_style("whitegrid")

    metagraph = bt.metagraph(netuid=netuid)

    # Initialize with first timestep data
    I = get_new_I(
        0,
        scored_validator_dfs,
        validator_uids,
        weight_column,
        metagraph)


    sorted_pairs = sorted(zip(I, range(len(I))))
    y_sorted, sorted_indices = zip(*sorted_pairs)

    color_map = np.array([[0.8, 0.8, 0.8, 1.0]] * 256) 
    highlight_color_map = {
        uid: c for uid, c in zip(
            highlight_uids, 
            plt.cm.viridis(np.linspace(0, 1, len(highlight_uids)))
        )
    }
    for uid in highlight_uids:
        color_map[sorted_indices.index(uid)] = highlight_color_map[uid]

    scatter = ax.scatter(range(256), y_sorted, c=color_map, s=50)
    
    ax.set_title("Miner Incentives", fontsize=20, pad=20)
    ax.set_xlabel("Miner", fontsize=14, labelpad=10)
    ax.set_ylabel("Incentive", fontsize=14, labelpad=10)
    ax.set_ylim(0, max(I) * 1.1)
    ax.set_xlim(-5, len(I) + 5)
    ax.set_xticks(range(0, len(I), 50))
    ax.set_xticklabels(range(0, len(I), 50), rotation=45, ha='right')
    plt.tight_layout()

    steps = min([len(df) for vali, df in scored_validator_dfs.items()])
    progress_bar = tqdm(total=steps, desc="Generating Animation")
    try:
        anim = FuncAnimation(fig, update, frames=steps, interval=200, blit=True)
        plt.close(fig)
    finally:
        progress_bar.close()  # Ensure the progress bar is closed after completion

    display(HTML(anim.to_jshtml()))

def plot_incentives(I_dict):

    plt.figure(figsize=(14, 6))
    fig = plt.gcf()

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i/len(I_dict)) for i in range(len(I_dict))]

    for i, (name, I) in enumerate(I_dict.items()):
        fig = plot_incentive(I, fig, colors[i], name, 0.15)

    sns.set_style("whitegrid")
    plt.title("Miner Incentives", fontsize=20, pad=20)
    plt.xlabel("Miner", fontsize=14, labelpad=10)
    plt.ylabel("Incentive", fontsize=14, labelpad=10)
    plt.ylim(0, max(I) * 1.1)
    plt.xticks(range(0, len(I), 50), list(range(0, len(I), 50)), rotation=45, ha='right')
    plt.legend(loc='best')
    plt.tight_layout()
    
    plt.show()


def plot_incentive(I, figure=None, color='#1f77b4', label=None, linewidth=0.5):

    if not figure:
        plt.figure(figsize=(12, 8))
    else:
        plt.figure(figure.number)

    sns.scatterplot(
        x=range(len(I)),
        y=sorted(I[I.nonzero()]),
        s=100,
        color=color,
        edgecolor='white',
        linewidth=linewidth,
        label=label
    )

    if figure:
        return plt.gcf()

    sns.set_style("whitegrid")
    plt.title("Miner Incentives", fontsize=20, pad=20)
    plt.xlabel("Miner", fontsize=14, labelpad=10)
    plt.ylabel("Incentive", fontsize=14, labelpad=10)
    plt.ylim(0, max(I) * 1.1)
    plt.xticks(range(0, len(I), 50), list(range(0, len(I), 50)), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


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
