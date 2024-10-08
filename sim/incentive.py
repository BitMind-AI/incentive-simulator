import numpy as np
import bittensor as bt


def compute_incentive(W, S):
    incentive = np.dot(W.T, S)
    incentive /= np.sum(incentive)
    return incentive


def compute_all_incentives(
    weight_history_dfs, 
    weight_column, 
    n_timesteps=None, 
    netuid=34, 
    metagraph=None):
    """  """
    
    I = []
    if n_timesteps is None:
        n_timesteps = min([len(df) for _, df in weight_history_dfs.items()])
    
    vali_uids = np.array([int(vali.split('-')[1]) for vali in weight_history_dfs.keys()])
    S = assemble_S(vali_uids, netuid, metagraph)
    for timestep in range(n_timesteps):
        W = assemble_W(weight_history_dfs, weight_column, idx=timestep)
        I.append(compute_incentive(W, S))
        
    return np.array(I)
    

def assemble_W(weight_history_dfs, W_column, N=256, idx=-1):
    W = []
    for vali, df in weight_history_dfs.items():
        weight_dict = df[W_column].iloc[idx]
        uids = list(weight_dict.keys())
        W_ = np.zeros(N)
        W_[uids] = [weight_dict[uid] for uid in uids]
        W.append(W_)
    return np.vstack(W)


def assemble_S(vali_uids, netuid=34, metagraph=None):
    if not metagraph:
        metagraph = bt.metagraph(netuid=netuid)
    return metagraph.S[vali_uids]

    