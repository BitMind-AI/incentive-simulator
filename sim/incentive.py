import numpy as np
import bittensor as bt


def compute_incentive(W, S):
    incentive = np.dot(W.T, S)
    incentive /= np.sum(incentive)
    return incentive


def assemble_W(data_dict, W_column, N=256, idx=-1):
    W = []
    for vali, df in data_dict.items():
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
    