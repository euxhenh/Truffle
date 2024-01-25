"""Utility functions for paths and trajectories.
"""
import copy
from collections import Counter
from itertools import pairwise

import networkx as nx
import numpy as np


def collapse_array(arr):
    """Collapses consecutive equal elements to a single one.
    E.g., [1, 1, 1, 3, 3, 2, 5, 5] to [1, 3, 2, 5].
    """
    if len(arr) == 0:
        return arr
    arr = np.array(arr)
    diff = np.diff(arr).astype(bool)
    diff = np.insert(diff, 0, True)
    return arr[diff]


def get_all_subsequences(path, min_length=2, unique=True):
    """
    Get all (non_contiguous) subsequences of length at least `min_length`.
    """
    def _all_subsequences(path):
        if len(path) == 1:
            return [path, []]
        first_node = path[0]
        rec_seq = _all_subsequences(path[1:])
        cur_seq = []
        for seq in rec_seq:
            cur_seq.append([first_node, *seq])
            cur_seq.append(seq)
        return cur_seq

    ans = [tuple(collapse_array(i)) for i in _all_subsequences(path)]
    if unique:
        ans = set(ans)
    ans = [i for i in ans if len(i) >= min_length]
    return ans


def get_all_subsequences_cont(path, min_length=2, unique=True):
    """
    Get all contiguous subsequences of length at least `min_length`.
    """
    ans = []
    for i in range(len(path)):
        for j in range(i + 1, len(path) + 1):
            ans.append(tuple(collapse_array(path[i:j])))
    if unique:
        ans = set(ans)
    ans = [i for i in ans if len(i) >= min_length]
    return ans


def get_subsequence_counter(counter, min_length=2, cont=False) -> Counter:
    """Gets another counter derived from `counter` that
    also counts subsequences. `counter` is assumed to map a path
    to its count.
    """
    fn = get_all_subsequences_cont if cont else get_all_subsequences
    subs_counter = Counter()
    for path, count in counter.items():
        all_subs = fn(path, min_length)
        for sub in all_subs:
            subs_counter[tuple(sub)] += count
    return subs_counter


def get_trajectory(G, K, src, trg):
    """Return an array of nodes visited on the trajectory from src to trg.
    """
    g = G[K]
    nodes = [src]
    while src != trg:
        transit = g[src].nonzero()[0]
        if len(transit) != 1:
            return []
        src = transit[0]
        nodes.append(src)
    return np.asarray(nodes)


def get_trajectories(G, commodities, labels, scores=None):
    """Return a list of dicts containing the path for every commodity.

    Parameters
    __________
    G: array of shape (nodes, nodes, commodities)
        The shortest edge-disjoint path matrix.
    commodities: Dict[int, Tuple(node, node)]
    labels: DataFrame
        Consists of cluster labels.
    """
    trajectories = {
        K: {
            "path": get_trajectory(G, K, src, trg),
            "src": src,
            "trg": trg,
        }
        for K, (src, trg) in commodities.items()
    }
    for K in list(trajectories.keys()):
        val = trajectories[K]
        if len(val['path']) == 0:
            trajectories.pop(K)

    for K, traj in trajectories.items():
        traj["states"] = labels.iloc[traj["path"]].to_numpy()
        traj["collapsed_states"] = collapse_array(traj["states"])
        traj["subjects"] = labels.index[traj["path"]].to_numpy()
        if scores is not None:
            traj['scores'] = scores.iloc[traj["path"]].to_numpy()

    return trajectories


def get_full_trajectories(trajectories, scores=None):
    """Returns a dict of full trajectories.
    """
    src_to_commodity = {val['src']: K for K, val in trajectories.items()}
    trg_to_commodity = {val['trg']: K for K, val in trajectories.items()}
    g = nx.DiGraph()
    g.add_edges_from(zip(src_to_commodity, trg_to_commodity))
    order = list(nx.topological_sort(g))

    full_trajectories = {}

    visited = set()
    for src in order:
        if src not in src_to_commodity or src in visited:
            continue
        visited.add(src)
        K = src_to_commodity[src]
        full_traj = copy.deepcopy(trajectories[K])
        full_traj['subj'] = [full_traj['src'], full_traj['trg']]
        full_traj.pop("src")
        trg = full_traj.pop("trg")
        Klist = [K]
        while trg in src_to_commodity:
            visited.add(trg)
            trg_K = src_to_commodity[trg]
            Klist.append(trg_K)
            trg_traj = trajectories[trg_K]

            keys = ['path', 'states', 'collapsed_states', 'subjects']
            if scores is not None:
                keys.append('scores')
            for key in keys:
                full_traj[key] = np.concatenate((full_traj[key], trg_traj[key][1:]))
            full_traj['subj'].append(trg_traj['trg'])

            trg = trg_traj["trg"]
        full_traj['Ks'] = Klist
        if scores is not None:
            full_traj['subj_scores'] = scores.iloc[full_traj['subj']].to_numpy()
        full_trajectories[K] = full_traj

    return full_trajectories


def pair_counter(counter):
    pair_c = Counter()
    for path, count in counter.items():
        for a, b in pairwise(path):
            pair_c[(a, b)] += count
    return pair_c


def pair_mat(counter, n):
    mat = np.zeros((n, n))
    for pair, count in counter.items():
        mat[pair[0], pair[1]] = count
    return mat
