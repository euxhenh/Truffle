"""
Runs the Truffle algorithm given an adjacency matrix of neighbors,
time point and patient information.
"""
import logging
from collections import Counter
from itertools import chain, combinations, pairwise
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from anndata import AnnData
from grinch.utils.ops import group_indices
from sklearn.utils.validation import check_consistent_length

from .mc_flow import multicommodity_flow
from .utils.path_utils import (
    get_full_trajectories, get_trajectories, pair_counter, pair_mat
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Truffle:
    """Trajectory inference using multi-commodity flow with node constraints.

    :param edge_capacity: Maximum flow capacity for an edge. Typically set to 1.
    :param node_capacity: Maximum flow that can pass through a node.
        If not None, should be at least 2 if most patients have multiple visits.
    :param max_path_len: Maximum path length to use for simplifying problem. If None,
        will set equal to the maximal shortest path length.
    """

    def __init__(self, max_path_len: int | str | None = 'auto'):
        self.max_path_len = max_path_len
        self.commodities_: Dict[int, Tuple[int, int]] = {}
        self.commodity_allowed_edges_: Dict[int, List[Tuple[int, int]]] = {}
        self.infeasible_commodities_: List[Tuple[int, int]] = []

    @property
    def n_commodities(self) -> int:
        """Return the number of commodities."""
        return len(self.commodities_)

    def prepare(
        self,
        adata: AnnData,
        adj: str = 'umap_distance',
        subject_id: str = 'subj',
        time_point: str = 'visit',
        time_point_order: str | np.ndarray | None = None,
    ):
        """Runs multi-commodity flow.

        :param adata: AnnData object
        :param adj: square adjacency matrix in obsp
        :param subject_id: subject membership array in obs
        :param time_point: array mapping each sample in subject_id to a time point/visit.
        :param time_point_order: temporal ordering of time_point.
            If None, will sort data in time_point.
        """
        self.n_nodes_ = adata.shape[0]
        self.adj_in_ = adj
        self.subject_id_in_ = subject_id
        self.time_point_in_ = time_point
        self.time_point_order_in_ = time_point_order

        adj = adata.obsp[adj]
        subject_id = adata.obs[subject_id].to_numpy()
        time_point = adata.obs[time_point].to_numpy()
        if time_point_order is None:
            time_point_order = np.unique(time_point)  # sorts
        elif isinstance(time_point_order, str):
            time_point_order = adata.uns[time_point_order]

        check_consistent_length(subject_id, time_point, adj)
        assert np.in1d(time_point, time_point_order).all()

        # Construct networkx graph
        edge_index = adj.nonzero()
        self.edge_weight = np.asarray(adj[edge_index]).ravel()
        self.edge_index = list(zip(*edge_index))
        logger.info(f"Graph has a total of {len(self.edge_index)} directed edges.")

        self.graph_ = nx.from_edgelist(self.edge_index, create_using=nx.DiGraph)

        # For each temporal step, construct the corresponding commodity
        self.commodities_ = self.init_commodities(subject_id, time_point, time_point_order)
        logger.info(
            f"Found a total of {self.n_commodities} commodities "
            f"across {np.unique(subject_id).size} subjects "
            f"and {np.unique(time_point).size} timepoints."
        )

        # some commodities may not be reachable
        max_sp_len = self.remove_infeasible_commodities(self.max_path_len)
        self.max_path_len = self.max_path_len if self.max_path_len != 'auto' else max_sp_len
        self.simplify_problem(self.edge_index, self.max_path_len)

    def fit(
        self,
        edge_capacity: int = 1,
        node_capacity: int | None = None,
    ):
        self.edge_capacity = edge_capacity
        self.node_capacity = node_capacity
        # Run MC Flow
        self.model_, self.results_ = multicommodity_flow(
            n_nodes=self.n_nodes_,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            commodities=self.commodities_,
            commodity_allowed_edges=self.commodity_allowed_edges_,
            edge_capacity=edge_capacity,
            node_capacity=node_capacity,
            constraint_commodity_nodes_only=False,
        )
        self.f_ = self.model_.f.get_values()

        logger.info(self.results_)

    @staticmethod
    def init_commodities(ids, visit_type, ord_visits) -> Dict[int, Tuple[int, int]]:
        """Compute all the pairs (src, trg) for each time hop.
        If an intermediate time step is missing, we connect
        its two adjacent time points.
        """
        ids = np.asarray(ids)
        visit_type = np.asarray(visit_type)
        ord_visits = np.asarray(ord_visits)
        assert np.setdiff1d(np.unique(visit_type), ord_visits).size == 0
        argidx = ord_visits.argsort()
        sorted_visits = ord_visits[argidx]
        commodities = []

        patients, groups = group_indices(ids, as_mask=False)
        for _, group in zip(patients, groups):
            if len(group) == 1:
                continue
            visits = visit_type[group]
            searched_idx = np.searchsorted(sorted_visits, visits)
            ord_idx = argidx[searched_idx]

            argidx_ord_idx = ord_idx.argsort()
            time_flow_idx = group[argidx_ord_idx]
            commodities.extend(list(pairwise(time_flow_idx)))

        return dict(enumerate(commodities))

    def remove_infeasible_commodities(self, max_path_len: int) -> int:
        """For sparse directed connectivity graphs it may be possible for
        no paths to exist that connect the source and sink of a commodity. We
        remove these commodities. Also, cache the maximal length of the
        shortest path assigned to a commodity.
        """
        shortest_path_lengths: List[int] = []
        K_to_remove = []

        for K, (a, b) in self.commodities_.items():
            if nx.has_path(self.graph_, a, b):
                plen = nx.shortest_path_length(self.graph_, a, b)
                if isinstance(max_path_len, int) and plen > max_path_len:
                    logger.info(f"Found no path of len <= {max_path_len} from {a} to {b}.")
                    K_to_remove.append(K)
                    self.infeasible_commodities_.append((a, b))
                else:
                    shortest_path_lengths.append(plen)
            else:
                logger.info(f"Found no path from {a} to {b}.")
                K_to_remove.append(K)
                self.infeasible_commodities_.append((a, b))

        for K in K_to_remove:
            self.commodities_.pop(K)

        # renumber commodities
        self.commodities_ = dict(enumerate(self.commodities_.values()))

        max_sp_len = max(shortest_path_lengths)
        logger.info(f"Minimal path length for all src to reach trg: {max_sp_len}")
        logger.info(f"Found {len(self.infeasible_commodities_)} infeasible commodities.")
        return max_sp_len

    def simplify_problem(self, edge_index, max_path_len):
        """Removes paths that are 'far away'"""
        logger.info(f"Using {max_path_len=}")

        if max_path_len is not None:
            # for every commodity compute all the paths of length <= max_path_len
            commodity_allowed_paths: Dict[int, List[List[int]]] = {
                K: list(nx.all_simple_paths(self.graph_, a, b, cutoff=max_path_len))
                for K, (a, b) in self.commodities_.items()
            }
            self.commodity_allowed_edges_.clear()
            # Convert all the allowed paths to single edges for every commodity.
            # E.g., given the path [1, 2, 3, 4], it will get converted to the list
            # [(1, 2), (2, 3), (3, 4)].
            for K, allowed_paths in commodity_allowed_paths.items():
                allowed_edges = set(chain(*list(map(pairwise, allowed_paths))))
                self.commodity_allowed_edges_[K] = list(allowed_edges)
        else:
            self.commodity_allowed_edges_ = {K: edge_index for K in self.commodities_}

    def get_state_diagram(
        self,
        adata: AnnData,
        cluster: str = 'leiden',
        scores: str | None = None,
        prune_q: float = 0.5,
    ):
        """Given an MC solution, construct the state diagram.

        :param adata: AnnData
        :param cluster: should point to an array of cluster memberships
        :param scores: A key pointing to an array of scores (e.g., ACT scores). If data
            has no scores, set to visit or any other variable that can be tracked in time.
        :param prune_q: Will remove edges so that each node has at least this fraction
            of its total outgoing edge weight.
        :return: (state_diagram, pruned_state_digram,
                  trajectories, full_trajectories,
                  initial_states, final_states)
        """
        G = self.G_from_f(self.f_, self.n_commodities, self.n_nodes_)
        scores = scores if scores is None else adata.obs[scores]

        trajectories = get_trajectories(
            G, self.commodities_, labels=adata.obs[cluster], scores=scores
        )
        full_trajectories = get_full_trajectories(trajectories, scores=scores)
        full_counter = Counter(
            [tuple(tr['collapsed_states']) for tr in full_trajectories.values()]
        )
        _pc = pair_counter(full_counter)
        n_clusters = len(np.unique(adata.obs[cluster]))
        state_diagram = pair_mat(_pc, n_clusters)
        pruned_state_diagram = self.prune_graph(state_diagram, prune_q)

        _initial_states = [tr['collapsed_states'][0] for tr in full_trajectories.values()]
        initial_states, initial_counts = np.unique(_initial_states, return_counts=True)
        _final_states = [tr['collapsed_states'][-1] for tr in full_trajectories.values()]
        final_states, final_counts = np.unique(_final_states, return_counts=True)

        return {
            'state_diagram': state_diagram,
            'pruned_state_diagram': pruned_state_diagram,
            'trajectories': trajectories,
            'full_trajectories': full_trajectories,
            'initial_states': dict(zip(initial_states, initial_counts)),
            'final_states': dict(zip(final_states, final_counts)),
        }

    @staticmethod
    def get_top_paths(state_diagram, initial_states, final_states, n_walks: int = 50000):
        """Return the top occurring paths from a state diagram.

        :param state_diagram: Square state diagram of transitions
        :param initial_states: dictionary mapping a state to the number of times it appears
            as an initial state
        :param final_states: same as above for final states
        :return: list of top paths ranked by weight
        """

        def random_walk(norm_state_diagram, init_p, final_p):
            """Helper function to perform a single random walk. Since we know the entire
            state diagram, it may be easier to compute the top paths directly, but this
            is easier to code right now."""
            n_nodes = norm_state_diagram.shape[0]
            path = []
            start = np.random.choice(n_nodes, p=init_p)
            path.append(start)

            while True:
                next_node = np.random.choice(n_nodes, p=norm_state_diagram[start])
                if next_node in path:
                    break
                path.append(next_node)
                if np.random.binomial(1, p=final_p[next_node]):  # stop or continue
                    break
                start = next_node
            return tuple(path)

        norm_state_diagram = state_diagram / state_diagram.sum(1)[:, None]
        n_nodes = state_diagram.shape[0]
        init_p = np.asarray([initial_states.get(i, 0) for i in range(n_nodes)], dtype=float)
        init_p /= init_p.sum()
        final_counts = np.asarray([final_states.get(i, 0) for i in range(n_nodes)], dtype=float)
        final_p = final_counts / (state_diagram.sum(1) + final_counts)

        walks = Counter()
        for _ in range(n_walks):
            walks[random_walk(norm_state_diagram, init_p, final_p)] += 1
        return walks

    @staticmethod
    def top_paths_from_state_diagram(
        state_diagram: np.ndarray,
        cutoff: int = 4,
        scale_by_length: bool = True,
    ):
        """Recover the top weighted paths from state diagram, scaled by length of path."""
        graph = nx.from_numpy_array(state_diagram, create_using=nx.DiGraph)

        all_paths = []
        n_nodes = state_diagram.shape[0]

        for (a, b) in combinations(range(n_nodes), 2):
            # ordered combinations
            all_paths.extend(list(nx.all_simple_paths(graph, a, b, cutoff=cutoff)))
            all_paths.extend(list(nx.all_simple_paths(graph, b, a, cutoff=cutoff)))

        paths_to_w = {}
        for path in all_paths:
            path_weight = nx.path_weight(graph, path, 'weight')
            if scale_by_length:
                path_weight /= len(path)
            paths_to_w[tuple(path)] = path_weight
        return Counter(paths_to_w)

    @staticmethod
    def G_from_f(f, n_commodities, n_nodes):
        edges = {k: v for k, v in f.items() if v != 0}  # take nonzero edges only
        edge_index = np.vstack(list(edges.keys()))
        edge_weight = np.asarray(list(edges.values()))

        G = np.zeros((n_commodities, n_nodes, n_nodes))
        x, y, z = edge_index.T
        G[x, y, z] = edge_weight

        return G

    @staticmethod
    def prune_graph(A: np.ndarray, prune_q: float = 0.5):
        """For each node, select the top edges that amount to at least q * 100%
        of its total outgoing edge weight.
        """
        assert 0 < prune_q <= 1
        A = A.copy().astype(float)
        for node in range(A.shape[0]):
            row = A[node]
            if row.sum() == 0:
                continue
            row = row[np.argsort(row)[::-1]]
            fractions = np.cumsum(row) / row.sum()
            # Stop at the first index that exceeds trim_q
            threshold = row[(fractions > prune_q).argmax()]
            A[node, A[node] < threshold] = 0
        return A

    def state_dict(self) -> Dict[str, Any]:
        """List of attribute names to be dumped."""
        state = {
            'max_path_len': self.max_path_len,
            'commodities_': self.commodities_,
            'commodity_allowed_edges_': self.commodity_allowed_edges_,
            'infeasible_commodities_': self.infeasible_commodities_,
        }

        for attr in ['n_nodes_', 'graph_', 'results_', 'f_',
                     'edge_capacity', 'node_capacity', 'adj_in_',
                     'subject_id_in_', 'time_point_in_', 'time_point_order_in_']:
            if hasattr(self, attr):
                state[attr] = getattr(self, attr)

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load object from state dict"""
        for key, attr in state_dict.items():
            setattr(self, key, attr)

    @classmethod
    def from_state_dict(cls, state_dict):
        obj = cls(max_path_len=state_dict['max_path_len'])
        obj.load_state_dict(state_dict)
        return obj
