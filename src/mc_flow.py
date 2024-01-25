"""Multicommodity flow linear program and helper functions.
"""
from typing import Dict, Hashable, List, Tuple

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def multicommodity_flow(
    n_nodes: int,
    edge_index: List[Tuple[int, int]],
    edge_weight: List[float] | np.ndarray,
    commodities: Dict[Hashable, Tuple[int, int]],
    commodity_allowed_edges: Dict[Hashable, List[Tuple[int, int]]] | None = None,
    edge_capacity: float = 1.0,
    node_capacity: float | None = None,
    constraint_commodity_nodes_only: bool = False,
    sense: str = 'minimize',
    solver: str = 'glpk',
) -> Tuple:
    """Given a flow network defined by `edge_index` and `edge_weight`,
    set the capacity of each edge to `max_edge_capacity`. Given K
    commodities as tuples (src, trg) where src is a source that supplies
    unit flow, and trg is a sink that demands unit flow, find the optimal
    flow function f_i for every commodity which assigns {0, 1} to each
    edge, such that the cost of all edges (u, v) with f_i(u, v) = 1 is
    minimized. The cost of each edge is given by edge_weight.

    Optionally, can specify max node capacity which will only allow a node
    to be visited this many times.

    Parameters
    __________
    commodity_allowed_paths: List[List[int]]
        A list of paths that a commodity can travel through. Useful to
        speed up solvers if the problem is too large. If None, will assume
        all edges are allowed and will initialize as edge_index.
    """
    assert len(edge_weight) == len(edge_index)
    assert max(max(ei[0], ei[1]) for ei in edge_index) < n_nodes
    assert max(max(K[0], K[1]) for K in commodities.values()) < n_nodes
    if commodity_allowed_edges is None:
        commodity_allowed_edges = {K: edge_index for K in commodities}
    assert len(commodity_allowed_edges) == len(commodities)

    model = pyo.ConcreteModel()

    # Adds n_nodes, nodes, edges, nodes_in, nodes_out, cost
    add_graph(model, n_nodes, edge_index, edge_weight)
    # Adds commodity_keys, commodities, commodity_allowed_edges,
    # commodity_transit_nodes, edge_to_commodities
    add_commodities(model, commodities, commodity_allowed_edges)

    def commodity_allowed_edge_pairs(model):
        for K, allowed_edges in model.commodity_allowed_edges.items():
            for u, v in allowed_edges:
                yield (K, u, v)

    model.f = pyo.Var(
        pyo.Set(initialize=commodity_allowed_edge_pairs),
        domain=pyo.Binary,
        doc='All (K, u, v) pairs for commodities and allowed edges',
    )

    # Adds edge_capacity, transit_flow_conservation, source_flow_conservation,
    # target_flow_conservation
    add_constraints(model, edge_capacity)
    # Optionally adds node_out_capacity, node_in_capacity
    add_node_constraints(model, node_capacity, constraint_commodity_nodes_only)

    # Define the objective as the cost of picking an edge times the flow
    # flowing through it.
    def obj(model):
        target = 0.0
        for u, v in model.edges:
            edge_flow = 0.0
            for K in model.commodity_keys:
                if (K, u, v) in model.f:
                    edge_flow += model.f[K, u, v]
            target += model.cost[u, v] * edge_flow
        return target

    sense = getattr(pyo, sense)
    model.objective = pyo.Objective(rule=obj, sense=sense)

    # -------- Solve ---------
    opt = SolverFactory(solver)
    print("Begin optimization.")
    results = opt.solve(model)

    return model, results


def add_graph(model, n_nodes, edge_index, edge_weight):
    """Add anything related to nodes and edges.
    """
    # -------- Define sets --------
    model.n_nodes = n_nodes
    model.nodes = pyo.RangeSet(0, n_nodes - 1)
    print(f"Initialized {len(model.nodes)} nodes.")

    model.edges = pyo.Set(initialize=edge_index, within=model.nodes * model.nodes)
    print(f"Initialized {len(model.edges)} edges.")

    # Define the adjacency lists for each node in sparse format.
    def nodes_out(model, node):
        for edge in model.edges:
            if edge[0] == node:
                yield edge[1]

    model.nodes_out = pyo.Set(model.nodes, within=model.nodes,
                              initialize=nodes_out, doc='Children of node')

    def nodes_in(model, node):
        for edge in model.edges:
            if edge[1] == node:
                yield edge[0]

    model.nodes_in = pyo.Set(model.nodes, within=model.nodes,
                             initialize=nodes_in, doc='Parents of node')

    edge_index_to_weight = dict(zip(edge_index, edge_weight))

    def cost_fn(_, u, v):
        return edge_index_to_weight[(u, v)]

    model.cost = pyo.Param(
        model.edges,
        initialize=cost_fn,
        within=pyo.NonNegativeReals,
        doc="The cost of picking each edge. Typically init'ed as a distance."
    )


def add_commodities(model, commodities, commodity_allowed_edges):
    """Adds commodities and all helper sets.
    """
    model.commodity_keys = pyo.Set(
        initialize=commodities.keys(), doc='Commodity index',
    )

    model.commodities = pyo.Set(
        model.commodity_keys,
        initialize=commodities,
        ordered=True,
        doc='Commodity index -> (src, trg)',
    )

    model.commodity_allowed_edges = pyo.Set(
        model.commodity_keys,
        initialize=commodity_allowed_edges,
        doc='Commodity index -> set of (u, v) pairs of allowed edges',
    )

    def commodity_transit_nodes(_, K):
        src, trg = commodities[K]
        allowed_nodes = np.unique(commodity_allowed_edges[K])
        return np.setdiff1d(allowed_nodes, [src, trg])

    model.commodity_transit_nodes = pyo.Set(
        model.commodity_keys,
        initialize=commodity_transit_nodes,
        doc='Commodity index -> set of transit nodes != src and != trg',
    )

    def commodity_transit_node_pairs(model):
        for K, transit_nodes in model.commodity_transit_nodes.items():
            for node in transit_nodes:
                yield (K, node)

    model.commodity_transit_node_pairs = pyo.Set(
        initialize=commodity_transit_node_pairs,
        doc='All (K, node) pairs for commodity K and transit node',
    )


# Some helper functions
def _K_outflow(model, K, node):
    """Compute outflow nodes for given node for commodity K."""
    to_sum = [model.f[K, node, v]
              for v in model.nodes_out[node]
              if (K, node, v) in model.f]
    return to_sum


def _K_inflow(model, K, node):
    """Compute inflow nodes for given node for commodity K."""
    to_sum = [model.f[K, u, node]
              for u in model.nodes_in[node]
              if (K, u, node) in model.f]
    return to_sum


def _K_flow(model, K, node, *, target):
    """Compute total flow nodes for given node for commodity K.
    Return a Constraint of the form `flow`==`target`.
    """
    to_sum_out = _K_outflow(model, K, node)
    to_sum_in = _K_inflow(model, K, node)
    if len(to_sum_out) == 0 and len(to_sum_in) == 0:  # no flow in this node
        return pyo.Constraint.Feasible
    flow = sum(to_sum_out) - sum(to_sum_in)
    return flow == target


def add_constraints(model, max_edge_capacity):
    """Add all edge and node flow related constraints"""
    def edge_capacity(model, u, v):
        to_sum = [model.f[K, u, v]
                  for K in model.commodity_keys
                  if (K, u, v) in model.f]
        if len(to_sum) == 0:  # edge not used
            return pyo.Constraint.Feasible
        return sum(to_sum) <= max_edge_capacity

    model.edge_capacity = pyo.Constraint(
        model.edges,
        rule=edge_capacity,
        doc=('The amount of flow through an edge (for all commodities)'
             ' should not exceed this amount.')
    )

    def transit_flow_conservation(model, K, node):
        return _K_flow(model, K, node, target=0)  # out == in

    model.transit_flow_conservation = pyo.Constraint(
        model.commodity_transit_node_pairs,
        rule=transit_flow_conservation,
        doc=('The amount of flow for commodity K entering a transit'
             ' node must fully exit'),
    )

    def source_flow_conservation(model, K):
        source, _ = model.commodities[K]
        return _K_flow(model, K, source, target=1)  # out == in + 1

    model.source_flow_conservation = pyo.Constraint(
        model.commodity_keys,
        rule=source_flow_conservation,
        doc='A flow must fully exit its source node'
    )

    def target_flow_conservation(model, K):
        _, target = model.commodities[K]
        return _K_flow(model, K, target, target=-1)  # out == in - 1

    model.target_flow_conservation = pyo.Constraint(
        model.commodity_keys,
        rule=target_flow_conservation,
        doc='A flow must fully enter its target node'
    )


def add_node_constraints(model, max_node_capacity=None,
                         constraint_commodity_nodes_only=False):
    """Adds inflow and outflow node constraints."""
    def _is_commodity_node(model, node) -> bool:
        for K, src_trg in model.commodities.items():
            if node in src_trg:
                return True
        return False

    def _node_capacity(model, node, flow_fn):
        if constraint_commodity_nodes_only and not _is_commodity_node(model, node):
            # If it does not belong to any commodity, don't constrain it
            return pyo.Constraint.Feasible

        to_sum = []
        for K in model.commodity_keys:
            to_sum.extend(flow_fn(model, K, node))
        if len(to_sum) <= max_node_capacity:  # assuming binary f
            return pyo.Constraint.Feasible
        out_flow = sum(to_sum)
        return out_flow <= max_node_capacity

    def node_out_capacity(model, node):
        return _node_capacity(model, node, _K_outflow)

    def node_in_capacity(model, node):
        return _node_capacity(model, node, _K_inflow)

    if max_node_capacity is not None:
        model.node_out_capacity = pyo.Constraint(
            model.nodes,
            rule=node_out_capacity,
            doc='Max amount of flow that can exit a node.'
        )
        model.node_in_capacity = pyo.Constraint(
            model.nodes,
            rule=node_in_capacity,
            doc='Max amount of flow that can enter a node.'
        )


def solution_to_np(
    model: pyo.ConcreteModel,
    n_commodities: int | None = None,
    n_nodes: int | None = None
) -> np.ndarray:
    """Convert the solution to a numpy array of shape
    (n_commodities, n_nodes, n_nodes). G[i] corresponds to the adjacency
    matrix for the i-th commodity and G[i, a, b] is an edge from source
    node a to target node b.

    Parameters
    ----------
    model: ConcreteModel
        Pyomo model. Must have a 'f' attribute corresponding to the
        solution.
    n_commodities: int, None
        Total number of commodities. If None, will infer from solution f.
    n_nodes: int, None
        Total number of nodes. If None, will infer from solution f.

    Returns
    -------
    G: array of shape (n_commodities, n_nodes, n_nodes).
        For each commodity K, the adjacency matrix G[K] describes the
        edges in the path from K's src to K's trg.
    """
    f_dict = model.f.get_values()
    if n_commodities is None:
        n_commodities = max(k[0] for k in f_dict) + 1
    if n_nodes is None:  # assumes connected graph
        n_nodes = max(max(k[1], k[2]) for k in f_dict) + 1
    G = np.zeros((n_commodities, n_nodes, n_nodes))
    for key, val in f_dict.items():
        G[key] = val
    return G
