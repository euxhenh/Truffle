import re
from functools import partial
from typing import Dict, List, Tuple

import anndata
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from grinch import Filter, StackedFilter
from scipy.cluster.hierarchy import leaves_list, linkage


def GSEA_pivot(
    data: Dict[str, pd.DataFrame],
    c_filter: Filter | StackedFilter,
    regex: str | None = r' \([^)]*\)',
    term_char_limit: int | None = 50,
    pivot_values: List[str] = ['NES', 'FWER p-val'],
):
    """Plot a heatmap of significant terms given a
    dictionary of clusters to GSEA results.
    """
    terms_to_show = []
    for df in data.values():
        terms_to_show.extend(df[c_filter(df)].Term.tolist())
    terms_to_show = np.unique(terms_to_show)

    df = pd.concat(data.values(), keys=data.keys())
    df = df.reset_index().rename(columns={'level_0': 'ID', 'level_1': 'Old Index'})
    df = df[np.in1d(df['Term'], terms_to_show)]

    def prep_term(term):
        if regex is not None:
            term = re.sub(regex, '', term)
        if term_char_limit is None:
            return term
        if len(term) > term_char_limit:
            term = term[:term_char_limit] + '...'
        return term

    if regex is not None:
        df['Term'] = [prep_term(term) for term in df['Term']]

    piv = pd.pivot_table(
        df,
        values=pivot_values,
        index=["Term"],
        columns=["ID"],
        fill_value=0,
    )

    return df, piv


def pivot_to_clustermap(
    piv_mask,
    hue: str = 'NES',
    asterisk: str = 'FWER p-val',
    figsize: Tuple[int, int] = (10, 18),
    cmap: str = 'viridis',
    savepath: str | None = None,
    **kwargs,
):
    """Convert a pivot table to a clustermap.
    """
    hue, asterisk = piv_mask[hue], piv_mask[asterisk]

    _mask = asterisk.to_numpy()
    sig = np.where(_mask <= 0.05, '*', '').astype('U2')
    sig[_mask == 0] = '**'
    sig = pd.DataFrame(sig, index=asterisk.index, columns=asterisk.columns)

    g = sns.clustermap(
        hue,
        figsize=figsize,
        yticklabels=1,
        col_cluster=False,
        cbar_pos=(0, .2, .03, .4),
        cmap=cmap,
        center=0,
        annot=sig,
        fmt='s',
        **kwargs,
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.ax_col_dendrogram.remove()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()


def merge_prerank(
    adata: anndata.AnnData,
    key: str = 'gsea_prerank',
    qval: float = 0.05,
    pval_correction: str = 'FWER p-val',
    topk: int = 10,
) -> pd.DataFrame:
    """Merges dataframes from all clusters.
    """
    cl_dict = adata.uns[key]
    df_list = []
    for cl, df in cl_dict.items():
        if len(df) == 0:
            df_list.append(df)
            continue

        df = df[df[pval_correction] <= qval]

        # Assumes df is sorted by FWER
        if hasattr(df, 'NES'):
            top_df = df[df.NES > 0][:topk]
            bot_df = df[df.NES < 0][:topk]
            df = pd.concat([top_df, bot_df])
        else:
            df = df[:topk]
        df_list.append(df)
        df['cluster'] = cl
    return pd.concat(df_list)


def order_samples(mat, x):
    Z = linkage(mat)
    leaves = leaves_list(Z)
    return mat[leaves], x[leaves]


def map_to_2d(xs, ys, ws):
    unq_x = np.unique(xs)
    unq_y = np.unique(ys)
    x_to_i = {x: i for i, x in enumerate(unq_x)}
    y_to_j = {y: j for j, y in enumerate(unq_y)}
    mat = np.zeros((len(unq_x), len(unq_y)))
    for x, y, w in zip(xs, ys, ws):
        mat[x_to_i[x], y_to_j[y]] = w
    mat, unq_x = order_samples(mat, unq_x)
    xx = []
    for x in unq_x:
        if x.startswith(k := 'GO_Biological_Process_2023__'):
            xx.append(x[len(k):])
        else:
            xx.append(x)
    unq_x = np.asarray(xx)

    return mat, unq_x, unq_y


def prerank_heatmap(
    adata,
    key='gsea_prerank',
    savepath='figs/prerank.png',
    qval=0.05,
    figsize=(8, 8),
    topk=10,
    pval_correction="FWER p-val",
    order='NES',
):
    df = merge_prerank(
        adata, key=key,
        qval=qval, pval_correction=pval_correction,
        topk=topk,
    )
    mat, x, y = map_to_2d(df.Term, df.cluster, getattr(df, order))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mat, mask=mat == 0, cmap='viridis', ax=ax)
    ax.set_xticks(np.arange(len(y)) + 0.5)
    ax.set_yticks(np.arange(len(x)) + 0.5)
    ax.set_xticklabels(y, ha='right')
    ax.set_yticklabels(x)
    ax.tick_params('y', rotation=0)
    ax.tick_params('x', rotation=20)
    ax.set_title('NES scores')
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()


def draw_A_as_graph(
        A, ax=None, pos=None, labels=None, nodes=None,
        whiten_nodes=None, node_size=1000,
):
    """Draws the graph given by the adjacency matrix A."""
    src, trg = A.nonzero()
    A_norm = A / A.max()
    weight = A_norm[src, trg]
    g = nx.DiGraph()
    if nodes is not None:
        g.add_nodes_from(nodes)
    for x, y, w in zip(src, trg, weight):
        g.add_edge(x, y, weight=w)
    if pos is None:
        pos = nx.shell_layout(g)

    if whiten_nodes is None:
        node_color = '#023047'
    else:
        assert nodes is not None
        node_color = np.asarray(['#023047'] * len(nodes))
        node_color[whiten_nodes] = '#CCCCCC'

    if labels is None:
        labels = {node: node for node in g.nodes}

    nx.draw(g, pos, ax=ax, edgelist=[])

    nx.draw_networkx_nodes(
        g, pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
    )

    nodes = np.arange(A.shape[0])
    if whiten_nodes is not None:
        nodes = np.delete(nodes, whiten_nodes)

    nx.draw_networkx_labels(
        g, pos,
        labels={node: labels[node] for node in nodes},
        font_color='#ffb703',
        font_weight='bold',
        font_size=20,
        ax=ax,
    )
    if whiten_nodes is not None:
        nx.draw_networkx_labels(
            g, pos,
            labels={node: labels[node] for node in whiten_nodes},
            font_color='#FFFFFF',
            font_weight='bold',
            font_size=20,
            ax=ax,
        )

    A_bin = A > 0
    draw_edge_fn = partial(
        nx.draw_networkx_edges,
        ax=ax,
        pos=pos,
        node_size=node_size,
        arrowstyle='simple',
        arrowsize=20,
    )

    no_bend = list(zip(*((A_bin != A_bin.T) * A_bin).nonzero()))
    no_bend.extend(list(zip(*(np.triu(A_bin == A_bin.T) * A_bin).nonzero())))
    bend = list(zip(*(np.tril(A_bin == A_bin.T) * A_bin).nonzero()))
    for a, b in bend.copy():
        if a == b:
            bend.remove((a, b))
            no_bend.remove((a, b))
    draw_edge_fn(
        g, edgelist=no_bend,
        alpha=[A_norm[e[0], e[1]] for e in no_bend],
        width=[A_norm[e[0], e[1]] for e in no_bend],
        arrowsize=40,
    )
    draw_edge_fn(
        g, edgelist=bend,
        alpha=[A_norm[e[0], e[1]] for e in bend],
        width=[A_norm[e[0], e[1]] for e in bend],
        connectionstyle='arc3, rad = 0.2',
        arrowsize=40,
    )
    return pos
